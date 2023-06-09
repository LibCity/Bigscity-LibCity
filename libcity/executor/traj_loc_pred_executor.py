from ray import tune
import torch
import torch.optim as optim
import numpy as np
import os
from logging import getLogger

from libcity.executor.abstract_executor import AbstractExecutor
from libcity.utils import get_evaluator


class TrajLocPredExecutor(AbstractExecutor):

    def __init__(self, config, model, data_feature):
        self.evaluator = get_evaluator(config)
        if(type(config['topk']) == type(0)):
            self.metrics = 'Recall@{}'.format(config['topk'])
        elif(type(config['topk']) == type([])):
            self.metrics = 'Recall@{}'.format(config['topk'][0])
        self.config = config
        self.model = model.to(self.config['device'])
        self.tmp_path = './libcity/tmp/checkpoint/'
        self.exp_id = self.config.get('exp_id', None)
        self.cache_dir = './libcity/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/evaluate_cache'.format(self.exp_id)
        self.loss_func = None  # TODO: 根据配置文件支持选择特定的 Loss Func 目前并未实装
        self._logger = getLogger()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

    def train(self, train_dataloader, eval_dataloader):
        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)
        metrics = {}
        metrics['accuracy'] = []
        metrics['loss'] = []
        lr = self.config['learning_rate']
        for epoch in range(self.config['max_epoch']):
            self._logger.info('start train')
            self.model, avg_loss = self.run(train_dataloader, self.model,
                                            self.config['learning_rate'], self.config['clip'])
            self._logger.info('==>Train Epoch:{:4d} Loss:{:.5f} learning_rate:{}'.format(
                epoch, avg_loss, lr))
            # eval stage
            self._logger.info('start evaluate')
            avg_eval_acc, avg_eval_loss = self._valid_epoch(eval_dataloader, self.model)
            self._logger.info('==>Eval Acc:{:.5f} Eval Loss:{:.5f}'.format(avg_eval_acc, avg_eval_loss))
            metrics['accuracy'].append(avg_eval_acc)
            metrics['loss'].append(avg_eval_loss)
            if self.config['hyper_tune']:
                # use ray tune to checkpoint
                with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    self.save_model(path)
                # ray tune use loss to determine which params are best
                tune.report(loss=avg_eval_loss, accuracy=avg_eval_acc)
            else:
                save_name_tmp = 'ep_' + str(epoch) + '.m'
                torch.save(self.model.state_dict(), self.tmp_path + save_name_tmp)
            self.scheduler.step(avg_eval_acc)
            # scheduler 会根据 avg_eval_acc 减小学习率
            # 若当前学习率小于特定值，则 early stop
            lr = self.optimizer.param_groups[0]['lr']
            if lr < self.config['early_stop_lr']:
                break
        if not self.config['hyper_tune'] and self.config['load_best_epoch']:
            best = np.argmax(metrics['accuracy'])  # 这个不是最好的一次吗？
            load_name_tmp = 'ep_' + str(best) + '.m'
            self.model.load_state_dict(
                torch.load(self.tmp_path + load_name_tmp))
        # 删除之前创建的临时文件夹
        for rt, dirs, files in os.walk(self.tmp_path):
            for name in files:
                remove_path = os.path.join(rt, name)
                os.remove(remove_path)
        os.rmdir(self.tmp_path)

    def load_model(self, cache_name):
        model_state, optimizer_state = torch.load(cache_name)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

    def save_model(self, cache_name):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        # save optimizer when load epoch to train
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), cache_name)

    def evaluate(self, test_dataloader):
        self.model.train(False)
        self.evaluator.clear()
        for batch in test_dataloader:
            batch.to_tensor(device=self.config['device'])
            scores = self.model.predict(batch)
            if self.config['evaluate_method'] == 'popularity':
                evaluate_input = {
                    'uid': batch['uid'].tolist(),
                    'loc_true': batch['target'].tolist(),
                    'loc_pred': scores.tolist()
                }
            else:
                # negative sample
                # loc_true is always 0
                loc_true = [0] * self.config['batch_size']
                evaluate_input = {
                    'uid': batch['uid'].tolist(),
                    'loc_true': loc_true,
                    'loc_pred': scores.tolist()
                }
            self.evaluator.collect(evaluate_input)
        self.evaluator.save_result(self.evaluate_res_dir)

    def run(self, data_loader, model, lr, clip):
        model.train(True)
        if self.config['debug']:
            torch.autograd.set_detect_anomaly(True)
        total_loss = []
        loss_func = self.loss_func or model.calculate_loss
        for batch in data_loader:
            # one batch, one step
            self.optimizer.zero_grad()
            batch.to_tensor(device=self.config['device'])
            loss = loss_func(batch)
            if self.config['debug']:
                with torch.autograd.detect_anomaly():
                    loss.backward()
            else:
                loss.backward()
            total_loss.append(loss.data.cpu().numpy().tolist())
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            except:
                pass
            self.optimizer.step()
        avg_loss = np.mean(total_loss, dtype=np.float64)
        return model, avg_loss

    def _valid_epoch(self, data_loader, model):
        model.train(False)
        self.evaluator.clear()
        total_loss = []
        loss_func = self.loss_func or model.calculate_loss
        for batch in data_loader:
            batch.to_tensor(device=self.config['device'])
            scores = model.predict(batch)
            loss = loss_func(batch)
            total_loss.append(loss.data.cpu().numpy().tolist())
            if self.config['evaluate_method'] == 'popularity':
                evaluate_input = {
                    'uid': batch['uid'].tolist(),
                    'loc_true': batch['target'].tolist(),
                    'loc_pred': scores.tolist()
                }
            else:
                # negative sample
                # loc_true is always 0
                loc_true = [0] * self.config['batch_size']
                evaluate_input = {
                    'uid': batch['uid'].tolist(),
                    'loc_true': loc_true,
                    'loc_pred': scores.tolist()
                }
            self.evaluator.collect(evaluate_input)
        avg_acc = self.evaluator.evaluate()[self.metrics]  # 随便选一个就行
        avg_loss = np.mean(total_loss, dtype=np.float64)
        return avg_acc, avg_loss

    def _build_optimizer(self):
        """
        根据全局参数`learner`选择optimizer
        """
        if self.config['optimizer'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'],
                                   weight_decay=self.config['L2'])
        elif self.config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['learning_rate'],
                                        weight_decay=self.config['L2'])
        elif self.config['optimizer'] == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.config['learning_rate'],
                                            weight_decay=self.config['L2'])
        elif self.config['optimizer'] == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.config['learning_rate'],
                                            weight_decay=self.config['L2'])
        elif self.config['optimizer'] == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.config['learning_rate'])
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'],
                                   weight_decay=self.config['L2'])
        return optimizer

    def _build_scheduler(self):
        """
        目前就固定的 scheduler 吧
        """
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max',
                                                         patience=self.config['lr_step'],
                                                         factor=self.config['lr_decay'],
                                                         threshold=self.config['schedule_threshold'])
        return scheduler
