import json
import torch
import torch.optim as optim
import numpy as np
import os

from trafficdl.executor.abstract_executor import AbstractExecutor
from trafficdl.utils import get_evaluator


class TrajLocPredExecutor(AbstractExecutor):

    def __init__(self, config, model):
        self.evaluator = get_evaluator(config)
        self.metrics = 'Recall@{}'.format(config['topk'])
        self.config = config
        self.model = model.to(self.config['device'])
        self.tmp_path = './trafficdl/tmp/checkpoint/'
        self.cache_dir = './trafficdl/cache/model_cache'
        self.evaluate_res_dir = './trafficdl/cache/evaluate_cache'
        self.loss_func = None  # TODO: 根据配置文件支持选择特定的 Loss Func 目前并未实装

    def train(self, train_dataloader, eval_dataloader):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      self.model.parameters()),
                               lr=self.config['learning_rate'],
                               weight_decay=self.config['L2'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'max', patience=self.config['lr_step'],
            factor=self.config['lr_decay'],
            threshold=self.config['schedule_threshold'])

        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)
        metrics = {}
        metrics['accuracy'] = []
        train_total_batch = len(train_dataloader.dataset) / \
            train_dataloader.batch_size
        eval_total_batch = len(eval_dataloader.dataset) / \
            eval_dataloader.batch_size
        lr = self.config['learning_rate']
        for epoch in range(self.config['max_epoch']):
            self.model, avg_loss = self.run(
                train_dataloader, self.model, optimizer,
                self.config['learning_rate'], self.config['clip'],
                train_total_batch, self.config['verbose'])
            print('==>Train Epoch:{:0>2d} Loss:{:.4f} learning_rate:{}'.format(
                epoch, avg_loss, lr))
            # eval stage
            avg_acc = self._valid_epoch(
                eval_dataloader, self.model, eval_total_batch,
                self.config['verbose'])
            print('==>Eval Acc:{:.4f}'.format(avg_acc))
            metrics['accuracy'].append(avg_acc)
            save_name_tmp = 'ep_' + str(epoch) + '.m'
            torch.save(self.model.state_dict(), self.tmp_path + save_name_tmp)
            scheduler.step(avg_acc)
            lr_last = lr
            lr = optimizer.param_groups[0]['lr']
            if lr_last > lr:
                load_epoch = np.argmax(metrics['accuracy'])
                load_name_tmp = 'ep_' + str(load_epoch) + '.m'
                self.model.load_state_dict(
                    torch.load(self.tmp_path + load_name_tmp))
                print('load epoch={} model state'.format(load_epoch))
            if lr <= 0.9 * 1e-5:
                break
        best = np.argmax(metrics['accuracy'])  # 这个不是最好的一次吗？
        avg_acc = metrics['accuracy'][best]
        # save metrics
        with open('./metrics.json', 'w') as f:
            json.dump(metrics, f)
        load_name_tmp = 'ep_' + str(best) + '.m'
        self.model.load_state_dict(torch.load(self.tmp_path + load_name_tmp))
        # 删除之前创建的临时文件夹
        for rt, dirs, files in os.walk(self.tmp_path):
            for name in files:
                remove_path = os.path.join(rt, name)
                os.remove(remove_path)
        os.rmdir(self.tmp_path)

    def load_model(self, cache_name):
        self.model.load_state_dict(torch.load(cache_name))

    def save_model(self, cache_name):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        torch.save(self.model.state_dict(), cache_name)

    def evaluate(self, test_dataloader):
        self.model.train(False)
        self.evaluator.clear()
        test_total_batch = len(test_dataloader.dataset) / \
            test_dataloader.batch_size
        cnt = 0
        for batch in test_dataloader:
            batch.to_tensor(device=self.config['device'])
            scores = self.model.predict(batch)
            evaluate_input = {
                'uid': batch['uid'].tolist(),
                'loc_true': batch['target'].tolist(),
                'loc_pred': scores.tolist()
            }
            cnt += 1
            if cnt % self.config['verbose'] == 0:
                print('finish batch {}/{}'.format(cnt, test_total_batch))
            self.evaluator.collect(evaluate_input)
        self.evaluator.save_result(self.evaluate_res_dir)

    def run(self, data_loader, model, optimizer, lr, clip, total_batch,
            verbose):
        model.train(True)
        total_loss = []
        cnt = 0
        loss_func = self.loss_func or model.calculate_loss
        for batch in data_loader:
            # one batch, one step
            optimizer.zero_grad()
            batch.to_tensor(device=self.config['device'])
            loss = loss_func(batch)
            loss.backward()
            total_loss.append(loss.data.cpu().numpy().tolist())
            try:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.add_(-lr, p.grad.data)
            except:
                pass
            optimizer.step()
            cnt += 1
            if cnt % verbose == 0:
                print('finish batch {}/{}'.format(cnt, total_batch))
        with open('run_loss.pny', 'wb') as f:
            np.save(f, total_loss)
        avg_loss = np.mean(total_loss, dtype=np.float64)
        return model, avg_loss

    def _valid_epoch(self, data_loader, model, total_batch, verbose):
        model.train(False)
        self.evaluator.clear()
        cnt = 0
        for batch in data_loader:
            batch.to_tensor(device=self.config['device'])
            scores = model.predict(batch)
            evaluate_input = {
                'uid': batch['uid'].tolist(),
                'loc_true': batch['target'].tolist(),
                'loc_pred': scores.tolist()
            }
            cnt += 1
            if cnt % verbose == 0:
                print('finish batch {}/{}'.format(cnt, total_batch))
            self.evaluator.collect(evaluate_input)
        avg_acc = self.evaluator.evaluate()[self.metrics]  # 随便选一个就行
        return avg_acc
