import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from logging import getLogger
from libcity.executor.abstract_executor import AbstractExecutor
from libcity.utils import get_evaluator


class NASRExecutor(AbstractExecutor):

    def __init__(self, config, model):
        self.config = config
        self.max_epoch = config['max_epoch']
        self.device = config['device']
        self.evaluator = get_evaluator(config)
        self.model = model.to(self.device)
        self._logger = getLogger()
        self.exp_id = self.config.get('exp_id', 'default')
        self.tmp_path = './libcity/tmp/{}/checkpoint/'.format(self.exp_id)
        self.cache_dir = './libcity/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/evaluate_cache'.format(self.exp_id)
        self.optimizer = self._build_optimizer()
        # NASR do not use scheduler
        self.scheduler = self._build_scheduler()

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)
        metrics = {'loss': []}
        lr = self.config['learning_rate']
        # NASR doesn't support hyper_tune pipeline
        if self.config['hyper_tune']:
            raise TypeError('NASR doesn\'t support hyper_tune pipeline!')
        self._logger.info('start train NASR:')
        self._logger.info(self.model)
        # first train g function
        for epoch in range(self.max_epoch):
            train_loss = self._train_epoch(train_dataloader, 'g_loss')
            eval_loss = self._valid_epoch(eval_dataloader, 'g_loss')
            self._logger.info('==>Train G Function Epoch:{:4d} Train Loss:{:.5f}, Val Loss:{:.5f} learning_rate:{}'.
                              format(epoch, train_loss, eval_loss, lr))
            metrics['loss'].append(eval_loss)
            save_name_tmp = 'ep_' + str(epoch) + '.pt'
            torch.save(self.model.state_dict(), self.tmp_path + save_name_tmp)
            self.scheduler.step(eval_loss)
            lr = self.optimizer.param_groups[0]['lr']
            if lr < self.config['early_stop_lr']:
                # early stopping
                break
        if self.config['load_best_epoch']:
            best = np.argmin(metrics['loss'])
            load_name_tmp = 'ep_' + str(best) + '.pt'
            self.model.load_state_dict(
                torch.load(self.tmp_path + load_name_tmp))
        # train h function
        for epoch in range(self.max_epoch):
            train_loss = self._train_epoch(train_dataloader, 'h_loss')
            eval_loss = self._valid_epoch(eval_dataloader, 'h_loss')
            self._logger.info('==>Train H Function Epoch:{:4d} Train Loss:{:.5f}, Val Loss:{:.5f} learning_rate:{}'.
                              format(epoch, train_loss, eval_loss, lr))
            metrics['loss'].append(eval_loss)
            save_name_tmp = 'ep_' + str(epoch) + '.pt'
            torch.save(self.model.state_dict(), self.tmp_path + save_name_tmp)
            self.scheduler.step(eval_loss)
            lr = self.optimizer.param_groups[0]['lr']
            if lr < self.config['early_stop_lr']:
                # early stopping
                break
        if self.config['load_best_epoch']:
            best = np.argmin(metrics['loss'])
            load_name_tmp = 'ep_' + str(best) + '.pt'
            self.model.load_state_dict(
                torch.load(self.tmp_path + load_name_tmp))
        # remove temp folder
        for rt, dirs, files in os.walk(self.tmp_path):
            for name in files:
                remove_path = os.path.join(rt, name)
                os.remove(remove_path)
        os.rmdir(self.tmp_path)

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self.model.train(False)
        self.evaluator.clear()
        for batch in tqdm(test_dataloader, desc="test model"):
            batch.to_tensor(device=self.config['device'])
            generate_trace = self.model.predict(batch)
            self.evaluator.collect({
                'generate_trace': generate_trace,
                'true_trace': batch['true_trace']
            })
        self.evaluator.save_result(self.evaluate_res_dir)

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        model_state, optimizer_state = torch.load(cache_name)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件

        Args:
            cache_name(str): 保存的文件名
        """
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        # save optimizer when load epoch to train
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), cache_name)

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

    def _train_epoch(self, train_dataloader, mode):
        """

        Args:
            train_dataloader: train data
            mode: loss mode. g_loss or h_loss

        Returns:

        """
        self.model.train(True)
        if self.config['debug']:
            torch.autograd.set_detect_anomaly(True)
        total_loss = []
        for batch in tqdm(train_dataloader, desc="train model"):
            # one batch, one step
            self.optimizer.zero_grad()
            batch.to_tensor(device=self.config['device'])
            batch.data['loss_mode'] = mode
            loss = self.model.calculate_loss(batch)
            if self.config['debug']:
                with torch.autograd.detect_anomaly():
                    loss.backward()
            else:
                loss.backward()
            total_loss.append(loss.item())
            self.optimizer.step()
        avg_loss = np.mean(total_loss, dtype=np.float64)
        return avg_loss

    def _valid_epoch(self, eval_dataloader, mode):
        self.model.train(False)
        total_loss = []
        for batch in tqdm(eval_dataloader, desc="eval model"):
            # one batch, one step
            batch.to_tensor(device=self.config['device'])
            batch.data['loss_mode'] = mode
            with torch.no_grad():
                loss = self.model.calculate_loss(batch)
            total_loss.append(loss.item())
        avg_loss = np.mean(total_loss, dtype=np.float64)
        return avg_loss

    def _build_scheduler(self):
        """
        目前就固定的 scheduler 吧
        """
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                         patience=self.config['lr_step'],
                                                         factor=self.config['lr_decay'],
                                                         threshold=self.config['schedule_threshold'])
        return scheduler
