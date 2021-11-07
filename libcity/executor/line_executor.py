import os
import time

import numpy as np
import torch
from ray import tune

from libcity.executor.traffic_state_executor import TrafficStateExecutor


class LINEExecutor(TrafficStateExecutor):
    def __init__(self, config, model):
        TrafficStateExecutor.__init__(self, config, model)
        self.loss_func = None

    def evaluate(self, test_dataloader):
        """
        use model to test data
        """
        self.evaluator.evaluate()

        with torch.no_grad():
            self.model.eval()
            # TODO 处理自定义 lossfunc
            loss_func = self.model.calculate_loss
            losses = []
            for batch in test_dataloader:
                batch.to_tensor(self.device)
                loss = loss_func(batch)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            return mean_loss

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config
        """
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []

        for epoch_idx in range(self._epoch_num, self.epochs):
            start_time = time.time()
            train_loss = self._train_epoch(train_dataloader, epoch_idx, self.loss_func)
            t1 = time.time()
            train_time.append(t1 - start_time)
            self._writer.add_scalar('training loss', train_loss, epoch_idx)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            val_loss = self._valid_epoch(eval_dataloader, epoch_idx, self.loss_func)
            end_time = time.time()
            eval_time.append(end_time - t2)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            if (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s' \
                    .format(epoch_idx, self.epochs, train_loss, val_loss, log_lr, (end_time - start_time))
                self._logger.info(message)

            if self.hyper_tune:
                # use ray tune to checkpoint
                with tune.checkpoint_dir(step=epoch_idx) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    self.save_model(path)
                # ray tune use loss to determine which params are best
                tune.report(loss=val_loss)

            if val_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break
        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)
            save_list = os.listdir(self.cache_dir)
            for save_file in save_list:
                if '.tar' in save_file:
                    os.remove(self.cache_dir + '/' + save_file)
        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx, loss_func=None):
        """
        完成模型一个轮次的训练

        Args:
            train_dataloader: 训练数据
            epoch_idx: 轮次数
            loss_func: 损失函数

        Returns:
            float: 训练集的损失值
        """
        self.model.train()
        loss_func = loss_func if loss_func is not None else self.model.calculate_loss
        losses = []
        for batch in train_dataloader:
            self.optimizer.zero_grad()
            batch.to_tensor(self.device)
            loss = loss_func(batch)
            self._logger.debug(loss.item())
            losses.append(loss.item())
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        return losses

    def _valid_epoch(self, eval_dataloader, epoch_idx, loss_func=None):
        """
        完成模型一个轮次的评估

        Args:
            eval_dataloader: 评估数据
            epoch_idx: 轮次数
            loss_func: 损失函数

        Returns:
            float: 验证集的损失值
        """
        with torch.no_grad():
            self.model.eval()
            loss_func = loss_func if loss_func is not None else self.model.calculate_loss
            losses = []
            for batch in eval_dataloader:
                batch.to_tensor(self.device)
                loss = loss_func(batch)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            self._writer.add_scalar('eval loss', mean_loss, epoch_idx)
            return mean_loss
