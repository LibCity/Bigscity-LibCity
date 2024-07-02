import os
import time
import numpy as np
import torch
from ray import tune
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter
from libcity.executor.abstract_executor import AbstractExecutor
from libcity.utils import get_evaluator, ensure_dir
from libcity.model import loss
from functools import partial
from libcity.executor.traffic_state_executor import TrafficStateExecutor


class STSSLExecutor(TrafficStateExecutor):
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)
        self.temp = config.get('temp', 4)



    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        with torch.no_grad():
            self.model.eval()
            # self.evaluator.clear()
            y_truths = []
            y_preds = []
            for batch in test_dataloader:
                batch.to_tensor(self.device)
                output = self.model.predict(batch)
                y_true = self._scaler.inverse_transform(batch['y'][..., :self.output_dim])
                y_pred = self._scaler.inverse_transform(output[..., :self.output_dim])
                y_truths.append(y_true.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())
                # evaluate_input = {'y_true': y_true, 'y_pred': y_pred}
                # self.evaluator.collect(evaluate_input)
            # self.evaluator.save_result(self.evaluate_res_dir)
            y_preds = np.concatenate(y_preds, axis=0)
            y_truths = np.concatenate(y_truths, axis=0)  # concatenate on batch
            outputs = {'prediction': y_preds, 'truth': y_truths}
            filename = \
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + '_' \
                + self.config['model'] + '_' + self.config['dataset'] + '_predictions.npz'
            np.savez_compressed(os.path.join(self.evaluate_res_dir, filename), **outputs)
            self.evaluator.clear()
            self.evaluator.collect({'y_true': torch.tensor(y_truths), 'y_pred': torch.tensor(y_preds)})
            test_result = self.evaluator.save_result(self.evaluate_res_dir)
            return test_result

    def dwa(self,L_old, L_new, T=2):
        '''
        L_old: list.
        '''
        L_old = torch.tensor(L_old, dtype=torch.float32)
        L_new = torch.tensor(L_new, dtype=torch.float32)
        N = len(L_new)  # task number
        r = L_old / L_new

        w = N * torch.softmax(r / T, dim=0)
        return w.numpy()
    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        self._logger.info("num_batches:{}".format(num_batches))

        loss_tm1 = loss_t = np.ones(3)  # (1.0, 1.0, 1.0)
        for epoch_idx in range(self._epoch_num, self.epochs):
            loss_tm2 = loss_tm1
            loss_tm1 = loss_t
            if (epoch_idx == 0) or (epoch_idx == 1):
                loss_weights = self.dwa(loss_tm1, loss_tm1, self.temp)
            else:
                loss_weights = self.dwa(loss_tm1, loss_tm2, self.temp)
            start_time = time.time()
            losses, train_epoch_loss, loss_t = self._train_epoch(train_dataloader, epoch_idx, self.loss_func, loss_weights)
            t1 = time.time()
            train_time.append(t1 - start_time)
            self._writer.add_scalar('training loss', np.mean(losses), epoch_idx)
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
                message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'.\
                    format(epoch_idx, self.epochs, np.mean(losses), val_loss, log_lr, (end_time - start_time))
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
        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx, loss_func=None, loss_weights=np.ones(3)):
        """
        完成模型一个轮次的训练

        Args:
            train_dataloader: 训练数据
            epoch_idx: 轮次数
            loss_func: 损失函数

        Returns:
            list: 每个batch的损失的数组
        """
        self.model.train()
        loss_func = loss_func if loss_func is not None else self.model.calculate_loss_with_weight
        losses = []

        total_loss = 0
        total_sep_loss = np.zeros(3)
        batch_count = 0
        for batch in train_dataloader:
            batch_count += 1
            self.optimizer.zero_grad()
            batch.to_tensor(self.device)
            loss, sep_loss = loss_func(batch, loss_weights)
            total_loss+=loss.item()
            total_sep_loss+=sep_loss
            self._logger.debug(loss.item())
            losses.append(loss.item())
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        total_loss/=batch_count
        total_sep_loss/=batch_count
        return losses, total_loss, total_sep_loss
