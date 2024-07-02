from libcity.executor import TrafficStateExecutor
from libcity.model import loss
import torch.nn as nn
import torch
import numpy as np
import os
import time


class MegaCRNExecutor(TrafficStateExecutor):
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)
        self.lamb = config.get("lamb", 0.01)
        self.lamb1 = config.get("lamb1", 0.01)
        self.separate_loss = nn.TripletMarginLoss(margin=1.0)
        self.compact_loss = loss.masked_mse_torch
        self.batches_seen = 0

    def _build_train_loss(self):
        def default_train_loss(batch):
            y = batch['y']
            output, h_att, query, pos, neg = self.model.predict(batch)
            y_true = self._scaler.inverse_transform(y[..., :self.output_dim])
            y_pred = self._scaler.inverse_transform(output[..., :self.output_dim])
            loss1 = loss.masked_mae_torch(y_pred, y_true, null_val=0.0)
            loss2 = self.separate_loss(query, pos.detach(), neg.detach())
            loss3 = self.compact_loss(query, pos.detach())
            return loss1 + self.lamb * loss2 + self.lamb1 * loss3

        if self.train_loss.lower() == 'none':
            self._logger.warning('Received none train loss func and will use the default loss func defined in the'
                                 ' MegaCRNExecutor.')
            return default_train_loss
        else:
            return super()._build_train_loss()

    def _train_epoch(self, train_dataloader, epoch_idx, loss_func=None):
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
        loss_func = loss_func if loss_func is not None else self.model.calculate_loss
        losses = []
        for batch in train_dataloader:
            batch.batches_seen = self.batches_seen
            self.optimizer.zero_grad()
            batch.to_tensor(self.device)
            loss = loss_func(batch)
            self._logger.debug(loss.item())
            losses.append(loss.item())
            self.batches_seen += 1
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
            float: 评估数据的平均损失值
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
                output, _, _, _, _ = self.model.predict(batch)
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
