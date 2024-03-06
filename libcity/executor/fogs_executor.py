import os
import time
import numpy as np
import torch
from torch import nn

from libcity.executor.traffic_state_executor import TrafficStateExecutor
from libcity.model import loss


class FOGSExecutor(TrafficStateExecutor):
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)
        self.use_trend = config.get('use_trend', True)
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.horizon = config.get('output_window', 12)
        self.trend_embedding = config.get('trend_embedding', False)
        self.output_window = config.get('output_window', 12)
        if self.trend_embedding:
            self.trend_bias_embeddings = nn.Embedding(288, self.num_nodes * self.output_window)

    def adjust_output(self, output, valx, valy_slot):

        if self.use_trend:
            B, T, N = output.shape
            x_truth = self._scaler.inverse_transform(valx).reshape(B, T, -1)  # 把x_truth也变成(B,T,N)
            x_truth = x_truth.permute(1, 0, 2)  # (B,T,N)->(T,B,N)
            output = output.permute(1, 0, 2)  # (T, B, N)

            if self.trend_embedding:
                bias = self.trend_bias_embeddings(valy_slot[:, 0])  # (B, N * T)
                bias = torch.reshape(bias, (-1, self.num_nodes, self.horizon))  # (B, N, T)
                bias = bias.permute(2, 0, 1)  # (B, N, T)->(T, B, N)
                bias = bias.to(self.device)
                predict = (1 + output) * x_truth[-1] + bias  # 将预测趋势变成流量值  (T, B, N)
            else:
                predict = (1 + output) * x_truth[-1]  # 将预测趋势变成流量值

            predict = predict.permute(1, 0, 2)  # (T, B, N)->(B, T, N)
        else:
            predict = self._scaler.inverse_transform(output)
        return predict

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
            losses = []
            for batch in eval_dataloader:
                batch.to_tensor(self.device)
                valx = batch['X']
                valy = batch['y'][:, :, :, 0]
                valy_slot = batch['y_slot']
                output = self.model.predict(batch)
                predict = self.adjust_output(output, valx, valy_slot)
                val_loss = loss.masked_mae_torch(predict, valy, 0.0)
                self._logger.debug(val_loss.item())
                losses.append(val_loss.item())
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
                valx = batch['X']
                valy = batch['y'][:, :, :, 0]
                valy_slot = batch['y_slot']
                output = self.model.predict(batch)
                predict = self.adjust_output(output, valx, valy_slot)
                y_truths.append(valy.cpu().numpy())
                y_preds.append(predict.cpu().numpy())
            y_preds = np.concatenate(y_preds, axis=0)
            y_truths = np.concatenate(y_truths, axis=0)  # concatenate on batch
            print("y_preds: ", y_preds.shape)
            print("y_truths: ", y_truths.shape)
            outputs = {'prediction': y_preds, 'truth': y_truths}
            filename = \
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + '_' \
                + self.config['model'] + '_' + self.config['dataset'] + '_predictions.npz'
            np.savez_compressed(os.path.join(self.evaluate_res_dir, filename), **outputs)
            self.evaluator.clear()
            self.evaluator.collect({'y_true': torch.tensor(y_truths), 'y_pred': torch.tensor(y_preds)})
            test_result = self.evaluator.save_result(self.evaluate_res_dir)
            return test_result