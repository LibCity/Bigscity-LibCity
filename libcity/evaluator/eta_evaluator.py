import os
import json
import datetime
import pandas as pd
from libcity.utils import ensure_dir
from libcity.model import loss
from logging import getLogger
from libcity.evaluator.abstract_evaluator import AbstractEvaluator


class ETAEvaluator(AbstractEvaluator):

    def __init__(self, config):
        self.metrics = config.get('metrics', ['MAE'])  # 评估指标, 是一个 list
        self.allowed_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'masked_MAE',
                                'masked_MSE', 'masked_RMSE', 'masked_MAPE', 'R2', 'EVAR']
        self.save_modes = config.get('save_modes', ['csv', 'json'])
        self.config = config
        self.result = {}  # 每一种指标的结果
        self.intermediate_result = {}  # 每一种指标每一个batch的结果
        self._check_config()
        self._logger = getLogger()

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        for metric in self.metrics:
            if metric not in self.allowed_metrics:
                raise ValueError('the metric {} is not allowed in ETAEvaluator'.format(str(metric)))

    def collect(self, batch):
        """
        收集一 batch 的评估输入

        Args:
            batch(dict): 输入数据，字典类型，包含两个Key:(y_true, y_pred):
                batch['y_true']: (batch_size, 1)
                batch['y_pred']: (batch_size, 1)
        """
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict of user')
        y_true = batch['y_true']  # tensor
        y_pred = batch['y_pred']  # tensor
        if y_true.shape != y_pred.shape:
            raise ValueError("batch['y_true'].shape is not equal to batch['y_pred'].shape")
        for metric in self.metrics:
            if metric not in self.intermediate_result:
                self.intermediate_result[metric] = []
        for metric in self.metrics:
            if metric == 'masked_MAE':
                self.intermediate_result[metric].append(
                    loss.masked_mae_torch(y_pred, y_true, 0).item())
            elif metric == 'masked_MSE':
                self.intermediate_result[metric].append(
                    loss.masked_mse_torch(y_pred, y_true, 0).item())
            elif metric == 'masked_RMSE':
                self.intermediate_result[metric].append(
                    loss.masked_rmse_torch(y_pred, y_true, 0).item())
            elif metric == 'masked_MAPE':
                self.intermediate_result[metric].append(
                    loss.masked_mape_torch(y_pred, y_true, 0).item())
            elif metric == 'MAE':
                self.intermediate_result[metric].append(
                    loss.masked_mae_torch(y_pred, y_true).item())
            elif metric == 'MSE':
                self.intermediate_result[metric].append(
                    loss.masked_mse_torch(y_pred, y_true).item())
            elif metric == 'RMSE':
                self.intermediate_result[metric].append(
                    loss.masked_rmse_torch(y_pred, y_true).item())
            elif metric == 'MAPE':
                self.intermediate_result[metric].append(
                    loss.masked_mape_torch(y_pred, y_true).item())
            elif metric == 'R2':
                self.intermediate_result[metric].append(
                    loss.r2_score_torch(y_pred, y_true).item())
            elif metric == 'EVAR':
                self.intermediate_result[metric].append(
                    loss.explained_variance_score_torch(y_pred, y_true).item())

    def evaluate(self):
        """
        返回之前收集到的所有 batch 的评估结果
        """
        for metric in self.metrics:
            self.result[metric] = sum(self.intermediate_result[metric]) / \
                                  len(self.intermediate_result[metric])
        return self.result

    def save_result(self, save_path, filename=None):
        """
        将评估结果保存到 save_path 文件夹下的 filename 文件中

        Args:
            save_path: 保存路径
            filename: 保存文件名
        """
        self.evaluate()
        ensure_dir(save_path)
        if filename is None:  # 使用时间戳
            filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                       self.config['model'] + '_' + self.config['dataset']

        if 'json' in self.save_modes:
            self._logger.info('Evaluate result is ' + json.dumps(self.result))
            with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
                json.dump(self.result, f)
            self._logger.info('Evaluate result is saved at ' +
                              os.path.join(save_path, '{}.json'.format(filename)))

        dataframe = {}
        if 'csv' in self.save_modes:
            for metric in self.metrics:
                dataframe[metric] = []
            for metric in self.metrics:
                dataframe[metric].append(self.result[metric])
            dataframe = pd.DataFrame(dataframe, index=range(1, 2))
            dataframe.to_csv(os.path.join(save_path, '{}.csv'.format(filename)), index=False)
            self._logger.info('Evaluate result is saved at ' +
                              os.path.join(save_path, '{}.csv'.format(filename)))
            self._logger.info("\n" + str(dataframe))
        return dataframe

    def clear(self):
        """
        清除之前收集到的 batch 的评估信息，适用于每次评估开始时进行一次清空，排除之前的评估输入的影响。
        """
        self.result = {}
        self.intermediate_result = {}
