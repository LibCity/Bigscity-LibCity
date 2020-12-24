import os
import json
import datetime
from trafficdl.utils import ensure_dir
from trafficdl.model import loss
from logging import getLogger

class TrafficSpeedPredEvaluator(object):

    def __init__(self, config):
        self.metrics = config['metrics']  # 评估指标, 是一个 list
        self.allowed_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE']
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
                raise ValueError('the metric {} is not allowed in TrafficSpeedPredEvaluator'.format(str(metric)))

    def collect(self, batch):
        '''
        收集一 batch 的评估输入
        '''
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict of user')
        for metric in self.metrics:
            if metric not in self.intermediate_result:
                self.intermediate_result[metric] = []
        y_true = batch['y_true']  # ndarray
        y_pred = batch['y_pred']  # ndarray
        for metric in self.metrics:
            if metric == 'MAE':
                self.intermediate_result[metric].append(loss.masked_mae_np(y_pred, y_true, 0))
            elif metric == 'MSE':
                self.intermediate_result[metric].append(loss.masked_mse_np(y_pred, y_true, 0))
            elif metric == 'RMSE':
                self.intermediate_result[metric].append(loss.masked_rmse_np(y_pred, y_true, 0))
            elif metric == 'MAPE':
                self.intermediate_result[metric].append(loss.masked_mape_np(y_pred, y_true, 0))

    def evaluate(self):
        '''
        返回之前收集到的所有 batch 的评估结果
        '''
        for metric in self.metrics:
            self.result[metric] = sum(self.intermediate_result[metric]) / len(self.intermediate_result[metric])
        return self.result

    def save_result(self, save_path, filename=None):
        '''
        将评估结果保存到 save_path 文件夹下的 filename 文件中
        '''
        self.evaluate()
        ensure_dir(save_path)
        if filename is None:  # 使用时间戳
            filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + self.config['model']
        self._logger.info('Evaluate result is ' + json.dumps(self.result))
        with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
            json.dump(self.result, f)
        self._logger.info('Evaluate result is saved at ' +
                          os.path.join(save_path, '{}.json'.format(filename)))

    def clear(self):
        '''
        清除之前收集到的 batch 的评估信息，适用于每次评估开始时进行一次清空，排除之前的评估输入的影响。
        '''
        self.result = {}
        self.intermediate_result = {}
