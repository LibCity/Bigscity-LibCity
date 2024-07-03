import os
import json
import datetime
import pandas as pd
from libcity.utils import ensure_dir
from libcity.model import loss
from logging import getLogger
from libcity.evaluator.abstract_evaluator import AbstractEvaluator
from libcity.evaluator.traffic_state_evaluator import TrafficStateEvaluator


class TrafficStateInOutEvaluator(TrafficStateEvaluator):

    def __init__(self, config):
        self.metrics = config.get('metrics', ['MAE'])  # 评估指标, 是一个 list
        self.allowed_metrics = ["IN_masked_MAE", "IN_masked_MAPE", "OUT_masked_MAE", "OUT_masked_MAPE"]
        self.save_modes = config.get('save_mode', ['csv', 'json'])
        self.mode = config.get('evaluator_mode', 'single')  # or average
        self.config = config
        self.len_timeslots = 0
        self.result = {}  # 每一种指标的结果
        self.intermediate_result = {}  # 每一种指标每一个batch的结果
        self._check_config()
        self._logger = getLogger()

    def collect(self, batch):
        """
        收集一 batch 的评估输入

        Args:
            batch(dict): 输入数据，字典类型，包含两个Key:(y_true, y_pred):
                batch['y_true']: (num_samples/batch_size, timeslots, ..., feature_dim)
                batch['y_pred']: (num_samples/batch_size, timeslots, ..., feature_dim)
        """
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict of user')
        y_true = batch['y_true']  # tensor
        y_pred = batch['y_pred']  # tensor
        if y_true.shape != y_pred.shape:
            raise ValueError("batch['y_true'].shape is not equal to batch['y_pred'].shape")
        self.len_timeslots = y_true.shape[1]
        for i in range(1, self.len_timeslots+1):
            for metric in self.metrics:
                if metric+'@'+str(i) not in self.intermediate_result:
                    self.intermediate_result[metric+'@'+str(i)] = []
        if self.mode.lower() == 'average':  # 前i个时间步的平均loss
            for i in range(1, self.len_timeslots+1):
                for metric in self.metrics:
                    if metric == 'IN_masked_MAE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mae_torch_with_mask_value(
                                y_pred[:, :i][..., 0], y_true[:, :i][..., 0], 0).item())
                    elif metric == 'IN_masked_MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_torch_with_mask_value(
                                y_pred[:, :i][..., 0], y_true[:, :i][..., 0], 0).item())
                    elif metric == 'OUT_masked_MAE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mae_torch_with_mask_value(
                                y_pred[:, :i][..., 1], y_true[:, :i][..., 1], 0).item())
                    elif metric == 'OUT_masked_MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_torch_with_mask_value(
                                y_pred[:, :i][..., 1], y_true[:, :i][..., 1], 0).item())
        elif self.mode.lower() == 'single':  # 第i个时间步的loss
            for i in range(1, self.len_timeslots + 1):
                for metric in self.metrics:
                    if metric == 'IN_masked_MAE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mae_torch_with_mask_value(
                                y_pred[:, i-1][..., 0], y_true[:, i-1][..., 0], 0).item())
                    elif metric == 'IN_masked_MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_torch_with_mask_value(
                                y_pred[:, i-1][..., 0], y_true[:, i-1][..., 0], 0).item())
                    elif metric == 'OUT_masked_MAE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mae_torch_with_mask_value(
                                y_pred[:, i-1][..., 1], y_true[:, i-1][..., 1], 0).item())
                    elif metric == 'OUT_masked_MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_torch_with_mask_value(
                                y_pred[:, i-1][..., 1], y_true[:, i-1][..., 1], 0).item())
        else:
            raise ValueError('Error parameter evaluator_mode={}, please set `single` or `average`.'.format(self.mode))
