import os
import json
import time
from collections import Counter
import numpy as np

from libcity.evaluator.abstract_evaluator import AbstractEvaluator


class GeoSANEvaluator(AbstractEvaluator):

    def __init__(self, config):
        self.metrics = config['evaluator_config']['metrics']  # 评估指标, only contains hr and ndcg
        self.topk = config['evaluator_config']['topk']
        self.num_neg = config['executor_config']['test']['num_negative_samples']
        self.cnter = Counter()
        self.result = {}
        self.allowed_metrics = ['hr', 'ndcg']
        self._check_config()

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        for i in self.metrics:
            if i.lower() not in self.allowed_metrics:
                raise ValueError('the metric is not allowed in \
                    TrajLocPredEvaluator')

    def collect(self, batch):
        """
        收集一 batch 的评估输入

        Args:
            batch(torch.Tensor): 模型输出结果([(1+K)*L, N])
        """
        idx = batch.sort(descending=True, dim=0)[1]
        order = idx.topk(1, dim=0, largest=False)[1]
        # order: N个输入中postive对应的位置索引
        self.cnter.update(order.squeeze().tolist())

    def evaluate(self):
        """
        返回之前收集到的所有 batch 的评估结果
        """
        array = np.zeros(self.num_neg + 1)
        for k, v in self.cnter.items():
            array[k] = v
        # hit rate and NDCG
        hr = array.cumsum()
        ndcg = 1 / np.log2(np.arange(0, self.num_neg + 1) + 2)
        ndcg = ndcg * array
        ndcg = ndcg.cumsum() / hr.max()
        hr = hr / hr.max()
        if 'NDCG' in self.metrics:
            self.result[f'NDCG@{self.topk}'] = float(ndcg[self.topk-1])
        if 'HR' in self.metrics:
            self.result[f'HR@{self.topk}'] = float(hr[self.topk-1])

    def save_result(self, save_path, filename=None):
        """
        将评估结果保存到 save_path 文件夹下的 filename 文件中

        Args:
            save_path: 保存路径
            filename: 保存文件名
        """
        self.evaluate()
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        if filename is None:
            # 使用时间戳
            filename = time.strftime(
                "%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
        print('evaluate result is ', json.dumps(self.result, indent=1))
        with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') \
                as f:
            json.dump(self.result, f)

    def clear(self):
        """
        清除之前收集到的 batch 的评估信息，适用于每次评估开始时进行一次清空，排除之前的评估输入的影响。
        """
        self.cnter.clear()
        self.result = {}
