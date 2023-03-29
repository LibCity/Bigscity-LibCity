import os
import json
import time
from collections import Counter
import numpy as np

from libcity.evaluator.abstract_evaluator import AbstractEvaluator


class GeoSANEvaluator(AbstractEvaluator):

    def __init__(self, config):
        self.metrics = config['metrics']  # 评估指标
        self.topk = config['topk']
        self.num_neg = config['executor_config']['test']['num_negative_samples']
        self.cnter = Counter()
        self.result = {}
        self.allowed_metrics = ['Precision', 'Recall', 'F1', 'MRR', 'MAP', 'NDCG']
        self._check_config()

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        for i in self.metrics:
            if i not in self.allowed_metrics:
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
        recall = hr / hr.max()
        precision = hr / (hr.max() * self.topk)
        if precision[self.topk-1] + recall[self.topk-1] == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * precision[self.topk-1] * recall[self.topk-1] / (precision[self.topk-1] + recall[self.topk-1])
        # 计算 MRR 和 MAP
        # 实际上在这个任务下 MRR = MAP，所以只需要计算一个就行
        # array 就是每个 target 在预测顺序中的排位情况
        topk_array = array[:self.topk]
        rank = 0.0
        for rank_index in range(self.topk):
            if topk_array[rank_index] > 0:
                rank = rank + (1.0 / (rank_index + 1)) * topk_array[rank_index]
        mrr = rank / hr.max()
        map = rank / hr.max()
        if 'Recall' in self.metrics:
            self.result[f'Recall@{self.topk}'] = recall[self.topk-1]
        if 'Precision' in self.metrics:
            self.result[f'Precision@{self.topk}'] = precision[self.topk-1]
        if 'F1' in self.metrics:
            self.result[f'F1@{self.topk}'] = f1_score
        if 'MAP' in self.metrics:
            self.result[f'MAP@{self.topk}'] = map
        if 'MRR' in self.metrics:
            self.result[f'MRR@{self.topk}'] = mrr
        if 'NDCG' in self.metrics:
            self.result[f'NDCG@{self.topk}'] = ndcg[self.topk-1]

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
