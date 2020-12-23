import numpy as np
import os
import json
import time

from trafficdl.evaluator.abstract_evaluator import AbstractEvaluator
from trafficdl.evaluator.eval_funcs import top_k
allowed_metrics = ['topk']
class TrajLocPredEvaluator(AbstractEvaluator):

    def __init__(self, config):
        self.metrics = config['metrics'] # 评估指标, 是一个 list
        self.config = config
        self.topk = 1
        self.result = {}
        self.intermediate_result = {} # 分用户存，每个用户下每个指标一个 list 目前不实装分 user 存
        self._check_config()

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        for i in self.metrics:
            if i not in allowed_metrics:
                raise ValueError('the metric is not allowed in TrajLocPredEvaluator')
        if 'topk' in self.metrics:
            self.topk = self.config['topk']
        
        for metric in self.metrics:
            if metric == 'topk':
                # 第一元素存放命中的次数，第二个存放总共的次数，每次累加最后就能够计算一个 ac 率了
                self.intermediate_result[metric] = [0, 0]
            
    
    def collect(self, batch):
        '''
        Args:
            batch (dict): contains three keys: uid, loc_true, and loc_pred.
            uid (list): 来自于 batch 中的 uid，通过索引可以确定 loc_true 与 loc_pred 中每一行（元素）是哪个用户的一次输入。
            loc_true (list): 期望地点(target)，来自于 batch 中的 target
            loc_pred (matrix): 实际上模型的输出，batch_size * output_dim. 
        '''
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict of user')
        # 初始化 intermediate_result
        # TODO: 评估目前就整体返回一个值算了，看不到分 user 统计的必要
        # for uid in batch:
        #     if uid not in self.intermediate_result:
        #         self.intermediate_result[uid] = {}
        #         for metric in self.metrics:
        #             self.intermediate_result[uid][metric] = []
        for metric in self.metrics:
            if metric == 'topk':
                res = top_k(batch['loc_pred'], batch['loc_true'], self.topk)
                unique, counts = np.unique(res, return_counts=True)
                self.intermediate_result[metric][0] += counts[1]
                self.intermediate_result[metric][1] += (counts[0] + counts[1])
                # # 把评估的结果写到不同的 user 里面
                # for i, uid in enumerate(batch[uid]):
                #     self.intermediate_result[metric].append(res[i])
            

    def evaluate(self):
        for metric in self.metrics:
            if metric == 'topk':
                self.result[metric] = self.intermediate_result[metric][0] / self.intermediate_result[metric][1]
        return self.result
    
    def save_result(self, save_path, filename=None):
        self.evaluate()
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if filename == None:
            # 使用时间戳
            filename = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
        print('evaluate result is ', json.dumps(self.result))
        with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
            json.dump(self.result, f)

    def clear(self):
        self.result = {}
        for metric in self.metrics:
            if metric == 'topk':
                # 第一元素存放命中的次数，第二个存放总共的次数，每次累加最后就能够计算一个 ac 率了
                self.intermediate_result[metric] = [0, 0]
