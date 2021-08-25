import os
import json
import time

import networkx as nx

from libtraffic.utils import ensure_dir
from logging import getLogger
from libtraffic.evaluator.abstract_evaluator import AbstractEvaluator


class MapMatchingEvaluator(AbstractEvaluator):
    def __init__(self, config):
        self.metrics = config['evaluator_config']['metrics']  # 评估指标, 是一个 list
        self.allowed_metrics = ['RMF', 'AN', 'AL']
        self.config = config
        self.res_dir = './libtraffic/cache/result_cache'
        self.result = {}  # 每一种指标的结果
        self._check_config()
        self._logger = getLogger()
        self.evaluate_result = {}

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        for metric in self.metrics:
            if metric not in self.allowed_metrics:
                raise ValueError('the metric {} is not allowed in TrafficStateEvaluator'.format(str(metric)))

    def collect(self, batch):
        self.rd_nwk = batch["rd_nwk"]
        self.truth_sequence = list(batch["route"])
        self.result = batch["result"]
        self.find_completed_sequence()
        # find the longest common subsequence
        self.find_lcs()
        self.rel_to_distance = {}
        for point1 in self.rd_nwk.adj:
            for point2 in self.rd_nwk.adj[point1]:
                rel = self.rd_nwk.adj[point1][point2]
                self.rel_to_distance[rel['rel_id']] = rel['distance']

    def evaluate(self):
        if 'RMF' in self.metrics:
            d_plus = 0
            d_sub = 0
            d_total = 0
            for rel_id in self.truth_sequence:
                d_total += self.rel_to_distance[rel_id]
            i = j = k = 0
            while i < len(self.lcs):
                while self.truth_sequence[j] != self.lcs[i]:
                    d_sub += self.rel_to_distance[self.truth_sequence[j]]
                    j += 1
                    if j == len(self.truth_sequence):
                        break
                while self.output_sequence[k] != self.lcs[i]:
                    d_plus += self.rel_to_distance[self.output_sequence[j]]
                    k += 1
                    if k == len(self.output_sequence):
                        break
                i += 1
            self.evaluate_result['RMF'] = (d_plus + d_sub) / d_total
        if 'AN' in self.metrics:
            self.evaluate_result['AN'] = len(self.lcs) / len(self.truth_sequence)
        if 'AL' in self.metrics:
            d_lcs = 0
            d_tru = 0
            for rel_id in self.lcs:
                d_lcs += self.rel_to_distance[rel_id]
            for rel_id in self.truth_sequence:
                d_tru += self.rel_to_distance[rel_id]
            self.evaluate_result['AL'] = d_lcs / d_tru


    def save_result(self, save_path, filename=None):
        """
        将评估结果保存到 save_path 文件夹下的 filename 文件中

        Args:
            save_path: 保存路径
            filename: 保存文件名
        """
        ensure_dir(self.res_dir)
        file_name = self.config.get('dataset', '') + '.out'
        with open(self.res_dir + '/' + file_name, 'w') as f:
            f.write('dyna_id,rel_id\n')
            for line in self.result:
                f.write(str(line[0]) + ',' + str(line[1]) + '\n')

        self.evaluate()
        ensure_dir(save_path)
        if filename is None:
            # 使用时间戳
            filename = time.strftime(
                "%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
        print('evaluate result is ', json.dumps(self.evaluate_result, indent=1))
        with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
            json.dump(self.evaluate_result, f)
            self._logger.info('Evaluate result is saved at ' + os.path.join(save_path, '{}.json'.format(filename)))


    def clear(self):
        pass

    def find_lcs(self):
        sequence1 = self.output_sequence
        sequence2 = self.truth_sequence
        len1 = len(sequence1)
        len2 = len(sequence2)
        res = [[0 for i in range(len1 + 1)] for j in range(len2 + 1)]  # python 初始化二维数组 [len1+1],[len2+1]
        for i in range(1, len2 + 1):  # 开始从1开始，到len2+1结束
            for j in range(1, len1 + 1):  # 开始从1开始，到len2+1结束
                if sequence2[i - 1] == sequence1[j - 1]:
                    res[i][j] = res[i - 1][j - 1] + 1
                else:
                    res[i][j] = max(res[i - 1][j], res[i][j - 1])
        lcs = []
        i = len(sequence2)
        j = len(sequence1)
        while i > 0 and j > 0:
            # 开始从1开始，到len2+1结束
            if sequence2[i - 1] == sequence1[j - 1]:
                lcs.append(sequence2[i - 1])
                i = i - 1
                j = j - 1
            else:
                if res[i - 1][j] > res[i][j - 1]:
                    i = i - 1
                elif res[i - 1][j] < res[i][j - 1]:
                    j = j - 1
                else:
                    i = i - 1
        lcs.reverse()
        self.lcs = lcs

    def find_completed_sequence(self):
        uncompleted_sequence = []
        for line in self.result:
            uncompleted_sequence.append(line[1])
        i = 0
        while i < uncompleted_sequence.count(None):
            uncompleted_sequence.remove(None)
            i += 1
        completed_sequence = []
        i = 0
        last_road = None
        last_point = None
        edges = list(self.rd_nwk.edges)
        while i < len(uncompleted_sequence):
            if last_road is not None:
                if last_road == uncompleted_sequence[i]:
                    i += 1
                else:
                    if last_point != edges[uncompleted_sequence[i]][0]:
                        path = nx.dijkstra_path(self.rd_nwk, source=last_point,
                                                target=edges[uncompleted_sequence[i]][0])
                        j = 0
                        for road in path:
                            if j != 0:
                                completed_sequence.append(road)
                            j += 1
                    else:
                        completed_sequence.append(edges[uncompleted_sequence[i]])
                    last_road = uncompleted_sequence[i]
                    last_point = edges[uncompleted_sequence[i]][1]
                    i += 1
            else:
                completed_sequence.append(uncompleted_sequence[i])
                last_road = uncompleted_sequence[i]
                last_point = edges[uncompleted_sequence[i]][1]
                i += 1
        self.output_sequence = completed_sequence