import os
import json
import datetime
import pandas as pd

import networkx as nx

from libtraffic.utils import ensure_dir
from logging import getLogger
from libtraffic.evaluator.abstract_evaluator import AbstractEvaluator


class MapMatchingEvaluator(AbstractEvaluator):
    def __init__(self, config):
        self.metrics = config['metrics']  # 评估指标, 是一个 list
        self.allowed_metrics = ['RMF', 'AN', 'AL']
        self.config = config
        self.save_modes = config.get('save_modes', ['csv', 'json'])
        self.evaluate_result = {}  # 每一种指标的结果
        self._check_config()
        self._logger = getLogger()
        self.rel_info = {}

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        for metric in self.metrics:
            if metric not in self.allowed_metrics:
                raise ValueError('the metric {} is not allowed in TrafficStateEvaluator'.format(str(metric)))

    def collect(self, batch):
        self.rd_nwk = batch["rd_nwk"]
        if batch["route"] is not None:
            self.truth_sequence = list(batch["route"])
        else:
            self.truth_sequence = None
        self.result = batch["result"]
        for point1 in self.rd_nwk.adj:
            for point2 in self.rd_nwk.adj[point1]:
                rel = self.rd_nwk.adj[point1][point2]
                self.rel_info[rel['rel_id']] = {}
                self.rel_info[rel['rel_id']]["distance"] = rel['distance']
                self.rel_info[rel['rel_id']]['point1'] = point1
                self.rel_info[rel['rel_id']]['point2'] = point2
        self.find_completed_sequence()
        if self.truth_sequence is not None:
            # find the longest common subsequence
            self.find_lcs()

    def evaluate(self):
        if 'RMF' in self.metrics:
            d_plus = 0
            d_sub = 0
            d_total = 0
            for rel_id in self.truth_sequence:
                d_total += self.rel_info[rel_id]['distance']
            i = j = k = 0
            while i < len(self.lcs):
                while self.truth_sequence[j] != self.lcs[i]:
                    d_sub += self.rel_info[self.truth_sequence[j]]['distance']
                    j += 1
                i += 1
                j += 1
            i = 0
            while j < len(self.truth_sequence):
                d_sub += self.rel_info[self.truth_sequence[j]]['distance']
                j += 1
            while i < len(self.lcs):
                while self.output_sequence[k] != self.lcs[i]:
                    d_plus += self.rel_info[self.output_sequence[j]]['distance']
                    k += 1
                i += 1
                k += 1
            while k < len(self.output_sequence):
                d_sub += self.rel_info[self.output_sequence[k]]['distance']
                k += 1
            self.evaluate_result['RMF'] = (d_plus + d_sub) / d_total
        if 'AN' in self.metrics:
            self.evaluate_result['AN'] = len(self.lcs) / len(self.truth_sequence)
        if 'AL' in self.metrics:
            d_lcs = 0
            d_tru = 0
            for rel_id in self.lcs:
                d_lcs += self.rel_info[rel_id]['distance']
            for rel_id in self.truth_sequence:
                d_tru += self.rel_info[rel_id]['distance']
            self.evaluate_result['AL'] = d_lcs / d_tru

    def save_result(self, save_path, filename=None):
        """
        将评估结果保存到 save_path 文件夹下的 filename 文件中

        Args:
            save_path: 保存路径
            filename: 保存文件名
            yyyy_mm_dd_hh_mm_ss_model_dataset_result.csv: 模型原始输出
            yyyy_mm_dd_hh_mm_ss_model_dataset_result.json(geojson): 原始输出扩充得到的连通路径
            yyyy_mm_dd_hh_mm_ss_model_dataset.json: 评价结果
            yyyy_mm_dd_hh_mm_ss_model_dataset.csv: 评价结果
        """
        ensure_dir(save_path)
        if filename is None:  # 使用时间戳
            filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                       self.config['model'] + '_' + self.config['dataset']
        dataframe = {'dyna_id': [], 'rel_id': []}

        self._logger.info('Result is saved at ' +
                          os.path.join(save_path, '{}_result.csv'.format(filename)))
        for line in self.result:
            dataframe['dyna_id'].append(str(line[0]))
            dataframe['rel_id'].append(str(line[1]))
        dataframe = pd.DataFrame(dataframe)
        dataframe.to_csv(os.path.join(save_path, '{}_result.csv'.format(filename)), index=False)

        self._logger.info('Completed sequence is saved at ' +
                          os.path.join(save_path, '{}_result.json'.format(filename)))
        evaluate_result = dict()
        evaluate_result['type'] = 'Feature'
        evaluate_result['properties'] = {}
        evaluate_result['geometry'] = {}
        evaluate_result['geometry']['type'] = 'LineString'
        evaluate_result['geometry']['coordinates'] = []
        lat_last = None
        lon_last = None
        for rel_id in self.output_sequence:
            lat_origin = self.rd_nwk.nodes[self.rel_info[rel_id]["point1"]]['lat']
            lon_origin = self.rd_nwk.nodes[self.rel_info[rel_id]["point1"]]['lon']
            lat_destination = self.rd_nwk.nodes[self.rel_info[rel_id]["point2"]]['lat']
            lon_destination = self.rd_nwk.nodes[self.rel_info[rel_id]["point2"]]['lon']
            if lat_last is None and lon_last is None:
                evaluate_result['geometry']['coordinates'].append([lon_origin, lat_origin])
                evaluate_result['geometry']['coordinates'].append([lon_destination, lat_destination])
                lat_last = lat_destination
                lon_last = lon_destination
            else:
                if lat_last == lat_origin and lon_last == lon_origin:
                    evaluate_result['geometry']['coordinates'].append([lon_destination, lat_destination])
                    lat_last = lat_destination
                    lon_last = lon_destination
                else:
                    evaluate_result['geometry']['coordinates'].append([lon_origin, lat_origin])
                    evaluate_result['geometry']['coordinates'].append([lon_destination, lat_destination])
                    lat_last = lat_destination
                    lon_last = lon_destination
        json.dump(evaluate_result, open(save_path + '/' + filename + '_result.json', 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=4)

        if self.truth_sequence is not None:
            self.evaluate()
            if 'json' in self.save_modes:
                self._logger.info('Evaluate result is ' + json.dumps(self.evaluate_result))
                with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
                    json.dump(self.evaluate_result, f)
                self._logger.info('Evaluate result is saved at ' +
                                  os.path.join(save_path, '{}.json'.format(filename)))

            dataframe = {}
            if 'csv' in self.save_modes:
                for metric in self.metrics:
                    dataframe[metric] = [self.evaluate_result[metric]]
                dataframe = pd.DataFrame(dataframe)
                dataframe.to_csv(os.path.join(save_path, '{}.csv'.format(filename)), index=False)
                self._logger.info('Evaluate result is saved at ' +
                                  os.path.join(save_path, '{}.csv'.format(filename)))
                self._logger.info("\n" + str(dataframe))

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
        while uncompleted_sequence.count(None) != 0:
            uncompleted_sequence.remove(None)
        completed_sequence = []
        i = 0
        last_road = None
        last_point = None
        while i < len(uncompleted_sequence):
            if last_road is not None:
                if last_road == uncompleted_sequence[i]:
                    i += 1
                else:
                    if last_point != self.rel_info[uncompleted_sequence[i]]['point1']:
                        try:
                            path = nx.dijkstra_path(self.rd_nwk,
                                                    source=last_point,
                                                    target=self.rel_info[uncompleted_sequence[i]]['point1'],
                                                    weight='distance')
                            j = 0
                            while j < len(path) - 1:
                                point1 = path[j]
                                point2 = path[j + 1]
                                for rel_id in self.rel_info.keys():
                                    if self.rel_info[rel_id]["point1"] == point1 and \
                                            self.rel_info[rel_id]["point2"] == point2:
                                        completed_sequence.append(rel_id)
                                        break
                                j += 1
                            completed_sequence.append(uncompleted_sequence[i])
                        except:
                            # shortest_path does not exist
                            completed_sequence.append(uncompleted_sequence[i])
                    else:
                        completed_sequence.append(uncompleted_sequence[i])
                    last_road = uncompleted_sequence[i]
                    last_point = self.rel_info[uncompleted_sequence[i]]['point2']
                    i += 1
            else:
                completed_sequence.append(uncompleted_sequence[i])
                last_road = uncompleted_sequence[i]
                last_point = self.rel_info[uncompleted_sequence[i]]['point2']
                i += 1
        self.output_sequence = completed_sequence
