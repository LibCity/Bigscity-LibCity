import os
import json
import datetime
import pandas as pd
import networkx as nx
from libcity.utils import ensure_dir
from logging import getLogger
from libcity.evaluator.abstract_evaluator import AbstractEvaluator


class MapMatchingEvaluator(AbstractEvaluator):
    def __init__(self, config):
        self.metrics = config['metrics']  # 评估指标, 是一个 list
        self.allowed_metrics = ['RMF', 'AN', 'AL']
        self.config = config
        self.save_modes = config.get('save_modes', ['csv', 'json'])
        self.multi_traj = config.get('multi_traj', False)
        self.evaluate_result = {}  # 每一种指标的结果
        self._check_config()
        self._logger = getLogger()
        self.rel_info = {}

        self.rd_nwk = None
        self.route = None
        self.result = None
        self.merged_result = None
        self.lcs = None

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        for metric in self.metrics:
            if metric not in self.allowed_metrics:
                raise ValueError('the metric {} is not allowed in TrafficStateEvaluator'.format(str(metric)))

    def collect(self, batch):
        """
        Args:
            batch: {'route': ground truth, 'result': matched result, 'rd_nwk': road network}

        set self.rd_nwk, self.result, self.rel_info,
        set self.merged_result based on self.result and self.rd_nwk
        set truth_sequence and self.lcs if we have ground truth
        """
        # self.rd_nwk
        self.rd_nwk = batch["rd_nwk"]

        # self.route (optional)
        if batch["route"] is not None:
            self.route = batch["route"]
        else:
            self.route = None

        # self.result
        self.result = batch["result"]

        # self.rel_info
        for point1 in self.rd_nwk.adj:
            for point2 in self.rd_nwk.adj[point1]:
                geo = self.rd_nwk.adj[point1][point2]
                self.rel_info[geo['geo_id']] = {}
                self.rel_info[geo['geo_id']]["distance"] = geo['distance']
                self.rel_info[geo['geo_id']]['point1'] = point1
                self.rel_info[geo['geo_id']]['point2'] = point2

        # self.merged_result
        self.merge_result()
        if self.route is not None:
            # find the longest common subsequence
            self.find_lcs()

    def evaluate(self):
        """
        evaluation saved at self.evaluate_result
        """
        for usr_id, usr_value in self.route.items():
            for traj_id, route in usr_value.items():
                route = route[:, 1]
                lcs = self.lcs[usr_id][traj_id]
                merged_result = self.merged_result[usr_id][traj_id]
                if 'RMF' in self.metrics:
                    d_plus = 0
                    d_sub = 0
                    d_total = 0
                    for rel_id in route:
                        d_total += self.rel_info[rel_id]['distance']
                    i = j = k = 0
                    while i < len(lcs):
                        while route[j] != lcs[i]:
                            d_sub += self.rel_info[route[j]]['distance']
                            j += 1
                        i += 1
                        j += 1
                    i = 0
                    while j < len(route):
                        d_sub += self.rel_info[route[j]]['distance']
                        j += 1
                    while i < len(lcs):
                        while merged_result[k] != lcs[i]:
                            d_plus += self.rel_info[merged_result[k]]['distance']
                            k += 1
                        i += 1
                        k += 1
                    while k < len(merged_result):
                        d_sub += self.rel_info[merged_result[k]]['distance']
                        k += 1

                    RMF = (d_plus + d_sub) / d_total

                    if usr_id not in self.evaluate_result.keys():
                        self.evaluate_result[usr_id] = {traj_id: {'RMF': RMF}}
                    else:
                        self.evaluate_result[usr_id][traj_id] = {'RMF': RMF}

                if 'AN' in self.metrics:
                    AN = len(lcs) / len(route)
                    self.evaluate_result[usr_id][traj_id]['AN'] = AN

                if 'AL' in self.metrics:
                    d_lcs = 0
                    d_tru = 0
                    for rel_id in lcs:
                        d_lcs += self.rel_info[rel_id]['distance']
                    for rel_id in route:
                        d_tru += self.rel_info[rel_id]['distance']

                    AL = d_lcs / d_tru

                    self.evaluate_result[usr_id][traj_id]['AL'] = AL

    def _save_atom(self, save_path, filename):
        """
        generate dyna
        """

        # path
        save_path = os.path.join(save_path, filename)
        ensure_dir(save_path)

        # open dyna
        dyna_file = open(os.path.join(save_path, filename + '_reult.dyna'), 'w')

        # title
        if self.multi_traj:
            dyna_file.write('dyna_id,type,time,entity_id,location,traj_id\n')
        else:
            dyna_file.write('dyna_id,type,time,entity_id,location\n')

        # dyna
        dyna_type = 'trajectory'
        dyna_id = 0
        for usr_id, usr_value in self.merged_result.items():
            for traj_id, merged_result in usr_value.items():
                for rel_id in merged_result:
                    if self.multi_traj:
                        dyna_file.write(str(dyna_id) + ',' + dyna_type + ',' + '' + ','
                                        + str(usr_id) + ',' + str(rel_id) + ',' + str(traj_id) + '\n')
                    else:
                        dyna_file.write(str(dyna_id) + ',' + dyna_type + ',' + '' + ','
                                        + str(usr_id) + ',' + str(rel_id) + '\n')
                    dyna_id += 1

        # close
        dyna_file.close()

        # config
        config = dict()
        config['geo'] = dict()
        config['geo']['including_types'] = ['LineString']
        config['geo']['LineString'] = dict()
        config['rel'] = dict()
        config['rel']['including_types'] = ['geo']
        config['rel']['geo'] = dict()
        config['usr'] = dict()
        config['usr']['properties'] = dict()
        config['info'] = dict()
        config['info']['geo_file'] = self.config.get('geo_file')
        config['info']['rel_file'] = self.config.get('rel_file')
        config['info']['dyna_file'] = self.config.get('dyna_file')
        config['info']['usr_file'] = self.config.get('usr_file')
        json.dump(config, open(os.path.join(save_path, 'config.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=4)

    def save_result(self, save_path, filename=None):
        """
        将评估结果保存到 save_path 文件夹下的 filename 文件中

        Args:
            save_path: 保存路径
            filename: 保存文件名
            yyyy_mm_dd_hh_mm_ss_model_dataset_result.geo .rel .dyna: 模型输出(原子文件)
            yyyy_mm_dd_hh_mm_ss_model_dataset_result.csv: 模型原始输出
            yyyy_mm_dd_hh_mm_ss_model_dataset_result.json(geojson): 原始输出扩充得到的连通路径
            yyyy_mm_dd_hh_mm_ss_model_dataset.json: 评价结果
            yyyy_mm_dd_hh_mm_ss_model_dataset.csv: 评价结果
        """
        ensure_dir(save_path)

        # set filename
        if filename is None:  # 使用时间戳
            filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                       self.config['model'] + '_' + self.config['dataset']

        # yyyy_mm_dd_hh_mm_ss_model_dataset_result.geo .rel .dyna: 模型输出(原子文件)
        self._save_atom(save_path, filename)

        # yyyy_mm_dd_hh_mm_ss_model_dataset_result.json: 模型输出(geojson)
        self._logger.info('geojson is saved at ' +
                          os.path.join(save_path, '{}_result.json'.format(filename)))
        geojson_obj = {'type': "FeatureCollection", 'features': []}
        for usr_id, usr_value in self.merged_result.items():
            for traj_id, merged_result in usr_value.items():
                feature_i = dict()
                feature_i['type'] = 'Feature'
                feature_i['properties'] = {'usr_id': usr_id, 'traj_id': traj_id}
                feature_i['geometry'] = {}
                feature_i['geometry']['type'] = 'LineString'
                feature_i['geometry']['coordinates'] = []
                lat_last = None
                lon_last = None
                for rel_id in merged_result:
                    lat_origin = self.rd_nwk.nodes[self.rel_info[rel_id]["point1"]]['lat']
                    lon_origin = self.rd_nwk.nodes[self.rel_info[rel_id]["point1"]]['lon']
                    lat_destination = self.rd_nwk.nodes[self.rel_info[rel_id]["point2"]]['lat']
                    lon_destination = self.rd_nwk.nodes[self.rel_info[rel_id]["point2"]]['lon']
                    if lat_last is None and lon_last is None:
                        feature_i['geometry']['coordinates'].append([lon_origin, lat_origin])
                        feature_i['geometry']['coordinates'].append([lon_destination, lat_destination])
                        lat_last = lat_destination
                        lon_last = lon_destination
                    else:
                        if lat_last == lat_origin and lon_last == lon_origin:
                            feature_i['geometry']['coordinates'].append([lon_destination, lat_destination])
                            lat_last = lat_destination
                            lon_last = lon_destination
                        else:
                            feature_i['geometry']['coordinates'].append([lon_origin, lat_origin])
                            feature_i['geometry']['coordinates'].append([lon_destination, lat_destination])
                            lat_last = lat_destination
                            lon_last = lon_destination
                geojson_obj['features'].append(feature_i)
        json.dump(geojson_obj, open(save_path + '/' + filename + '_result.json', 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=4)

        if self.route is not None:

            # evaluate
            self.evaluate()

            # yyyy_mm_dd_hh_mm_ss_model_dataset.json: 评价结果
            if 'json' in self.save_modes:
                self._logger.info('Evaluate result is ' + json.dumps(self.evaluate_result))
                with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
                    json.dump(self.evaluate_result, f, indent=4)
                self._logger.info('Evaluate result is saved at ' +
                                  os.path.join(save_path, '{}.json'.format(filename)))

            # yyyy_mm_dd_hh_mm_ss_model_dataset.csv: 评价结果
            csv_res = []
            if 'csv' in self.save_modes:
                for usr_id, usr_value in self.evaluate_result.items():
                    for traj_id, _ in usr_value.items():
                        csv_res_i = [usr_id, traj_id]
                        for metric in self.metrics:
                            csv_res_i.append(self.evaluate_result[usr_id][traj_id][metric])
                        csv_res.append(csv_res_i)
                df = pd.DataFrame(csv_res)
                df.columns = ['usr_id', 'traj_id'] + self.allowed_metrics
                df.to_csv(os.path.join(save_path, '{}.csv'.format(filename)), index=False)
                self._logger.info('Evaluate result is saved at ' +
                                  os.path.join(save_path, '{}.csv'.format(filename)))
                self._logger.info("\n" + str(df))

    def clear(self):
        pass

    def find_lcs(self):
        """
        self.merged_result + self.route => self.lcs
        Returns:

        """
        self.lcs = {}

        for usr_id, usr_value in self.route.items():
            for traj_id, result in usr_value.items():
                seq1 = result[:, 1]
                seq2 = self.merged_result[usr_id][traj_id]
                len1, len2 = len(seq1), len(seq2)
                res = [[0 for _ in range(len1 + 1)] for _ in range(len2 + 1)]  # python 初始化二维数组 [len1+1],[len2+1]
                for i in range(1, len2 + 1):  # 开始从1开始，到len2+1结束
                    for j in range(1, len1 + 1):  # 开始从1开始，到len2+1结束
                        if seq2[i - 1] == seq1[j - 1]:
                            res[i][j] = res[i - 1][j - 1] + 1
                        else:
                            res[i][j] = max(res[i - 1][j], res[i][j - 1])
                lcs = []
                i, j = len2, len1
                while i > 0 and j > 0:
                    # 开始从1开始，到len2+1结束
                    if seq2[i - 1] == seq1[j - 1]:
                        lcs.append(seq2[i - 1])
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
                if usr_id not in self.lcs.keys():
                    self.lcs[usr_id] = {traj_id: lcs}
                else:
                    self.lcs[usr_id][traj_id] = lcs

    def merge_result(self):
        """
        merge breaks in self.result.
        and the output will be saved at self.merged_result
        """
        self.merged_result = {}
        for usr_id, usr_value in self.result.items():
            for traj_id, result in usr_value.items():
                result = list(result[:, 1])
                result = list(filter(None, result))
                merged_result = []
                i = 0
                last_road = None
                last_point = None
                while i < len(result):
                    if last_road is not None:
                        if last_road == result[i]:
                            i += 1
                        else:
                            if last_point != self.rel_info[result[i]]['point1']:
                                try:
                                    path = nx.dijkstra_path(self.rd_nwk,
                                                            source=last_point,
                                                            target=self.rel_info[result[i]]['point1'],
                                                            weight='distance')
                                    j = 0
                                    while j < len(path) - 1:
                                        point1 = path[j]
                                        point2 = path[j + 1]
                                        for rel_id in self.rel_info.keys():
                                            if self.rel_info[rel_id]["point1"] == point1 and \
                                                    self.rel_info[rel_id]["point2"] == point2:
                                                merged_result.append(rel_id)
                                                break
                                        j += 1
                                    merged_result.append(result[i])
                                except:
                                    # shortest_path does not exist
                                    merged_result.append(result[i])
                            else:
                                merged_result.append(result[i])
                            last_road = result[i]
                            last_point = self.rel_info[result[i]]['point2']
                            i += 1
                    else:
                        merged_result.append(result[i])
                        last_road = result[i]
                        last_point = self.rel_info[result[i]]['point2']
                        i += 1
                if usr_id not in self.merged_result.keys():
                    self.merged_result[usr_id] = {traj_id: merged_result}
                else:
                    self.merged_result[usr_id][traj_id] = merged_result
