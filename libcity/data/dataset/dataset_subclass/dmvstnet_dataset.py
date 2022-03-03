import collections

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pickle
import os

from libcity.data.dataset import TrafficStateGridDataset


class DMVSTNetDataset(TrafficStateGridDataset):

    def __init__(self, config):
        super().__init__(config)
        self.points_per_hour = 3600 // self.time_intervals  # 每小时的时间片数
        self.period = 7 * 24 * self.points_per_hour  # 一周的时间点数目，间隔为5min，用于求dtw_edge_index
        self.load_from_local = self.config.get('load_from_local', True)
        self.dtw_power = self.config.get('dtw_power', 0.75)
        cache_path = './libcity/cache/dataset_cache/dtw_graph_' + self.dataset + '.npz'
        if self.load_from_local and os.path.exists(cache_path):  # 提前算好了dtw_edge_index，并从本地导入
            with open(cache_path, 'rb') as f:
                self.dtw_graph = pickle.load(f)
        else:  # 临时求dtw_matirx (临时求会耗时很久)
            self.dtw_graph = self.get_dtw_grpah()
            with open(cache_path, 'wb') as f:
                pickle.dump(self.dtw_graph, f)

    # 返回语义邻接边集（该部分直接截取自源码，做为函数直接调用）
    # 根据.dyna文件求取语义邻接矩阵，通过调用edge_index_func将其转化为语义邻接边集
    def get_dtw_grpah(self):
        i = 0
        for filename in self.data_files:
            if i == 0:
                df = self._load_dyna(filename)  # (len_time, node_num, feature_dim)
            else:
                df = np.concatenate((df, self._load_dyna(filename)), axis=0)
            i += 1
        df = df.reshape((df.shape[0], self.num_nodes, -1))
        df = df[:, :, 0]
        line = df.shape[0]
        order = np.arange(line).reshape(line, 1)
        df = np.concatenate((df, order), axis=1)
        df = pd.DataFrame(df)
        df['symbol'] = df[self.num_nodes] % self.period

        for i in tqdm(range(self.period)):
            df_i = df[df['symbol'] == i]
            values_i = df_i.values[:, :-1]
            mean_i = np.mean(values_i, axis=0)[np.newaxis, :]
            if i == 0:
                mean = mean_i
            else:
                mean = np.concatenate((mean, mean_i), axis=0)

        mean = mean.T
        dtw_matrix = np.zeros((self.num_nodes, self.num_nodes))

        for index_x in tqdm(range(self.num_nodes)):
            for index_y in range(index_x, self.num_nodes):
                x = mean[index_x]
                y = mean[index_y]
                distance, _ = fastdtw(x, y, dist=euclidean)
                dtw_matrix[index_x][index_y] = distance

        for i in range(self.num_nodes):
            for j in range(0, i):
                dtw_matrix[i][j] = dtw_matrix[j][i]

        std = np.std(dtw_matrix)
        dtw_matrix = dtw_matrix / std
        dtw_matrix = np.exp(-1 * dtw_matrix)

        dtw_threshold = 0.83

        edgedistdict = collections.defaultdict(int)
        nodedistdict = collections.defaultdict(int)

        weightsdict = collections.defaultdict(int)
        nodedegrees = collections.defaultdict(int)

        weightsum = 0
        negprobsum = 0

        for i in range(self.num_nodes):
            dtw_count_i = 0
            for j in range(self.num_nodes):
                weight = dtw_matrix[i][j]
                if weight > dtw_threshold:
                    dtw_count_i += 1
                    edgedistdict[tuple([i, j])] = weight
                    nodedistdict[i] += weight
                    weightsdict[tuple([i, j])] = weight
                    nodedegrees[i] += weight
                    weightsum += weight
                    negprobsum += np.power(weight, self.dtw_power)

        for node, outdegree in nodedistdict.items():
            nodedistdict[node] = np.power(outdegree, self.dtw_power) / negprobsum

        for edge, weight in edgedistdict.items():
            edgedistdict[edge] = weight // weightsum

        return [edgedistdict, nodedistdict, weightsdict, nodedegrees]

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"scaler": self.scaler, "adj_mx": self.adj_mx,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim, "ext_dim": self.ext_dim,

                "output_dim": self.output_dim, "len_row": self.len_row, "len_column": self.len_column,
                "dtw_graph": self.dtw_graph,  # 将语义邻接图作为data_feature返回
                "num_batches": self.num_batches}
