import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pickle
import os

from libcity.data.dataset import TrafficStatePointDataset
# from libcity.data.dataset import TrafficStateGridDataset


"""
主要功能是定义了一种计算语义邻接矩阵的方法，并缓存到dataset_cache/，并通过get_data_feature返回
注意这里要求邻接矩阵构造成0-1矩阵，在STAGGCNDataset.json中进行了设置
STAGGCNDataset既可以继承TrafficStatePointDataset，也可以继承TrafficStateGridDataset以处理网格数据
修改成TrafficStateGridDataset时，只需要修改：
1.TrafficStatePointDataset-->TrafficStateGridDataset
2.self.use_row_column = False, 可以加到self.parameters_str中
3.计算 DTW 邻接矩阵前需要 reshape 为 (time_len, num_nodes, feature)
"""


class STAGGCNDataset(TrafficStatePointDataset):

    def __init__(self, config):
        super().__init__(config)
        self.points_per_hour = 3600 // self.time_intervals  # 每小时的时间片数
        self.period = 7 * 24 * self.points_per_hour  # 一周的时间点数目，间隔为5min，用于求dtw_edge_index
        self.edge_index = self.get_edge_index()
        self.load_from_local = self.config.get('load_from_local', True)
        cache_path = './libcity/cache/dataset_cache/dtw_edge_index_' + self.dataset + '.npz'
        if self.load_from_local and os.path.exists(cache_path):  # 提前算好了dtw_edge_index，并从本地导入
            with open(cache_path, 'rb') as f:
                self.dtw_edge_index = pickle.load(f)
        else:  # 临时求dtw_edge_index (临时求会耗时很久)
            self.dtw_edge_index = self.get_dtw_edge_index()
            with open(cache_path, 'wb') as f:
                pickle.dump(self.dtw_edge_index, f)

    # 返回语义邻接边集（该部分直接截取自源码，做为函数直接调用）
    # 根据.dyna文件求取语义邻接矩阵，通过调用edge_index_func将其转化为语义邻接边集
    def get_dtw_edge_index(self):
        i = 0
        for filename in self.data_files:
            if i == 0:
                df = self._load_dyna(filename)  # (len_time, node_num, feature_dim)
            else:
                df = np.concatenate((df, self._load_dyna(filename)), axis=0)
            i += 1
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

        count_min, count_max = self.num_nodes, 0
        count_zero = 0
        count_avg = 0

        matrix = np.identity(self.num_nodes)

        for i in range(self.num_nodes):
            dtw_count_i = 0
            for j in range(self.num_nodes):
                if dtw_matrix[i][j] > dtw_threshold:
                    dtw_count_i += 1
                    matrix[i][j] = 1

            count_avg += dtw_count_i
            if dtw_count_i == 1:
                count_zero += 1
            if dtw_count_i > count_max:
                count_max = dtw_count_i
            if dtw_count_i < count_min:
                count_min = dtw_count_i

        return self.edge_index_func(matrix)

    # 返回空间邻接边集
    # 根据.geo文件求取空间邻接矩阵，通过调用edge_index_func将其转化为空间邻接边集
    def get_edge_index(self):
        return self.edge_index_func(self.adj_mx)

    # 用于将邻接矩阵转化为邻接边集
    def edge_index_func(self, matrix):
        # print(matrix, matrix.max(), matrix.min())
        a, b = [], []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i][j] == 1:  # matrix是0-1矩阵
                    a.append(i)
                    b.append(j)
        edge = [a, b]
        edge_index = torch.tensor(edge, dtype=torch.long)
        return edge_index

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"scaler": self.scaler, "adj_mx": self.adj_mx,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "ext_dim": self.ext_dim,
                "edge_index": self.edge_index,  # 将空间邻接边集作为data_feature返回
                "dtw_edge_index": self.dtw_edge_index,  # 将语义邻接边集作为data_feature返回
                "num_batches": self.num_batches}
