import os
from libcity.data.dataset import TrafficStatePointDataset
from sklearn.cluster import SpectralClustering
import torch
import numpy as np


class HGCNDataset(TrafficStatePointDataset):

    def __init__(self, config):
        super().__init__(config)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'point_based_{}.npz'.format(self.parameters_str))

        # 聚类中心（区域）的个数
        self.cluster_nodes = self.config.get('cluster_nodes', 40)
        # 区域邻接矩阵初始化
        self.adj_mx_cluster = None
        # 聚类的中心向量矩阵，区域[节点]
        self.centers_ind_groups = self.get_cluster()
        self.calculate_adj_mx_cluster()
        # trans矩阵
        self.transmit = np.zeros((self.num_nodes, self.cluster_nodes), dtype=np.float32)
        for j in range(self.cluster_nodes):
            for i in self.centers_ind_groups[j]:
                self.transmit[i][j] = 1
        self.transmit = torch.tensor(self.transmit)

    def get_cluster(self):
        '''
        :return: 聚类后的中心向量矩阵，区域[节点]
        '''
        self._logger.info("Start Calculate the adj_max_cluster!")
        sc = SpectralClustering(n_clusters=self.cluster_nodes,
                                affinity="precomputed",
                                assign_labels="discretize")
        sc.fit(self.adj_mx)
        labels = sc.labels_.tolist()
        groups = [[] for i in range(self.cluster_nodes)]
        for i in range(self.cluster_nodes):
            for j in range(len(labels)):
                if labels[j] == i:
                    groups[i].append(j)
        return groups

    def calculate_adj_mx_cluster(self):
        '''
        :return: #聚类结果[cluster_num][]   聚类标识[cluster_num][]
        '''
        self.adj_mx_cluster = np.zeros((self.cluster_nodes, self.cluster_nodes), dtype=np.float32)
        if self.init_weight_inf_or_zero.lower() == 'inf':
            self.adj_mx_cluster[:] = np.inf
        for i in range(self.cluster_nodes):
            for j in range(self.cluster_nodes):
                cluster_sum = 0
                for vi in self.centers_ind_groups[i]:
                    for vj in self.centers_ind_groups[j]:
                        cluster_sum += self.adj_mx[vi][vj]
                self.adj_mx_cluster[i][j] = cluster_sum
        distances = self.adj_mx_cluster[~np.isinf(self.adj_mx_cluster)].flatten()
        std = distances.std()
        self.adj_mx_cluster = np.exp(-np.square(self.adj_mx_cluster / std))
        self.adj_mx_cluster[self.adj_mx_cluster < self.weight_adj_epsilon] = 0

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度
        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"scaler": self.scaler, "adj_mx": self.adj_mx,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "adj_mx_cluster": self.adj_mx_cluster,
                "centers_ind_groups": self.centers_ind_groups, "transmit": self.transmit}
