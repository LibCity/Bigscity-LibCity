import os
from trafficdl.data.dataset import TrafficStateDataset
import scipy.sparse as sp
import torch
import numpy as np
import random
import torch.nn as nn


class KMeansClusterer(nn.Module):

    def __init__(self, F, cluster_num):
        '''
            #k均值聚类
        '''
        self.F = F.t()
        # F的列向量为特征向量  self.F行向量为特征向量
        self.cluster_num = cluster_num
        self.centers = self.__pick_start_point(self.F, cluster_num)

    def cluster(self):
        centers_groups = []
        centers_ind_groups = []
        for i in range(self.cluster_num):
            centers_groups.append(torch.tensor([]))
            centers_ind_groups.append([])
        item_ind = 0
        for item in self.F:
            distances = torch.pow(torch.sum((self.centers - item) ** 2, dim=1), 0.5)
            index = int(torch.argmin(distances))
            if len(centers_groups[index]) == 0:
                centers_groups[index] = item.reshape(1, -1)
            else:
                centers_groups[index] = torch.cat([centers_groups[index], item.reshape(1, -1)], axis=0)
            centers_ind_groups[index].append(item_ind)
            item_ind += 1
        new_center = self.centers
        for item in range(len(centers_groups)):
            new_center[item] = torch.sum(centers_groups[item], dim=0) / len(centers_groups[item])
        # 中心点未改变，说明达到稳态，结束递归
        if (self.centers != new_center).sum() == 0:
            return centers_groups, centers_ind_groups

        self.centers = new_center
        return self.cluster()

    def __pick_start_point(self, F, cluster_num):

        if cluster_num < 0 or cluster_num > F.shape[1]:
            raise Exception("簇数设置有误")

        # 随机点的下标
        indexes = random.sample(np.arange(0, F.shape[0]).tolist(), cluster_num)
        centers = torch.unique(F, dim=0)[indexes]
        return centers


class HGCNDataset(TrafficStateDataset):

    def __init__(self, config):
        super().__init__(config)
        self.cache_file_name = os.path.join('./trafficdl/cache/dataset_cache/',
                                            'point_based_{}.npz'.format(self.parameters_str))

        # 聚类中心（区域）的个数
        self.cluster_nodes = self.config.get('cluster_nodes', 40)
        # 区域邻接矩阵初始化
        self.adj_mx_cluster = None
        self.calculate_adj_mx_cluster()
        # 聚类的中心向量矩阵，区域[节点]
        self.centers_groups, self.centers_ind_groups = self.get_cluster()
        # trans矩阵
        self.transmit = np.zeros((self.num_nodes, self.cluster_nodes), dtype=np.float32)
        for j in range(self.cluster_nodes):
            for i in self.centers_ind_groups[j]:
                self.transmit[i][j] = 1
        self.transmit = torch.tensor(self.transmit)

    def _load_geo(self):
        """
        加载.geo文件，格式[geo_id, type, coordinates, properties(若干列)]
        """
        super()._load_geo()

    def _load_rel(self):
        """
        加载.rel文件，格式[rel_id, type, origin_id, destination_id, properties(若干列)]
        Returns:
            np.ndarray: self.adj_mx, N*N的邻接矩阵
        """
        super()._load_rel()

    def _load_dyna(self, filename):
        """
        加载.dyna文件，格式[dyna_id, type, time, entity_id, properties(若干列)]
        其中全局参数`data_col`用于指定需要加载的数据的列，不设置则默认全部加载
        Args:
            filename(str): 数据文件名，不包含后缀
        Returns:
            np.ndarray: 数据数组, 3d-array (len_time, num_nodes, feature_dim)
        """
        return super()._load_dyna_3d(filename)

    def calculate_normalized_laplacian(self, adj):
        """
        #计算正则化后的拉普拉斯矩阵
        # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
        # D = diag(A 1)
        :param adj:
        :return:
        """
        adj = sp.coo_matrix(adj)
        d = np.array(adj.sum(1))
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(
            d_mat_inv_sqrt).tocoo().toarray()
        return torch.tensor(normalized_laplacian)

    def calculate_F(self, mat, k):
        ''''#求特征值'''
        e, v = torch.eig(mat, eigenvectors=True)
        '''
        #取前k个特征向量构成F
        #k为聚类中心的个数
        '''
        F = v[0:k].T
        return F

    def get_cluster(self):
        '''
        :return: 聚类后的中心向量矩阵，区域[节点]
        '''
        L = self.calculate_normalized_laplacian(self.adj_mx)
        F = self.calculate_F(L, self.cluster_nodes)
        Cluster = KMeansClusterer(F, self.cluster_nodes)
        return Cluster.cluster()

    def calculate_adj_mx_cluster(self):
        '''
        :return: #聚类结果[cluster_num][]   聚类标识[cluster_num][]
        '''
        self._logger.info("Start Calculate the adj_max_cluster!")

        centers_groups, centers_ind_groups = self.get_cluster()
        self.adj_mx_cluster = np.zeros((self.cluster_nodes, self.cluster_nodes), dtype=np.float32)
        if self.init_weight_inf_or_zero.lower() == 'inf':
            self.adj_mx_cluster[:] = np.inf
        for i in range(self.cluster_nodes):
            for j in range(self.cluster_nodes):
                cluster_sum = 0
                for vi in centers_ind_groups[i]:
                    for vj in centers_ind_groups[j]:
                        cluster_sum += self.adj_mx[vi][vj]
                self.adj_mx_cluster[i][j] = cluster_sum
        distances = self.adj_mx_cluster[~np.isinf(self.adj_mx_cluster)].flatten()
        std = distances.std()
        self.adj_mx_cluster = torch.tensor(np.exp(-np.square(self.adj_mx_cluster / std)))
        self.adj_mx_cluster[self.adj_mx_cluster < self.weight_adj_epsilon] = 0

    def _add_external_information(self, df, ext_data=None):
        """
        增加外部信息（一周中的星期几/day of week，一天中的某个时刻/time of day，外部数据）
        Args:
            df(np.ndarray): 交通状态数据多维数组, (len_time, num_nodes, feature_dim)
            ext_data(np.ndarray): 外部数据
        Returns:
            np.ndarray: 融合后的外部数据和交通状态数据, (len_time, num_nodes, feature_dim_plus)
        """
        return super()._add_external_information_3d(df, ext_data)

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
