import math
import os
import datetime
import numpy as np
import pandas as pd
from fastdtw import fastdtw
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from libcity.data.dataset import TrafficStatePointDataset
from libcity.utils import ensure_dir


class STWaveDataset(TrafficStatePointDataset):

    def __init__(self, config):
        self.normalized_k = config.get("normalized_k", 0.1)
        super().__init__(config)
        self.heads = config.get("heads", 8)
        self.dims = config.get("dims", 16)

    def _load_rel(self):
        def get_adjacency_matrix(distance_df, sensor_id_to_ind, normalized_k=0.1):
            num_sensors = len(sensor_ids)
            dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
            dist_mx[:] = np.inf

            # Fills cells in the matrix with distances
            for row in distance_df.values:
                if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
                    continue
                dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

            # Calculates the standard deviation as theta
            distances = dist_mx[~np.isinf(dist_mx)].flatten()
            std = distances.std()
            adj_mx = np.exp(-np.square(dist_mx / std))

            # Sets entries that lower than a threshold to zero for sparsity
            adj_mx[adj_mx < normalized_k] = 0
            return adj_mx

        distance_df = pd.read_csv(self.data_path + self.rel_file + '.rel',
                                  dtype={'origin_id': 'str', 'destination_id': 'str'})
        distance_df = distance_df.drop(['rel_id', 'type'], axis=1)
        t = distance_df['origin_id'] == distance_df['destination_id']
        if np.sum(t) != 0:
            for i in range(len(t)):
                if t[i] == True:
                    distance_df = distance_df.drop([i])

        # Builds sensor id to index map
        sensor_id_to_ind = {}
        sensor_ids = distance_df['origin_id'].tolist() + distance_df['destination_id'].tolist()
        sensor_ids = list(set(sensor_ids))
        sensor_ids = [int(i) for i in sensor_ids]

        sensor_ids = [str(i) for i in range(len(sensor_ids))]
        for i, sensor_id in enumerate(sensor_ids):
            sensor_id_to_ind[sensor_id] = i

        # Generate adjacency matrix
        adj_mx = get_adjacency_matrix(distance_df, sensor_id_to_ind, self.normalized_k)
        adj_mx += np.eye(len(sensor_ids))
        self.adj_mx = adj_mx

    def _add_external_information(self, df, ext_data=None):
        return self._add_external_information_3d(df, ext_data)

    def _add_external_information_3d(self, df, ext_data=None):
        """
        增加外部信息（一周中的星期几/day of week，一天中的某个时刻/time of day，外部数据）

        Args:
            df(np.ndarray): 交通状态数据多维数组, (len_time, num_nodes, feature_dim)
            ext_data(np.ndarray): 外部数据

        Returns:
            np.ndarray: 融合后的外部数据和交通状态数据, (len_time, num_nodes, feature_dim_plus)
        """
        num_samples, num_nodes, feature_dim = df.shape
        is_time_nan = pd.DatetimeIndex(self.timesolts).isnull().any()
        data_list = [df]
        if self.add_time_in_day and not is_time_nan:
            time_ind = (self.timesolts - self.timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        if self.add_day_in_week and not is_time_nan:
            dayofweek = []
            for day in self.timesolts.astype("datetime64[D]"):
                dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            day_in_week = np.tile(dayofweek, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(day_in_week)
        # 外部数据集
        if ext_data is not None:
            if not is_time_nan:
                indexs = []
                for ts in self.timesolts:
                    ts_index = self.idx_of_ext_timesolts[ts]
                    indexs.append(ts_index)
                select_data = ext_data[indexs]  # T * ext_dim 选出所需要的时间步的数据
                for i in range(select_data.shape[1]):
                    data_ind = select_data[:, i]
                    data_ind = np.tile(data_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                    data_list.append(data_ind)
            else:  # 没有给出具体的时间戳，只有外部数据跟原数据等长才能默认对接到一起
                if ext_data.shape[0] == df.shape[0]:
                    select_data = ext_data  # T * ext_dim
                    for i in range(select_data.shape[1]):
                        data_ind = select_data[:, i]
                        data_ind = np.tile(data_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                        data_list.append(data_ind)
        data = np.concatenate(data_list, axis=-1)
        return data

    def loadGraph(self, data):
        """
        loadGraph
        @param data: org flow data
        @return:
        """

        def construct_tem_adj(data, num_node):
            data_mean = np.mean([data[24 * 12 * i: 24 * 12 * (i + 1)] for i in range(data.shape[0] // (24 * 12))],
                                axis=0)
            data_mean = data_mean.squeeze().T
            dtw_distance = np.zeros((num_node, num_node))
            for i in range(num_node):
                for j in range(i, num_node):
                    dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
            for i in range(num_node):
                for j in range(i):
                    dtw_distance[i][j] = dtw_distance[j][i]

            nth = np.sort(dtw_distance.reshape(-1))[
                  int(np.log2(dtw_distance.shape[0]) * dtw_distance.shape[0]):
                  int(np.log2(dtw_distance.shape[0]) * dtw_distance.shape[0]) + 1]  # NlogN edges
            tem_matrix = np.zeros_like(dtw_distance)
            tem_matrix[dtw_distance <= nth] = 1
            tem_matrix = np.logical_or(tem_matrix, tem_matrix.T).astype(int)
            return tem_matrix

        def laplacian(W):
            """Return the Laplacian of the weight matrix."""
            # Degree matrix.
            d = W.sum(axis=0)
            # Laplacian matrix.
            d = 1 / np.sqrt(d)
            D = sp.diags(d, 0)
            I = sp.identity(d.size, dtype=W.dtype)
            L = I - D * W * D
            return L

        def largest_k_lamb(L, k):
            lamb, U = eigsh(L, k=k, which='LM')
            return (lamb, U)

        def get_eigv(adj, k):
            L = laplacian(adj)
            eig = largest_k_lamb(L, k)
            return eig

        # calculate spatial and temporal graph wavelets
        # adj = np.load(spatial_graph)
        # adj = adj + np.eye(adj.shape[0])
        data = data[..., 0]
        adj = self.adj_mx
        # gen tem_adj
        dims = self.heads * self.dims
        tem_adj = construct_tem_adj(data, adj.shape[0])
        spawave = get_eigv(adj, dims)
        temwave = get_eigv(tem_adj, dims)

        # derive neighbors
        sampled_nodes_number = int(math.log(adj.shape[0], 2))
        graph = csr_matrix(adj)
        dist_matrix = dijkstra(csgraph=graph)
        dist_matrix[dist_matrix == 0] = dist_matrix.max() + 10
        localadj = np.argpartition(dist_matrix, sampled_nodes_number, -1)[:, :sampled_nodes_number]

        return localadj, spawave, temwave

    def _generate_data(self):
        """
        加载数据文件(.dyna/.grid/.od/.gridod)和外部数据(.ext)，且将二者融合，以X，y的形式返回

        Returns:
            tuple: tuple contains:
                x(np.ndarray): 模型输入数据，(num_samples, input_length, ..., feature_dim) \n
                y(np.ndarray): 模型输出数据，(num_samples, output_length, ..., feature_dim)
        """
        # 处理多数据文件问题
        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:  # str
            data_files = [self.data_files].copy()
        # 加载外部数据
        if self.load_external and os.path.exists(self.data_path + self.ext_file + '.ext'):  # 外部数据集
            ext_data = self._load_ext()
        else:
            ext_data = None
        x_list, y_list = [], []
        df_list = []
        for filename in data_files:
            df = self._load_dyna(filename)  # (len_time, ..., feature_dim)
            if self.load_external:
                df = self._add_external_information(df, ext_data)
            x, y = self._generate_input_data(df)
            # x: (num_samples, input_length, ..., input_dim)
            # y: (num_samples, output_length, ..., output_dim)
            x_list.append(x)
            y_list.append(y)
            df_list.append(df)
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        df = np.concatenate(df_list)
        self._logger.info("Dataset created")
        self._logger.info("x shape: " + str(x.shape) + ", y shape: " + str(y.shape))
        return x, y, df

    def _split_train_val_test(self, x, y):
        """
        划分训练集、测试集、验证集，并缓存数据集

        Args:
            x(np.ndarray): 输入数据 (num_samples, input_length, ..., feature_dim)
            y(np.ndarray): 输出数据 (num_samples, input_length, ..., feature_dim)

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        test_rate = 1 - self.train_rate - self.eval_rate
        num_samples = x.shape[0]
        num_test = round(num_samples * test_rate)
        num_train = round(num_samples * self.train_rate)
        num_val = num_samples - num_test - num_train
        # train
        x_train, y_train = x[:num_train], y[:num_train]
        # val
        x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
        # test
        x_test, y_test = x[-num_test:], y[-num_test:]
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _generate_train_val_test(self):
        x, y, df = self._generate_data()
        num_samples = x.shape[0]
        x_train, y_train, x_val, y_val, x_test, y_test = self._split_train_val_test(x, y)
        # train data
        num_train = round(num_samples * self.train_rate)
        train_data = df[:num_train]
        self.localadj, self.spawave, self.temwave = self.loadGraph(train_data)
        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                self.cache_file_name,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                x_val=x_val,
                y_val=y_val,
                localadj=self.localadj,
                spawave=self.spawave,
                temwave=self.temwave,
            )
            self._logger.info('Saved at ' + self.cache_file_name)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _load_cache_train_val_test(self):
        """
        加载之前缓存好的训练集、测试集、验证集

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        self._logger.info('Loading ' + self.cache_file_name)
        cat_data = np.load(self.cache_file_name, allow_pickle=True)
        x_train = cat_data['x_train']
        y_train = cat_data['y_train']
        x_test = cat_data['x_test']
        y_test = cat_data['y_test']
        x_val = cat_data['x_val']
        y_val = cat_data['y_val']
        self.localadj = cat_data['localadj']
        self.spawave = cat_data['spawave']
        self.temwave = cat_data['temwave']
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))
        return x_train, y_train, x_val, y_val, x_test, y_test

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "ext_dim": self.ext_dim,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches, "localadj": self.localadj,
                "spawave": self.spawave, "temwave": self.temwave}
