import math
import os

import numpy as np
import pandas as pd
from fastdtw import fastdtw
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from libcity.data.dataset import TrafficStatePointDataset
from libcity.data.utils import generate_dataloader
from libcity.utils import ensure_dir


class STWaveDataset(TrafficStatePointDataset):

    def __init__(self, config):
        self.normalized_k = config.get("normalized_k", 0.1)
        super().__init__(config)
        self.feature_name = {'X': 'float', 'y': 'float', 'TE': 'int'}
        self.vocab_size = 24 * 60 * 60 // config.get("time_intervals", 300)
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

    def _generate_split_te(self, num_step):
        """
        generate TE

        @return:
        """

        def seq2instance(data, P, Q):
            num_step, nodes, dims = data.shape
            num_sample = num_step - P - Q + 1
            x = np.zeros(shape=(num_sample, P, nodes, dims))
            y = np.zeros(shape=(num_sample, Q, nodes, dims))
            for i in range(num_sample):
                x[i] = data[i: i + P]
                y[i] = data[i + P: i + P + Q]
            return x, y

        TE = np.zeros([num_step, 2])
        TE[:, 1] = np.array([i % self.vocab_size for i in range(num_step)])
        TE[:, 0] = np.array([(i // self.vocab_size) % 7 for i in range(num_step)])
        TE_tile = np.repeat(np.expand_dims(TE, 1), self.num_nodes, 1)
        test_ratio = 1 - self.train_rate - self.eval_rate
        train_steps = round(self.train_rate * num_step)
        test_steps = round(test_ratio * num_step)
        val_steps = num_step - train_steps - test_steps
        trainTE = TE_tile[: train_steps]
        valTE = TE_tile[train_steps: train_steps + val_steps]
        testTE = TE_tile[-test_steps:]
        trainXTE, trainYTE = seq2instance(trainTE, self.input_window, self.output_window)
        valXTE, valYTE = seq2instance(valTE, self.input_window, self.output_window)
        testXTE, testYTE = seq2instance(testTE, self.input_window, self.output_window)
        trainTE = np.concatenate([trainXTE, trainYTE], axis=1)
        valTE = np.concatenate([valXTE, valYTE], axis=1)
        testTE = np.concatenate([testXTE, testYTE], axis=1)
        return trainTE, valTE, testTE

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
        # generate te
        trainTE, valTE, testTE = self._generate_split_te(num_samples)
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
                trainTE=trainTE,
                valTE=valTE,
                testTE=testTE,
                localadj=self.localadj,
                spawave=self.spawave,
                temwave=self.temwave,
            )
            self._logger.info('Saved at ' + self.cache_file_name)
        return x_train, y_train, x_val, y_val, x_test, y_test, trainTE, valTE, testTE

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
        trainTE = cat_data['trainTE']
        valTE = cat_data['valTE']
        testTE = cat_data['testTE']
        self.localadj = cat_data['localadj']
        self.spawave = cat_data['spawave']
        self.temwave = cat_data['temwave']
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))
        return x_train, y_train, x_val, y_val, x_test, y_test, trainTE, valTE, testTE

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        # 加载数据集
        x_train, y_train, x_val, y_val, x_test, y_test, trainTE, valTE, testTE = [], [], [], [], [], [], [], [], []
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train, y_train, x_val, y_val, x_test, y_test, trainTE, valTE, testTE = self._load_cache_train_val_test()
            else:
                x_train, y_train, x_val, y_val, x_test, y_test, trainTE, valTE, testTE = self._generate_train_val_test()
        # 在测试集上添加随机扰动
        if self.robustness_test:
            x_test = self._add_noise(x_test)
        # 数据归一化
        self.feature_dim = x_train.shape[-1]
        self.ext_dim = self.feature_dim - self.output_dim
        self.scaler = self._get_scalar(self.scaler_type,
                                       x_train[..., :self.output_dim], y_train[..., :self.output_dim])
        self.ext_scaler = self._get_scalar(self.ext_scaler_type,
                                           x_train[..., self.output_dim:], y_train[..., self.output_dim:])
        x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_val[..., :self.output_dim] = self.scaler.transform(x_val[..., :self.output_dim])
        y_val[..., :self.output_dim] = self.scaler.transform(y_val[..., :self.output_dim])
        x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])
        if self.normal_external:
            x_train[..., self.output_dim:] = self.ext_scaler.transform(x_train[..., self.output_dim:])
            y_train[..., self.output_dim:] = self.ext_scaler.transform(y_train[..., self.output_dim:])
            x_val[..., self.output_dim:] = self.ext_scaler.transform(x_val[..., self.output_dim:])
            y_val[..., self.output_dim:] = self.ext_scaler.transform(y_val[..., self.output_dim:])
            x_test[..., self.output_dim:] = self.ext_scaler.transform(x_test[..., self.output_dim:])
            y_test[..., self.output_dim:] = self.ext_scaler.transform(y_test[..., self.output_dim:])
        # 把训练集的X和y聚合在一起成为list，测试集验证集同理
        # x_train/y_train: (num_samples, input_length, ..., feature_dim)
        # train_data(list): train_data[i]是一个元组，由x_train[i]和y_train[i]组成
        train_data = list(zip(x_train, y_train, trainTE))
        eval_data = list(zip(x_val, y_val, valTE))
        test_data = list(zip(x_test, y_test, testTE))
        # 转Dataloader
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample)
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

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
