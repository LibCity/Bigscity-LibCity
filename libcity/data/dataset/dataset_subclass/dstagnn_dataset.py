import os

import numpy as np
from scipy.optimize import linprog

from libcity.data.dataset import TrafficStatePointDataset
from libcity.utils import ensure_dir


class DSTAGNNDataset(TrafficStatePointDataset):

    def __init__(self, config):
        super().__init__(config)
        self.sparsity = config.get('sparsity', 0.01)
        self.period = config.get('period', 288)

    def gen_ag_adj(self, df):
        """
        创建AG图
        """

        def wasserstein_distance(p, q, D):
            A_eq = []
            for i in range(len(p)):
                A = np.zeros_like(D)
                A[i, :] = 1
                A_eq.append(A.reshape(-1))
            for i in range(len(q)):
                A = np.zeros_like(D)
                A[:, i] = 1
                A_eq.append(A.reshape(-1))
            A_eq = np.array(A_eq)
            b_eq = np.concatenate([p, q])
            D = np.array(D)
            D = D.reshape(-1)
            np.nan_to_num(A_eq, copy=False, nan=0.0, posinf=None, neginf=None)
            np.nan_to_num(D, copy=False, nan=0.0, posinf=None, neginf=None)
            result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
            myresult = result.fun

            return myresult

        def spatial_temporal_aware_distance(x, y):
            x, y = np.array(x), np.array(y)
            x_norm = (x ** 2).sum(axis=1, keepdims=True) ** 0.5
            y_norm = (y ** 2).sum(axis=1, keepdims=True) ** 0.5
            p = x_norm[:, 0] / x_norm.sum()
            q = y_norm[:, 0] / y_norm.sum()
            D = 1 - np.dot(x / x_norm, (y / y_norm).T)
            return wasserstein_distance(p, q, D)

        def spatial_temporal_similarity(x, y):
            return 1 - spatial_temporal_aware_distance(x, y)

        # start
        period = self.period
        sparsity = self.sparsity
        num_samples, ndim, _ = df.shape
        num_train = int(num_samples * self.train_rate)
        num_sta = int(num_train / period) * period
        data = df[:num_sta, :, :1].reshape([-1, period, ndim])

        d = np.zeros([ndim, ndim])
        for i in range(ndim):
            for j in range(i + 1, ndim):
                d[i, j] = spatial_temporal_similarity(data[:, :, i], data[:, :, j])

        sta = d + d.T
        adj = sta
        id_mat = np.identity(ndim)
        adjl = adj + id_mat
        adjlnormd = adjl / adjl.mean(axis=0)

        adj = 1 - adjl + id_mat
        A_adj = np.zeros([ndim, ndim])
        R_adj = np.zeros([ndim, ndim])
        adj_percent = sparsity

        top = int(ndim * adj_percent)

        for i in range(adj.shape[0]):
            a = adj[i, :].argsort()[0:top]
            for j in range(top):
                A_adj[i, a[j]] = 1
                R_adj[i, a[j]] = adjlnormd[i, a[j]]

        for i in range(ndim):
            for j in range(ndim):
                if (i == j):
                    R_adj[i][j] = adjlnormd[i, j]

        A_adj = np.float64(A_adj > 0)
        R_adj = np.float64(R_adj > 0)
        return A_adj, R_adj

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
            df_list.append(df)
            x, y = self._generate_input_data(df)
            # x: (num_samples, input_length, ..., input_dim)
            # y: (num_samples, output_length, ..., output_dim)
            x_list.append(x)
            y_list.append(y)
        all_df = np.concatenate(df_list)
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        self._logger.info("final all df shape: " + str(all_df.shape))
        self.a_adj, self.r_adj = self.gen_ag_adj(all_df)
        self._logger.info("Dataset created")
        self._logger.info("x shape: " + str(x.shape) + ", y shape: " + str(y.shape))
        self._logger.info("a_adj shape: " + str(self.a_adj.shape) + ", r_adj shape: " + str(self.r_adj.shape))
        return x, y

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
                a_adj=self.a_adj,
                r_adj=self.r_adj
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
        cat_data = np.load(self.cache_file_name)
        x_train = cat_data['x_train']
        y_train = cat_data['y_train']
        x_test = cat_data['x_test']
        y_test = cat_data['y_test']
        x_val = cat_data['x_val']
        y_val = cat_data['y_val']
        self.a_adj = cat_data['a_adj']
        self.r_adj = cat_data['r_adj']
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))
        return x_train, y_train, x_val, y_val, x_test, y_test

    def get_data_feature(self):
        """
        返回数据集特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "num_nodes": self.num_nodes,
                "adj_TMD": self.a_adj, "adj_pa": self.r_adj}
