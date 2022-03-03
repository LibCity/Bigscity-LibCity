import os
import math
import numpy as np

from libcity.data.dataset import TrafficStatePointDataset
# from libcity.data.dataset import TrafficStateGridDataset
from libcity.utils import ensure_dir


"""
主要功能是定义了一种根据原始交通状态数据计算邻接矩阵的方法，并且保存了计算好的邻接矩阵
当然，采用框架预定义的根据距离定义邻接矩阵的方法也是可以的
因此STG2Seq模型也支持基础的TrafficStatePointDataset、TrafficStateGridDataset
STG2SeqDataset既可以继承TrafficStatePointDataset，也可以继承TrafficStateGridDataset以处理网格数据
修改成TrafficStateGridDataset时，只需要修改：
1.TrafficStatePointDataset-->TrafficStateGridDataset
2.self.use_row_column = False, 可以加到self.parameters_str中
3.需要修改_generate_graph_with_data函数！
"""


class STG2SeqDataset(TrafficStatePointDataset):

    def __init__(self, config):
        super().__init__(config)
        self.use_row_column = False
        self.parameters_str += '_save_adj'
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'point_based_{}.npz'.format(self.parameters_str))

    def _load_rel(self):
        """
        根据网格结构构建邻接矩阵，一个格子跟他周围的8个格子邻接

        Returns:
            np.ndarray: self.adj_mx, N*N的邻接矩阵
        """
        pass
        # self._logger.info("Generate rel file from data, shape=" + str(self.adj_mx.shape))

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
            df_list.append(df.copy())
            if self.load_external:
                df = self._add_external_information(df, ext_data)
            x, y = self._generate_input_data(df)
            # x: (num_samples, input_length, ..., input_dim)
            # y: (num_samples, output_length, ..., output_dim)
            x_list.append(x)
            y_list.append(y)
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        df = np.concatenate(df_list)
        self._logger.info("Dataset created")
        self._logger.info("x shape: " + str(x.shape) + ", y shape: " + str(y.shape))
        return x, y, df

    def _split_train_val_test(self, x, y, df=None):
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

        self.adj_mx = self._generate_graph_with_data(data=df, length=num_test)
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
                adj_mx=self.adj_mx
            )
            self._logger.info('Saved at ' + self.cache_file_name)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _generate_train_val_test(self):
        """
        加载数据集，并划分训练集、测试集、验证集，并缓存数据集

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        x, y, df = self._generate_data()
        return self._split_train_val_test(x, y, df)

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
        self.adj_mx = cat_data['adj_mx']
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))
        sparsity = self.adj_mx.sum() / (self.adj_mx.shape[0] * self.adj_mx.shape[1])
        self._logger.info("Generate rel file from data, shape=" + str(self.adj_mx.shape))
        self._logger.info("Sparsity of the adjacent matrix is: " + str(sparsity))
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _generate_graph_with_data(self, data, length, threshold=0.05):
        # data shape is [sample_nums, node_nums, dims] or [sample_nums, node_nums]
        sample_nums, node_num, dim = data.shape[0], data.shape[1], data.shape[2]
        # print(data.shape)
        adj_mx = np.zeros((node_num, node_num))
        demand_zero = np.zeros((length, dim))
        for i in range(node_num):
            node_i = data[-length:, i, :]
            adj_mx[i, i] = 1
            if np.array_equal(node_i, demand_zero):
                continue
            else:
                for j in range(i + 1, node_num):
                    node_j = data[-length:, j, :]
                    distance = math.exp(-(np.abs((node_j - node_i)).sum() / length*dim))
                    if distance > threshold:
                        adj_mx[i, j] = 1
                        adj_mx[j, i] = 1
        sparsity = adj_mx.sum() / (node_num * node_num)
        self._logger.info("Generate rel file from data, shape=" + str(adj_mx.shape))
        self._logger.info("Sparsity of the adjacent matrix is: " + str(sparsity))
        return adj_mx
