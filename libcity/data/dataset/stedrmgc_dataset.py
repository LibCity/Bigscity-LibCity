import os

import numpy as np
from tqdm import tqdm

from libcity.data.dataset import TrafficStateOdDataset


class STEDRMGCDataset(TrafficStateOdDataset):
    def __init__(self, config):
        super(STEDRMGCDataset, self).__init__(config)
        self.x_offset = config.get('lag', [1 - 24 * 7, 1 - 24, -1, 0])
        self.y_offset = [1, ]

    def _load_rel(self):
        super()._load_rel()
        self.num_nodes = self.adj_mx.shape[0]
        self.adj_mx = self.generate_neighborhood_matrix(self.adj_mx)
        self.adj_mx.append(np.eye(self.num_nodes * self.num_nodes))
        self.adj_mx.append(self.generate_corr_matrix())
        pass

    def generate_corr_matrix(self):
        self._logger.info("Generating corr matrix for od pairs, this might take a long time.")
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
        df_ = []
        for filename in data_files:
            df_.append(self._load_dyna(filename))

        df = np.concatenate(df_, axis=1)
        df = df.transpose((1, 2, 0, 3))
        try:
            df = df.squeeze(axis=-1)
        except:
            print("feature dim must be 1")

        num_nodes = df.shape[1]
        matrix = np.zeros((num_nodes * num_nodes, num_nodes * num_nodes))

        for o_1 in tqdm(range(num_nodes)):
            for d_1 in range(num_nodes):
                for o_2 in range(num_nodes):
                    for d_2 in range(num_nodes):
                        matrix[o_1 * num_nodes + d_1, o_2 * num_nodes + d_2] = \
                            np.corrcoef(df[o_1, d_1], df[o_2, d_2])[0][1]
        return matrix

    def generate_neighborhood_matrix(self, geo_matrix):
        num_nodes = geo_matrix.shape[0]
        origin_matrix = np.zeros((num_nodes * num_nodes, num_nodes * num_nodes))
        for o_1 in range(num_nodes):
            for d_1 in range(num_nodes):
                for o_2 in range(num_nodes):
                    for d_2 in range(num_nodes):
                        origin_matrix[o_1 * num_nodes + d_1, o_2 * num_nodes + d_2] = geo_matrix[o_1][o_2]

        destination_matrix = np.zeros((num_nodes * num_nodes, num_nodes * num_nodes))
        for o_1 in range(num_nodes):
            for d_1 in range(num_nodes):
                for o_2 in range(num_nodes):
                    for d_2 in range(num_nodes):
                        destination_matrix[o_1 * num_nodes + d_1, o_2 * num_nodes + d_2] = geo_matrix[o_2][o_2]
        return [origin_matrix, destination_matrix]

    def _generate_input_data(self, df):
        """
        根据全局参数`input_window`和`output_window`切分输入，产生模型需要的张量输入，
        即使用过去`input_window`长度的时间序列去预测未来`output_window`长度的时间序列

        Args:
            df(np.ndarray): 数据数组，shape: (len_time, ..., feature_dim)

        Returns:
            tuple: tuple contains:
                x(np.ndarray): 模型输入数据，(epoch_size, input_length, ..., feature_dim) \n
                y(np.ndarray): 模型输出数据，(epoch_size, output_length, ..., feature_dim)
        """
        num_samples = df.shape[0]

        x_offsets = np.array(self.x_offset)
        y_offsets = np.array(self.y_offset)

        x, y = [], []
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))
        for t in range(min_t, max_t):
            x_t = df[t + x_offsets, ...]
            y_t = df[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)

        return x, y

    def get_data_feature(self):
        return {"scaler": self.scaler, "adj_mx": self.adj_mx,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim, "ext_dim": self.ext_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches,
                "num_filters": len(self.adj_mx)}
