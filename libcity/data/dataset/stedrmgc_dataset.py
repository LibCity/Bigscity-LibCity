import numpy as np

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

    def generate_neighborhood_matrix(self, geo_matrix):
        num_nodes = geo_matrix.shape[0]
        origin_matrix = np.zeros((num_nodes * num_nodes, num_nodes * num_nodes))
        for o_i in range(num_nodes):
            for o_j in range(num_nodes):
                for d_i in range(num_nodes):
                    for d_j in range(num_nodes):
                        origin_matrix[o_i * num_nodes + o_j, d_i * num_nodes + d_j] = geo_matrix[o_i][d_i]

        destination_matrix = np.zeros((num_nodes * num_nodes, num_nodes * num_nodes))
        for o_i in range(num_nodes):
            for o_j in range(num_nodes):
                for d_i in range(num_nodes):
                    for d_j in range(num_nodes):
                        destination_matrix[o_i * num_nodes + o_j, d_i * num_nodes + d_j] = geo_matrix[d_i][d_i]
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
