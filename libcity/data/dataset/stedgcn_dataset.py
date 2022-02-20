import numpy as np

from libcity.data.dataset import TrafficStateOdDataset


class STEDGCNDataset(TrafficStateOdDataset):
    def __init__(self, config):
        super(STEDGCNDataset, self).__init__(config)
        self.x_offset = config.get('lag', [1 - 24 * 7, 1 - 24, -1, 0])
        self.y_offset = [1, ]

    # TODO 需要多个邻接矩阵

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

        x.squeeze(axis=-1)
        y.squeeze(axis=-1)

        return x, y
