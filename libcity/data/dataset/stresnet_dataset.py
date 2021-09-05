import os
import numpy as np
from libcity.data.dataset import TrafficStateCPTDataset, TrafficStateGridDataset
from libcity.utils.dataset import timestamp2array, timestamp2vec_origin


class STResNetDataset(TrafficStateGridDataset, TrafficStateCPTDataset):
    """
    STResNet外部数据源代码只用了ext_y, 没有用到ext_x!
    """

    def __init__(self, config):
        super().__init__(config)
        self.external_time = self.config.get('external_time', True)
        self.parameters_str = \
            self.parameters_str + '_' + str(self.len_closeness) \
            + '_' + str(self.len_period) + '_' + str(self.len_trend) \
            + '_' + str(self.interval_period) + '_' + str(self.interval_trend)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'grid_based_{}.npz'.format(self.parameters_str))
        self.pad_forward_period = 0
        self.pad_back_period = 0
        self.pad_forward_trend = 0
        self.pad_back_trend = 0

    def _get_external_array(self, timestamp_list, ext_data=None, previous_ext=False, ext_time=True):
        """
        根据时间戳数组，获取对应时间的外部特征

        Args:
            timestamp_list: 时间戳序列
            ext_data: 外部数据
            previous_ext: 是否是用过去时间段的外部数据，因为对于预测的时间段Y，
                            一般没有真实的外部数据，所以用前一个时刻的数据，**多步预测则用提前多步的数据**

        Returns:
            np.ndarray: External data shape is (len(timestamp_list), ext_dim)
        """
        data = []
        if ext_time:
            vecs_timestamp = timestamp2array(
                timestamp_list, 24 * 60 * 60 // self.time_intervals)  # len(timestamp_list) * dim
        else:
            vecs_timestamp = timestamp2vec_origin(timestamp_list)  # len(timestamp_list) * dim
        data.append(vecs_timestamp)
        # 外部数据集
        if ext_data is not None:
            indexs = []
            for ts in timestamp_list:
                if previous_ext:
                    # TODO: 多步预测这里需要改
                    ts_index = self.idx_of_ext_timesolts[ts - self.offset_frame]
                else:
                    ts_index = self.idx_of_ext_timesolts[ts]
                indexs.append(ts_index)
            select_data = ext_data[indexs]  # len(timestamp_list) * ext_dim 选出所需要的时间步的数据
            data.append(select_data)
        if len(data) > 0:
            data = np.hstack(data)
        else:
            data = np.zeros((len(timestamp_list), 0))
        return data  # (len(timestamp_list), ext_dim)

    def _load_ext_data(self, ts_x, ts_y):
        """
        加载对应时间的外部数据(.ext)

        Args:
            ts_x: 输入数据X对应的时间戳，shape: (num_samples, T_c+T_p+T_t)
            ts_y: 输出数据Y对应的时间戳，shape:(num_samples, )

        Returns:
            tuple: tuple contains:
                ext_x(np.ndarray): 对应时间的外部数据, shape: (num_samples, T_c+T_p+T_t, ext_dim),
                ext_y(np.ndarray): 对应时间的外部数据, shape: (num_samples, ext_dim)
        """
        # 加载外部数据
        if self.load_external and os.path.exists(self.data_path + self.ext_file + '.ext'):  # 外部数据集
            ext_data = self._load_ext()
            ext_data = 1. * (ext_data - ext_data.min()) / (ext_data.max() - ext_data.min())
        else:
            ext_data = None
        ext_x = []
        for ts in ts_x:
            ext_x.append(self._get_external_array(ts, ext_data, ext_time=self.external_time))
        ext_x = np.asarray(ext_x)
        # ext_x: (num_samples_plus, T_c+T_p+T_t, ext_dim)
        ext_y = self._get_external_array(ts_y, ext_data, previous_ext=True, ext_time=self.external_time)
        # ext_y: (num_samples_plus, ext_dim)
        return ext_x, ext_y

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是网格的个数，
        len_row是网格的行数，len_column是网格的列数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        lp = self.len_period * (self.pad_forward_period + self.pad_back_period + 1)
        lt = self.len_trend * (self.pad_forward_trend + self.pad_back_trend + 1)
        return {"scaler": self.scaler, "adj_mx": self.adj_mx,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim, "ext_dim": self.ext_dim,
                "output_dim": self.output_dim, "len_row": self.len_row, "len_column": self.len_column,
                "len_closeness": self.len_closeness, "len_period": lp, "len_trend": lt, "num_batches": self.num_batches}
