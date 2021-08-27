import os

import numpy as np
import pandas as pd

from libtraffic.data.dataset import TrafficStateDataset


class TrafficStateOdDataset(TrafficStateDataset):
    def __init__(self, config):
        super().__init__(config)
        self.cache_file_name = os.path.join('./libtraffic/cache/dataset_cache/',
                                            'od_based_{}.npz'.format(self.parameters_str))
        self._load_rel()  # don't care whether there is a .rel file

    def _load_dyna(self, filename):
        self._logger.info("Loading file " + filename + '.od')
        odfile = pd.read_csv(self.data_path + filename + '.od')
        if self.data_col != '':  # 根据指定的列加载数据集
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:  # str
                data_col = [self.data_col].copy()
            data_col.insert(0, 'time')
            data_col.insert(1, 'origin_id')
            data_col.insert(2, 'destination_id')
            odfile = odfile[data_col]
        else:  # 不指定则加载所有列
            odfile = odfile[odfile.columns[2:]]  # 从time列开始所有列
        # 求时间序列
        self.timesolts = list(odfile['time'][:int(odfile.shape[0] / self.num_nodes / self.num_nodes)])
        self.idx_of_timesolts = dict()
        if not odfile['time'].isna().any():  # 时间没有空值
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx

        feature_dim = len(odfile.columns) - 3
        df = odfile[odfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = np.zeros((self.num_nodes, self.num_nodes, len_time, feature_dim))
        for i in range(self.num_nodes):
            origin_index = i * len_time * self.num_nodes  # 每个起点占据len_t*n行
            for j in range(self.num_nodes):
                destination_index = j * len_time  # 每个终点占据len_t行
                index = origin_index + destination_index
                data[i][j] = df[index:index + len_time].values
        data = data.transpose((2, 0, 1, 3))  # (len_time, num_nodes, num_nodes, feature_dim)
        self._logger.info("Loaded file " + filename + '.od' + ', shape=' + str(data.shape))
        return data

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是网格的个数，
        len_row是网格的行数，len_column是网格的列数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"scaler": self.scaler, "adj_mx": self.adj_mx,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim, "ext_dim": self.ext_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches}
