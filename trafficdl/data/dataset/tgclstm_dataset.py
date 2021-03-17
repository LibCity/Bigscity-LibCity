import numpy as np
import pandas as pd
from trafficdl.data.dataset import TrafficStatePointDataset


class TGCLSTMDataset(TrafficStatePointDataset):
    def __init__(self, config):
        self.FFR = []
        super(TGCLSTMDataset, self).__init__(config)

    def _load_rel(self):
        relfile = pd.read_csv(self.data_path + self.rel_file + '.rel')
        if self.weight_col != '':  # 根据weight_col确认权重列
            self.distance_df = relfile[~relfile[self.weight_col].isna()][[
                'origin_id', 'destination_id', self.weight_col,
                'FFR_5min', 'FFR_10min', 'FFR_15min', 'FFR_20min', 'FFR_25min']]
        else:
            raise ValueError("Don't know which column to be loaded! Please set `weight_col` parameter!")
        # 把数据转换成矩阵的形式
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        self.adj_mx[:] = np.inf
        for row in self.distance_df.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = row[2]
        self._logger.info("Loaded file " + self.dataset + '.rel')
        # 得到可达性矩阵
        for i in range(3, 8):
            ffr_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
            ffr_mx[:] = np.inf
            for row in self.distance_df.values:
                if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                    continue
                ffr_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = row[i]
            self.FFR.append(ffr_mx)
        # 计算权重
        if self.calculate_weight:
            self._calculate_adjacency_matrix()

    def get_data_feature(self):
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "FFR": self.FFR,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim}
