import numpy as np
import pandas as pd
from trafficdl.data.dataset import TrafficSpeedDataset


class TGCLSTMDataset(TrafficSpeedDataset):
    def __init__(self, config):
        self.FFR = []
        super(TGCLSTMDataset, self).__init__(config)

    def _load_rel(self):
        relfile = pd.read_csv(self.data_path + '.rel')
        self.distance_df = relfile[~relfile[self.config['weight_col']].isna()][
            ['origin_id', 'destination_id', self.config['weight_col'],
             'FFR_5min', 'FFR_10min', 'FFR_15min', 'FFR_20min', 'FFR_25min']]
        # 把数据转换成矩阵的形式
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        self.adj_mx[:] = np.inf
        for row in self.distance_df.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = row[2]
        # 得到可达性矩阵
        for i in range(3, 8):
            FFR_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
            FFR_mx[:] = np.inf
            for row in self.distance_df.values:
                if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                    continue
                FFR_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = row[i]
            self.FFR.append(FFR_mx)

    def get_data_feature(self):
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "FFR": self.FFR,
                "data_loader": self.eval_dataloader,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim}
