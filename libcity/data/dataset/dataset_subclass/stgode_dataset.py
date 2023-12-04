import os
import pickle

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from tqdm import tqdm

from libcity.data.dataset import TrafficStatePointDataset


class STGODEDataset(TrafficStatePointDataset):
    def __init__(self, config):
        self.sigma2 = config.get('sigma2', 10)
        self.thres2 = config.get('thres2', 0.5)
        super().__init__(config)
        self.load_from_local = self.config.get('load_from_local', True)
        self.points_per_day = 24 * 60 * 60 // self.time_intervals
        self.sigma1 = self.config.get('sigma1', 0.1)
        self.thres1 = self.config.get('thres1', 0.6)
        cache_path = './libcity/cache/dataset_cache/dtw_distance_index_' + self.dataset + '.npz'
        if self.load_from_local and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.dist_matrix = pickle.load(f)
        else:
            self.dist_matrix = self.get_dtw_matrix()
            with open(cache_path, 'wb') as f:
                pickle.dump(self.dist_matrix, f)

    def get_dtw_matrix(self):
        i = 0
        for filename in self.data_files:
            if i == 0:
                df = self._load_dyna(filename)  # (len_time, node_num, feature_dim)
            else:
                df = np.concatenate((df, self._load_dyna(filename)), axis=0)
            i += 1
        # print(df.shape)
        df = df[:, :, 0]
        # print(self.points_per_day)
        data_mean = np.mean([df[self.points_per_day * i: self.points_per_day * (i + 1)] for i in
                             range(df.shape[0] // self.points_per_day)], axis=0)
        # print(data_mean.shape)
        data_mean = data_mean.squeeze().T
        # print(data_mean.shape)
        dtw_distance = np.zeros((self.num_nodes, self.num_nodes))
        for i in tqdm(range(self.num_nodes)):
            for j in range(i, self.num_nodes):
                dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
        for i in range(self.num_nodes):
            for j in range(i):
                dtw_distance[i][j] = dtw_distance[j][i]
        mean = np.mean(dtw_distance)
        std = np.std(dtw_distance)
        dtw_distance = (dtw_distance - mean) / std
        dtw_distance = np.exp(-dtw_distance ** 2 / self.sigma1 ** 2)
        dtw_matrix = np.zeros_like(dtw_distance)
        dtw_matrix[dtw_distance > self.thres1] = 1
        return dtw_matrix

    def _load_rel(self):
        map_info = pd.read_csv(self.data_path + self.rel_file + '.rel')
        dist_matrix = np.zeros((self.num_nodes, self.num_nodes)) + np.float64('inf')
        for i in range(map_info.shape[0]):
            start = map_info['origin_id'][i]
            end = map_info['destination_id'][i]
            dist_matrix[start][end] = map_info['weight'][i]
        std = np.std(dist_matrix[dist_matrix != np.float64('inf')])
        mean = np.mean(dist_matrix[dist_matrix != np.float64('inf')])
        dist_matrix = (dist_matrix - mean) / std
        self.adj_mx = np.exp(- dist_matrix ** 2 / self.sigma2 ** 2)
        self.adj_mx[self.adj_mx < self.thres2] = 0

    def get_data_feature(self):
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "ext_dim": self.ext_dim,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches,
                "A_se_hat": self.dist_matrix}
