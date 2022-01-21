import os

import numpy as np
import torch
from scipy.spatial import distance
from libcity.data.dataset import TrafficStateOdDataset


class MPGCN_Dataset(TrafficStateOdDataset):
    def __init__(self, config):
        super().__init__(config)
        self.O_D_G,self.D_D_G = self.construct_dyn_G()


    def construct_dyn_G(self , perceived_period:int=7):
        for filename in self.data_files:
            OD_data = super(MPGCN_Dataset,self)._load_od_4d(filename)
        train_len = int(OD_data.shape[0])
        num_periods_in_history = train_len // perceived_period  # dump the remainder，做取整处理
        OD_history = OD_data[:num_periods_in_history * perceived_period, :, :, :]
        O_dyn_G, D_dyn_G = [], []
        for t in range(perceived_period):
            OD_t_avg = np.mean(OD_history[t::perceived_period,:,:,:], axis=0).squeeze(axis=-1)#对每个星期t的数据求取均值
            O, D = OD_t_avg.shape

            O_G_t = np.zeros((O, O))    # initialize O graph at t
            for i in range(O):
                for j in range(O):
                    if(np.any(OD_t_avg[i,:]) and np.any(OD_t_avg[j,:])):
                        O_G_t[i, j] = distance.cosine(OD_t_avg[i,:], OD_t_avg[j,:])     # eq (6)，算出i和j的相似度
            D_G_t = np.zeros((D, D))    # initialize D graph at t
            for i in range(D):
                for j in range(D):
                    if(np.any(OD_t_avg[:,i]) and np.any(OD_t_avg[:,j])):
                        D_G_t[i, j] = distance.cosine(OD_t_avg[:,i], OD_t_avg[:,j])     # eq (7)
            O_dyn_G.append(O_G_t), D_dyn_G.append(D_G_t) #7*N*N

        return np.stack(O_dyn_G, axis=0), np.stack(D_dyn_G, axis=0)

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是网格的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"scaler": self.scaler, "adj_mx": self.adj_mx,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim, "ext_dim": self.ext_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches,"O_D_G":self.O_D_G,"D_D_G":self.D_D_G}