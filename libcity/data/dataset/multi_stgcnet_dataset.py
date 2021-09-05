import os
from libcity.data.dataset import TrafficStateCPTDataset
from libcity.data.dataset import TrafficStatePointDataset


class MultiSTGCnetDataset(TrafficStatePointDataset, TrafficStateCPTDataset):

    def __init__(self, config):
        super().__init__(config)
        self.parameters_str = \
            self.parameters_str + '_' + str(self.len_closeness) \
            + '_' + str(self.len_period) + '_' + str(self.len_trend) \
            + '_' + str(self.pad_forward_period) + '_' + str(self.pad_back_period) \
            + '_' + str(self.pad_forward_trend) + '_' + str(self.pad_back_trend) \
            + '_' + str(self.interval_period) + '_' + str(self.interval_trend)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'point_based_{}.npz'.format(self.parameters_str))

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度,
        len_closeness/len_period/len_trend分别是三段数据的长度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        lp = self.len_period * (self.pad_forward_period + self.pad_back_period + 1)
        lt = self.len_trend * (self.pad_forward_trend + self.pad_back_trend + 1)
        return {"scaler": self.scaler, "adj_mx": self.adj_mx,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "ext_dim": self.ext_dim,
                "len_closeness": self.len_closeness, "len_period": lp, "len_trend": lt,
                "num_batches": self.num_batches}
