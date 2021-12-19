import os
from libcity.data.dataset import TrafficStateDataset


class RoadNetWorkDataset(TrafficStateDataset):
    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        assert os.path.exists(self.data_path + self.geo_file + '.geo')
        assert os.path.exists(self.data_path + self.rel_file + '.rel')
        super().__init__(config)

    def get_data(self):
        """
        返回数据的DataLoader，此类只负责返回路网结构adj_mx，而adj_mx在data_feature中，这里什么都不返回
        """
        return None, None, None

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"adj_mx": self.adj_mx, "num_nodes": self.num_nodes,
                "geo_to_ind": self.geo_to_ind, "ind_to_geo": self.ind_to_geo}
