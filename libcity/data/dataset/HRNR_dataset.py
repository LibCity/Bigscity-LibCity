import os
import pandas as pd
import pickle
import random
from torch.utils.data import DataLoader
from logging import getLogger
from libcity.data.dataset import AbstractDataset
from libcity.data.list_dataset import ListDataset
from libcity.utils import ensure_dir


class HRNRDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get("dataset", "")
        self.cache_dataset = self.config.get("cache_dataset", True)  # TODO: save cached dataset

        self.data_path = './raw_data/' + self.dataset + '/'
        if not os.path.exists(self.data_path):
            raise ValueError("Dataset {} not exist! Please ensure the path "
                             "'./raw_data/{}/' exist!".format(self.dataset, self.dataset))

        self.cache_file_folder = "./libcity/cache/dataset_cache"
        ensure_dir(self.cache_file_folder)
        self.geo_file = self.data_path + self.config.get("geo_file", self.dataset) + ".geo"
        self.rel_file = self.data_path + self.config.get("rel_file", self.dataset) + ".rel"

        # HRNR data
        self.node_features = os.path.join(self.cache_file_folder, self.config.get("node_features"))
        self.label_train_set = os.path.join(self.cache_file_folder, self.config.get("label_train_set"))
        self.adj = os.path.join(self.cache_file_folder, self.config.get("adj"))

        self.num_nodes = 0
        self.adj_matrix = None
        self._logger = getLogger()
        self._transfer_files()
        self._calc_transfer_matrix()

    def _transfer_files(self):
        """
        加载.geo .rel，生成HRNR所需的部分文件
        .geo [geo_id, type, coordinates, lane, type, length, bridge]
        .rel [rel_id, type, origin_id, destination_id]
        """
        geo = pd.read_csv(self.data_path + self.geo_file + ".geo")
        rel = pd.read_csv(self.data_path + self.rel_file + ".rel")
        self.num_nodes = geo.shape[0]
        geo_ids = list(geo["geo_id"])
        self._logger.info("Geo_N is " + str(self.num_nodes))
        feature_dict = {}
        for geo_id in geo_ids:
            feature_dict[geo_id] = [geo_id, "Point", None, geo["lane"][geo_id],
                                    geo["type"][geo_id], geo["length"][geo_id],
                                    geo["bridge"][geo_id]]

        # node_features [[lane, type, length, id]]
        node_feature_list = []
        for geo_id in geo_ids:
            node_features = feature_dict[geo_id]
            node_feature_list.append(node_features[3:6] + [geo_id])
        pickle.dump(node_feature_list, open(self.node_features, "wb"))

        # label_pred_train_set [id]
        is_bridge_ids = []
        for geo_id in geo_ids:
            if str(feature_dict[geo_id][6]) == 1:
                is_bridge_ids.append(geo_id)

        # CompleteAllGraph [[0,1,...,0]]
        self.adj_matrix = [[0 for i in range(0, self.num_nodes)] for j in range(0, self.num_nodes)]
        for row in rel.itertuples():
            origin = getattr(row, "origin_id")
            destination = getattr(row, "destination_id")
            self.adj_matrix[origin][destination] = 1
        pickle.dump(self.adj_matrix, open(self.adj, "wb"))

    def _calc_transfer_matrix(self):
        # TODO: calculate T^SR T^RZ with 2 loss functions
        pass

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            batch_data: dict
        """
        # 正样本和负样本 1:1 提取
        label_pred_train = pickle.load(open(self.label_train_set, "rb"))
        label_pred_train_false = []
        true_sample_cnt = len(label_pred_train)
        while len(label_pred_train_false) < true_sample_cnt:
            x = random.randint(0, self.num_nodes - 1)
            if x not in label_pred_train and x not in label_pred_train_false:
                label_pred_train_false.append(x)
        all_data = []
        for i in range(0, true_sample_cnt):
            all_data.append((label_pred_train[i], 1))
            all_data.append((label_pred_train_false[i], 0))

        batch_size = self.config.get("batch_size", 100)
        train_rate = self.config.get("train_rate", 0.7)

        train_index = round(true_sample_cnt * train_rate)

        train_dataloader = DataLoader(dataset=ListDataset(all_data[0:train_index]), batch_size=batch_size)
        eval_dataloader = None
        test_dataloader = DataLoader(dataset=ListDataset(all_data[train_index:]), batch_size=batch_size)
        return train_dataloader, eval_dataloader, test_dataloader

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"adj_mx": self.adj_matrix, "num_nodes": self.num_nodes}
