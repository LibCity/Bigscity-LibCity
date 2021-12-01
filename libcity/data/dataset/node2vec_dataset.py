import pandas as pd

import numpy as np

import networkx as nx

from libcity.data.dataset.abstract_dataset import AbstractDataset


class Node2VecDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        #加载rel文件
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        self.rel_file = self.dataset + '.rel'

        self.graph = None

    def get_data(self):
        """
        根据rel文件生成有向图

        """
        map_info = pd.read_csv(self.data_path + self.rel_file)
        start = np.array(map_info['origin_id'])
        end = np.array(map_info['destination_id'])
        nG = nx.DiGraph()
        for i in range(len(start)):
            nG.add_edge(start[i],end[i])
        self.graph = nG
        return [], [], []

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"G": self.graph}



