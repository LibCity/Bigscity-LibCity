import pandas as pd
import numpy as np
import networkx as nx
from logging import getLogger
import scipy.sparse as sp
import os
from pip._internal.utils.misc import ensure_dir

class ReadGraph():
    def __init__(self, config):
        self.config = config
        #加载rel文件
        self.dataset = self.config.get('dataset', 'BJ_roadmap')
        self.data_path = './raw_data/' + self.dataset + '/'
        self.rel_file = self.config.get('rel_file', self.dataset)
    #根据rel文件生成有向图
    def _load_graph(self):
        map_info = pd.read_csv(self.data_path + self.rel_file + '.rel')
        start = np.array(map_info['origin_id'])
        end = np.array(map_info['destination_id'])
        nG = nx.DiGraph()
        for i in range(len(start)):
            nG.add_edge(start[i],end[i])
        return nG

class Node2vecDataset():
    def __init__(self, config, sentences):
        self.config = config
        self.dataset = self.config.get('dataset', 'BJ_roadmap')
        self.cache_dataset = self.config.get('cache_dataset', True)
        self.train_rate = self.config.get('train_rate ', 0.7)
        self.eval_rate = self.config.get('eval_rate', 0.1)
        self.scaler_type = self.config.get('scaler', 'none')
        # 路径等参数
        self.parameters_str = \
            str(self.dataset) + '_' + str(self.train_rate) + '_' \
            + str(self.eval_rate) + '_' + str(self.scaler_type)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'road_rep_{}.npz'.format(self.parameters_str))
        self.cache_file_folder = './libcity/cache/dataset_cache/'
        ensure_dir(self.cache_file_folder)
        self.data_path = './raw_data/' + self.dataset + '/'
        if not os.path.exists(self.data_path):
            raise ValueError("Dataset {} not exist! Please ensure the path "
                             "'./raw_data/{}/' exist!".format(self.dataset, self.dataset))
        # 加载数据集的config.json文件
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        # 加载node2游走结果
        self.sentences = sentences
        # 初始化
        self.num_nodes = 0
        self._logger = getLogger()
        self._load_geo()

    def _load_geo(self):
        """
        加载.geo文件，格式[geo_id, type, coordinates, properties(若干列)]
        """
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.num_nodes = len(self.geo_ids)
        self.geo_to_ind = {}
        for index, idx in enumerate(self.geo_ids):
            self.geo_to_ind[idx] = index
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_nodes=' + str(len(self.geo_ids)))
        self.road_info = geofile

    def _split_train_val_test(self):
        # mask 索引
        num_walk = len(self.sentences)
        sindex = list(range(num_walk))
        np.random.seed(1234)
        np.random.shuffle(sindex)

        test_rate = 1 - self.train_rate - self.eval_rate
        num_test = round(num_walk * test_rate)
        num_train = round(num_walk * self.train_rate)
        num_val = num_walk - num_test - num_train

        train_num = np.array(sorted(sindex[0: num_train]))
        valid_num = np.array(sorted(sindex[num_train: num_train + num_val]))
        test_num = np.array(sorted(sindex[-num_test:]))
        train_mask = [self.sentences[i] for i in train_num]
        valid_mask = [self.sentences[i] for i in valid_num]
        test_mask = [self.sentences[i] for i in test_num]
        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                self.cache_file_name,
                train_mask=train_mask,
                valid_mask=valid_mask,
                test_mask=test_mask
            )
            self._logger.info('Saved at ' + self.cache_file_name)
        self._logger.info("len train feature\t" + str(len(train_mask)))
        self._logger.info("len eval feature\t" + str(len(valid_mask)))
        self._logger.info("len test feature\t" + str(len(test_mask)))
        return train_mask, valid_mask, test_mask

    def _load_cache_train_val_test(self):
        """
        加载之前缓存好的训练集、测试集、验证集
        """
        self._logger.info('Loading ' + self.cache_file_name)
        cat_data = np.load(self.cache_file_name, allow_pickle=True)
        train_mask = cat_data['train_mask']
        valid_mask = cat_data['valid_mask']
        test_mask = cat_data['test_mask']
        self._logger.info("len train feature\t" + str(len(train_mask)))
        self._logger.info("len eval feature\t" + str(len(valid_mask)))
        self._logger.info("len test feature\t" + str(len(test_mask)))
        return train_mask, valid_mask, test_mask

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            batch_data: dict
        """
        # 加载数据集
        if self.cache_dataset and os.path.exists(self.cache_file_name):
            train_mask, valid_mask, test_mask = self._load_cache_train_val_test()
        else:
            train_mask, valid_mask, test_mask = self._split_train_val_test()
        self.train_dataloader = {'mask': train_mask}
        self.eval_dataloader = {'mask': valid_mask}
        self.test_dataloader = {'mask': test_mask}
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader