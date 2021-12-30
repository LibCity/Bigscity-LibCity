import os
import random
from logging import getLogger

import numpy as np
import pandas as pd

from libcity.data.dataset import AbstractDataset
from libcity.data.utils import generate_dataloader
from libcity.utils import ensure_dir


class Alias:
    def __init__(self, prob):
        """
        使用 alias 方法，生成指定定分布
        Args:
            prob: list 目标概率分布
        """
        length = len(prob)
        self.length = length
        accept, alias = [0] * length, [0] * length
        insufficient, exceed = [], []
        prob_ = np.array(prob) * length
        for i, prob in enumerate(prob_):
            if prob < 1.0:
                insufficient.append(i)
            else:
                exceed.append(i)

        while insufficient and exceed:
            small_idx, large_idx = insufficient.pop(), exceed.pop()
            accept[small_idx] = prob_[small_idx]
            alias[small_idx] = large_idx
            prob_[large_idx] = prob_[large_idx] - (1 - prob_[small_idx])
            if prob_[large_idx] < 1.0:
                insufficient.append(large_idx)
            else:
                exceed.append(large_idx)

        while exceed:
            large_idx = exceed.pop()
            accept[large_idx] = 1
        while insufficient:
            small_idx = insufficient.pop()
            accept[small_idx] = 1

        self.accept = accept
        self.alias = alias

    def sample(self):
        idx = random.randint(0, self.length - 1)
        if random.random() >= self.accept[idx]:
            return self.alias[idx]
        else:
            return idx


class LINEDataset(AbstractDataset):

    def __init__(self, config):
        # 数据集参数
        self.dataset = config.get('dataset')
        self.negative_ratio = config.get('negative_ratio', 5)  # 负采样数，对于大数据集，适合 2-5
        self.batch_size = config.get('batch_size', 32)
        self.times = config.get('times')
        self.scaler = None
        # 数据集比例
        self.train_rate = config.get('train_rate', 0.7)
        self.eval_rate = config.get('eval_rate', 0.1)
        self.scaler_type = config.get('scaler', 'none')
        # 缓存
        self.cache_dataset = config.get('cache_dataset', True)
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
        # 读取原子文件
        self.geo_file = config.get('geo_file', self.dataset)
        self.rel_file = config.get('rel_file', self.dataset)

        # 框架相关
        self._logger = getLogger()
        self.feature_name = {'I': 'int', 'J': 'int', 'Neg': 'int'}
        self.num_workers = config.get('num_workers', 0)

        self._load_geo()
        self._load_rel()

        # 采样条数
        self.num_samples = self.num_edges * (1 + self.negative_ratio) * self.times

    def _load_geo(self):
        """
        加载.geo文件，格式[geo_id, type, coordinates, properties(若干列)]
        """
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.num_nodes = len(self.geo_ids)
        self._geo_to_ind = {}
        for index, idx in enumerate(self.geo_ids):
            self._geo_to_ind[idx] = index
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_nodes=' + str(self.num_nodes))

    def _load_rel(self):
        """
        加载.rel文件，格式[rel_id, type, origin_id, destination_id, properties(若干列)],
        生成N*N的矩阵，默认.rel存在的边表示为1，不存在的边表示为0

        Returns:
            np.ndarray: self.adj_mx, N*N的邻接矩阵
        """
        map_info = pd.read_csv(self.data_path + self.rel_file + '.rel')
        if 'weight' in map_info.columns:
            self.edges = [(self._geo_to_ind[e[0]], self._geo_to_ind[e[1]], e[2]) for e in
                          map_info[['origin_id', 'destination_id', 'weight']].values]
        else:
            self.edges = [(self._geo_to_ind[e[0]], self._geo_to_ind[e[1]], 1) for e in
                          map_info[['origin_id', 'destination_id']].values]
        self.num_edges = len(self.edges)
        self._logger.info("Loaded file " + self.rel_file + '.rel' + ', num_edges=' + str(self.num_edges))

    def _gen_sampling_table(self, POW=0.75):
        node_degree = np.zeros(self.num_nodes)
        for edge in self.edges:
            node_degree[edge[0]] += edge[2]
        # 节点负采样所需 Alias 表
        norm_prob = node_degree ** POW
        norm_prob = node_degree / norm_prob.sum()
        self.node_alias = Alias(norm_prob)
        # 边采样所需 Alias 表

        norm_prob = 0
        for edge in self.edges:
            norm_prob += edge[2]
        norm_prob = [p[2] / norm_prob for p in self.edges]
        self.edge_alias = Alias(norm_prob)

    def _generate_data(self):
        """
        LINE 采用的是按类似于 Skip-Gram 的训练方式，类似于 Word2Vec(Skip-Gram)，将单词对类比成图中的一条边，
        LINE 同时采用了两个优化，一个是对边按照正比于边权重的概率进行采样，另一个是类似于 Word2Vec 当中的负采样方法，
        在采样一条边时，同时产生该边起始点到目标点（按正比于度^0.75的概率采样获得）的多个"负采样"边。
        最后，为了通过 Python 的均匀分布随机数产生符合目标分布的采样，使用 O(1) 的 alias 采样方法
        """
        # 加载数据集
        self._load_geo()
        self._load_rel()

        # 生成采样数据
        self._gen_sampling_table()
        I = []  # 起始点
        J = []  # 终止点
        Neg = []  # 是否为负采样

        pad_sample = self.num_samples % (1 + self.negative_ratio)

        for _ in range(self.num_samples // (1 + self.negative_ratio)):
            # 正样本
            edge = self.edges[self.edge_alias.sample()]
            I.append(edge[0])
            J.append(edge[1])
            Neg.append(1)
            # 负样本
            for _ in range(self.negative_ratio):
                I.append(edge[0])
                J.append(self.node_alias.sample())
                Neg.append(-1)

        # 填满 epoch
        if pad_sample > 0:
            edge = self.edges[self.edge_alias.sample()]
            I.append(edge[0])
            J.append(edge[1])
            Neg.append(1)
            pad_sample -= 1
            if pad_sample > 0:
                for _ in range(pad_sample):
                    I.append(edge[0])
                    J.append(self.node_alias.sample())
                    Neg.append(-1)

        test_rate = 1 - self.train_rate - self.eval_rate
        num_test = round(self.num_samples * test_rate)
        num_train = round(self.num_samples * self.train_rate)
        num_eval = self.num_samples - num_test - num_train

        # train
        I_train, J_train, Neg_train = I[:num_train], J[:num_train], Neg[:num_train]
        # eval
        I_eval, J_eval, Neg_eval = I[num_train:num_train + num_eval], J[num_train:num_train + num_eval], \
                                   Neg[num_train:num_train + num_eval]
        # test
        I_test, J_test, Neg_test = I[-num_test:], J[-num_test:], Neg[-num_test:]

        self._logger.info(
            "train\tI: {}, J: {}, Neg: {}".format(str(len(I_train)), str(len(J_train)), str(len(Neg_train))))
        self._logger.info(
            "eval\tI: {}, J: {}, Neg: {}".format(str(len(I_eval)), str(len(J_eval)), str(len(Neg_eval))))
        self._logger.info(
            "test\tI: {}, J: {}, Neg: {}".format(str(len(I_test)), str(len(J_test)), str(len(Neg_test))))

        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                self.cache_file_name,
                I_train=I_train,
                J_train=J_train,
                Neg_train=Neg_train,
                I_test=I_test,
                J_test=J_test,
                Neg_test=Neg_test,
                I_eval=I_eval,
                J_eval=J_eval,
                Neg_eval=Neg_eval
            )
            self._logger.info('Saved at ' + self.cache_file_name)

        return I_train, J_train, Neg_train, I_eval, J_eval, Neg_eval, I_test, J_test, Neg_test

    def _load_cache(self):
        """
        加载之前缓存好的训练集、测试集、验证集
        """

        self._logger.info('Loading ' + self.cache_file_name)
        cat_data = np.load(self.cache_file_name)
        I_train = cat_data['I_train']
        J_train = cat_data['J_train']
        Neg_train = cat_data['Neg_train']
        I_test = cat_data['I_test']
        J_test = cat_data['J_test']
        Neg_test = cat_data['Neg_test']
        I_eval = cat_data['I_eval']
        J_eval = cat_data['J_eval']
        Neg_eval = cat_data['Neg_eval']

        self._logger.info(
            "train\tI: {}, J: {}, Neg: {}".format(str(len(I_train)), str(len(J_train)), str(len(Neg_train))))
        self._logger.info(
            "eval\tI: {}, J: {}, Neg: {}".format(str(len(I_eval)), str(len(J_eval)), str(len(Neg_eval))))
        self._logger.info(
            "test\tI: {}, J: {}, Neg: {}".format(str(len(I_test)), str(len(J_test)), str(len(Neg_test))))

        return I_train, J_train, Neg_train, I_eval, J_eval, Neg_eval, I_test, J_test, Neg_test

    def get_data(self):
        """
                返回数据的DataLoader，包括训练数据、测试数据、验证数据

                Returns:
                    batch_data: dict
                """
        # 加载数据集
        if self.cache_dataset and os.path.exists(self.cache_file_name):
            I_train, J_train, Neg_train, I_eval, J_eval, Neg_eval, I_test, J_test, Neg_test = self._load_cache()
        else:
            I_train, J_train, Neg_train, I_eval, J_eval, Neg_eval, I_test, J_test, Neg_test = self._generate_data()

        train_data = list(zip(I_train, J_train, Neg_train))
        eval_data = list(zip(I_eval, J_eval, Neg_eval))
        test_data = list(zip(I_test, J_test, Neg_test))

        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name, self.batch_size, self.num_workers)

        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"scaler": self.scaler, "num_edges": self.num_edges,
                "num_nodes": self.num_nodes}
