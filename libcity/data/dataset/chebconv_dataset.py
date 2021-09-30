import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from logging import getLogger

from libcity.utils import StandardScaler, NormalScaler, NoneScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler, ensure_dir
from libcity.data.dataset import AbstractDataset


class ChebConvDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.cache_dataset = self.config.get('cache_dataset', True)
        self.train_rate = self.config.get('train_rate', 0.7)
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
        # 初始化
        self.adj_mx = None
        self.scaler = None
        self.feature_dim = 0
        self.num_nodes = 0
        self._logger = getLogger()
        self._load_geo()
        self._load_rel()

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

    def _load_rel(self):
        """
        加载.rel文件，格式[rel_id, type, origin_id, destination_id, properties(若干列)],
        生成N*N的矩阵，默认.rel存在的边表示为1，不存在的边表示为0

        Returns:
            np.ndarray: self.adj_mx, N*N的邻接矩阵
        """
        map_info = pd.read_csv(self.data_path + self.rel_file + '.rel')
        # 使用稀疏矩阵构建邻接矩阵
        adj_row = []
        adj_col = []
        adj_data = []
        adj_set = set()
        cnt = 0
        for i in range(map_info.shape[0]):
            if map_info['origin_id'][i] in self.geo_to_ind and map_info['destination_id'][i] in self.geo_to_ind:
                f_id = self.geo_to_ind[map_info['origin_id'][i]]
                t_id = self.geo_to_ind[map_info['destination_id'][i]]
                if (f_id, t_id) not in adj_set:
                    adj_set.add((f_id, t_id))
                    adj_row.append(f_id)
                    adj_col.append(t_id)
                    adj_data.append(1.0)
                    cnt = cnt + 1
        self.adj_mx = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=(self.num_nodes, self.num_nodes))
        save_path = self.cache_file_folder + "{}_adj_mx.npz".format(self.dataset)
        sp.save_npz(save_path, self.adj_mx)
        self._logger.info('Total link between geo = {}'.format(cnt))
        self._logger.info('Adj_mx is saved at {}'.format(save_path))

    def _split_train_val_test(self):
        # TODO: 这里进行规范化，相关内容抽象成函数，通过外部设置参数确定对哪些列进行数据预处理，即可统一
        # node_features = self.road_info[['highway', 'length', 'lanes', 'tunnel', 'bridge',
        #                                 'maxspeed', 'width', 'service', 'junction', 'key']].values
        # 'tunnel', 'bridge', 'service', 'junction', 'key'是01 1+1+1+1+1
        # 'lanes', 'highway'是类别 47+6
        # 'length', 'maxspeed', 'width'是浮点 1+1+1 共61
        node_features = self.road_info[self.road_info.columns[3:]]

        # 对部分列进行归一化
        norm_dict = {
            'length': 1,
            'maxspeed': 5,
            'width': 6
        }
        for k, v in norm_dict.items():
            d = node_features[k]
            min_ = d.min()
            max_ = d.max()
            dnew = (d - min_) / (max_ - min_)
            node_features = node_features.drop(k, 1)
            node_features.insert(v, k, dnew)

        # 对部分列进行独热编码
        onehot_list = ['lanes', 'highway']
        for col in onehot_list:
            dum_col = pd.get_dummies(node_features[col], col)
            node_features = node_features.drop(col, axis=1)
            node_features = pd.concat([node_features, dum_col], axis=1)

        node_features = node_features.values
        np.save(self.cache_file_folder + '{}_node_features.npy'.format(self.dataset), node_features)

        # mask 索引
        sindex = list(range(self.num_nodes))
        np.random.seed(1234)
        np.random.shuffle(sindex)

        test_rate = 1 - self.train_rate - self.eval_rate
        num_test = round(self.num_nodes * test_rate)
        num_train = round(self.num_nodes * self.train_rate)
        num_val = self.num_nodes - num_test - num_train

        train_mask = np.array(sorted(sindex[0: num_train]))
        valid_mask = np.array(sorted(sindex[num_train: num_train + num_val]))
        test_mask = np.array(sorted(sindex[-num_test:]))

        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                self.cache_file_name,
                node_features=node_features,
                train_mask=train_mask,
                valid_mask=valid_mask,
                test_mask=test_mask
            )
            self._logger.info('Saved at ' + self.cache_file_name)
        self._logger.info("len train feature\t" + str(len(train_mask)))
        self._logger.info("len eval feature\t" + str(len(valid_mask)))
        self._logger.info("len test feature\t" + str(len(test_mask)))
        return node_features, train_mask, valid_mask, test_mask

    def _load_cache_train_val_test(self):
        """
        加载之前缓存好的训练集、测试集、验证集
        """
        self._logger.info('Loading ' + self.cache_file_name)
        cat_data = np.load(self.cache_file_name, allow_pickle=True)
        node_features = cat_data['node_features']
        train_mask = cat_data['train_mask']
        valid_mask = cat_data['valid_mask']
        test_mask = cat_data['test_mask']
        self._logger.info("len train feature\t" + str(len(train_mask)))
        self._logger.info("len eval feature\t" + str(len(valid_mask)))
        self._logger.info("len test feature\t" + str(len(test_mask)))
        return node_features, train_mask, valid_mask, test_mask

    def _get_scalar(self, scaler_type, data):
        """
        根据全局参数`scaler_type`选择数据归一化方法

        Args:
            data: 训练数据X

        Returns:
            Scaler: 归一化对象
        """
        if scaler_type == "normal":
            scaler = NormalScaler(maxx=data.max())
            self._logger.info('NormalScaler max: ' + str(scaler.max))
        elif scaler_type == "standard":
            scaler = StandardScaler(mean=data.mean(), std=data.std())
            self._logger.info('StandardScaler mean: ' + str(scaler.mean) + ', std: ' + str(scaler.std))
        elif scaler_type == "minmax01":
            scaler = MinMax01Scaler(
                maxx=data.max(), minn=data.min())
            self._logger.info('MinMax01Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif scaler_type == "minmax11":
            scaler = MinMax11Scaler(
                maxx=data.max(), minn=data.min())
            self._logger.info('MinMax11Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif scaler_type == "log":
            scaler = LogScaler()
            self._logger.info('LogScaler')
        elif scaler_type == "none":
            scaler = NoneScaler()
            self._logger.info('NoneScaler')
        else:
            raise ValueError('Scaler type error!')
        return scaler

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            batch_data: dict
        """
        # 加载数据集
        if self.cache_dataset and os.path.exists(self.cache_file_name):
            node_features, train_mask, valid_mask, test_mask = self._load_cache_train_val_test()
        else:
            node_features, train_mask, valid_mask, test_mask = self._split_train_val_test()
        # 数据归一化
        self.feature_dim = node_features.shape[-1]
        self.scaler = self._get_scalar(self.scaler_type, node_features)
        node_features = self.scaler.transform(node_features)
        self.train_dataloader = {'node_features': node_features, 'mask': train_mask}
        self.eval_dataloader = {'node_features': node_features, 'mask': valid_mask}
        self.test_dataloader = {'node_features': node_features, 'mask': test_mask}
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"scaler": self.scaler, "adj_mx": self.adj_mx,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim}
