import os
import pandas as pd
import numpy as np
import datetime
from logging import getLogger

from trafficdl.data.dataset import AbstractDataset
from trafficdl.data.utils import generate_dataloader
from trafficdl.utils import StandardScaler, NormalScaler, NoneScaler, MinMax01Scaler, MinMax11Scaler, ensure_dir


class TrafficStateDataset(AbstractDataset):
    """
    交通状态预测数据集的基类。
    默认使用`input_window`的数据预测`output_window`对应的数据，即一个X，一个y。
    一般将外部数据融合到X中共同进行预测，因此数据为[X, y]。
    默认使用`train_rate`和`eval_rate`在样本数量(num_samples)维度上直接切分训练集、测试集、验证集。
    """

    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.batch_size = self.config.get('batch_size', 64)
        self.cache_dataset = self.config.get('cache_dataset', True)
        self.num_workers = self.config.get('num_workers', 0)
        self.pad_with_last_sample = self.config.get('pad_with_last_sample', True)
        self.train_rate = self.config.get('train_rate', 0.7)
        self.eval_rate = self.config.get('eval_rate', 0.1)
        self.scaler_type = self.config.get('scaler', 'none')
        self.load_external = self.config.get('load_external', False)
        self.normal_external = self.config.get('normal_external', False)
        self.add_time_in_day = self.config.get('add_time_in_day', False)
        self.add_day_in_week = self.config.get('add_day_in_week', False)
        self.input_window = self.config.get('input_window', 12)
        self.output_window = self.config.get('output_window', 12)
        self.parameters_str = \
            str(self.dataset) + '_' + str(self.input_window) + '_' + str(self.output_window) + '_' \
            + str(self.train_rate) + '_' + str(self.eval_rate) + '_' + str(self.scaler_type) + '_' \
            + str(self.batch_size) + '_' + str(self.add_time_in_day) + '_' \
            + str(self.add_day_in_week) + '_' + str(self.pad_with_last_sample)
        self.cache_file_name = os.path.join('./trafficdl/cache/dataset_cache/',
                                            'traffic_state_{}.npz'.format(self.parameters_str))
        self.cache_file_folder = './trafficdl/cache/dataset_cache/'
        ensure_dir(self.cache_file_folder)
        self.data_path = './raw_data/' + self.dataset + '/'
        if not os.path.exists(self.data_path):
            raise ValueError("Dataset {} not exist! Please ensure the path "
                             "'./raw_data/{}/' exist!".format(self.dataset, self.dataset))
        # 加载数据集的config.json文件
        self.weight_col = self.config.get('info', {}).get('weight_col', '')
        self.data_col = self.config.get('info', {}).get('data_col', '')
        self.ext_col = self.config.get('info', {}).get('ext_col', '')
        self.geo_file = self.config.get('info', {}).get('geo_file', self.dataset)
        self.rel_file = self.config.get('info', {}).get('rel_file', self.dataset)
        self.data_files = self.config.get('info', {}).get('data_files', self.dataset)
        self.ext_file = self.config.get('info', {}).get('ext_file', self.dataset)
        self.output_dim = self.config.get('info', {}).get('output_dim', 1)
        self.init_weight_inf_or_zero = self.config.get('info', {}).get('init_weight_inf_or_zero', 'inf')
        self.set_weight_link_or_dist = self.config.get('info', {}).get('set_weight_link_or_dist', 'dist')
        self.calculate_weight_adj = self.config.get('info', {}).get('calculate_weight_adj', False)
        self.weight_adj_epsilon = self.config.get('info', {}).get('weight_adj_epsilon', 0.1)
        # 初始化
        self.data = None
        self.feature_name = {'X': 'float', 'y': 'float'}  # 此类的输入只有X和y
        self.adj_mx = None
        self.scaler = None
        self.feature_dim = 0
        self.num_nodes = 0
        self._logger = getLogger()
        if os.path.exists(self.data_path + self.geo_file + '.geo'):
            self._load_geo()
        else:
            raise ValueError('Not found .geo file!')
        if os.path.exists(self.data_path + self.rel_file + '.rel'):  # .rel file is not necessary
            self._load_rel()
        else:
            self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)

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

    def _load_grid_geo(self):
        """
        加载.geo文件，格式[geo_id, type, coordinates, row_id, column_id, properties(若干列)]
        """
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.num_nodes = len(self.geo_ids)
        self.geo_to_ind = {}
        self.geo_to_rc = {}
        for index, idx in enumerate(self.geo_ids):
            self.geo_to_ind[idx] = index
        for i in range(geofile.shape[0]):
            self.geo_to_rc[geofile['geo_id'][i]] = [geofile['row_id'][i], geofile['column_id'][i]]
        self.len_row = max(list(geofile['row_id'])) + 1
        self.len_column = max(list(geofile['column_id'])) + 1
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_grids=' + str(len(self.geo_ids))
                          + ', grid_size=' + str((self.len_row, self.len_column)))

    def _load_rel(self):
        """
        加载.rel文件，格式[rel_id, type, origin_id, destination_id, properties(若干列)],
        生成N*N的矩阵，其中权重所在的列名用全局参数`weight_col`来指定,
        .rel文件中缺少的位置的权重填充为np.inf,
        全局参数`calculate_weight_adj`表示是否需要对加载的.rel的默认权重进行进一步计算,
        如果需要，则调用函数self._calculate_adjacency_matrix()进行计算

        Returns:
            np.ndarray: self.adj_mx, N*N的邻接矩阵
        """
        relfile = pd.read_csv(self.data_path + self.rel_file + '.rel')
        if self.weight_col != '':  # 根据weight_col确认权重列
            if isinstance(self.weight_col, list):
                if len(self.weight_col) != 1:
                    raise ValueError('`weight_col` parameter must be only one column!')
                self.weight_col = self.weight_col[0]
            self.distance_df = relfile[~relfile[self.weight_col].isna()][[
                'origin_id', 'destination_id', self.weight_col]]
        else:
            if len(relfile.columns) != 5:  # properties不只一列，且未指定weight_col，报错
                raise ValueError("Don't know which column to be loaded! Please set `weight_col` parameter!")
            else:  # properties只有一列，那就默认这一列是权重列
                self.weight_col = relfile.columns[-1]
                self.distance_df = relfile[~relfile[self.weight_col].isna()][[
                    'origin_id', 'destination_id', self.weight_col]]
        # 把数据转换成矩阵的形式
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        if self.init_weight_inf_or_zero.lower() == 'inf':
            self.adj_mx[:] = np.inf
        for row in self.distance_df.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            if self.set_weight_link_or_dist.lower() == 'dist':  # 保留原始的距离数值
                self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = row[2]
            else:  # self.set_weight_link_or_dist.lower()=='link' 只保留01的邻接性
                self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = 1
        self._logger.info("Loaded file " + self.rel_file + '.rel, shape=' + str(self.adj_mx.shape))
        # 计算权重
        if self.calculate_weight_adj:
            self._calculate_adjacency_matrix()

    def _load_grid_rel(self):
        """
        根据网格结构构建邻接矩阵，一个格子跟他周围的8个格子邻接

        Returns:
            np.ndarray: self.adj_mx, N*N的邻接矩阵
        """
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        dirs = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        for i in range(self.len_row):
            for j in range(self.len_column):
                index = i * self.len_column + j  # grid_id
                for d in dirs:
                    nei_i = i + d[0]
                    nei_j = j + d[1]
                    if nei_i >= 0 and nei_i < self.len_row and nei_j >= 0 and nei_j < self.len_column:
                        nei_index = nei_i * self.len_column + nei_j  # neighbor_grid_id
                        self.adj_mx[index][nei_index] = 1
                        self.adj_mx[nei_index][index] = 1
        self._logger.info("Generate grid rel file, shape=" + str(self.adj_mx.shape))

    def _calculate_adjacency_matrix(self):
        """
        使用带有阈值的高斯核计算邻接矩阵的权重，如果有其他的计算方法，可以覆盖这个函数,
        公式为：$ w_{ij} = \exp \left(- \\frac{d_{ij}^{2}}{\sigma^{2}} \\right) $, $\sigma$ 是方差,
        小于阈值`weight_adj_epsilon`的值设为0：$  w_{ij}[w_{ij}<\epsilon]=0 $

        Returns:
            np.ndarray: self.adj_mx, N*N的邻接矩阵
        """
        self._logger.info("Start Calculate the weight by Gauss kernel!")
        distances = self.adj_mx[~np.isinf(self.adj_mx)].flatten()
        std = distances.std()
        self.adj_mx = np.exp(-np.square(self.adj_mx / std))
        self.adj_mx[self.adj_mx < self.weight_adj_epsilon] = 0

    def _load_dyna(self, filename):
        """
        加载数据文件(.dyna/.grid/.od/.gridod)，子类必须实现这个方法来指定如何加载数据文件，返回对应的多维数据,
        提供5个实现好的方法加载上述几类文件，并转换成不同形状的数组:
        `_load_dyna_3d`/`_load_grid_3d`/`_load_grid_4d`/`_load_grid_od_4d`/`_load_grid_od_6d`

        Args:
            filename(str): 数据文件名，不包含后缀

        Returns:
            np.ndarray: 数据数组
        """
        raise NotImplementedError('Please implement the function `_load_dyna()`.')

    def _load_dyna_3d(self, filename):
        """
        加载.dyna文件，格式[dyna_id, type, time, entity_id, properties(若干列)],
        .geo文件中的id顺序应该跟.dyna中一致,
        其中全局参数`data_col`用于指定需要加载的数据的列，不设置则默认全部加载

        Args:
            filename(str): 数据文件名，不包含后缀

        Returns:
            np.ndarray: 数据数组, 3d-array: (len_time, num_nodes, feature_dim)
        """
        # 加载数据集
        dynafile = pd.read_csv(self.data_path + filename + '.dyna')
        if self.data_col != '':  # 根据指定的列加载数据集
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:  # str
                data_col = [self.data_col.copy()]
            data_col.insert(0, 'time')
            data_col.insert(1, 'entity_id')
            dynafile = dynafile[data_col]
        else:  # 不指定则加载所有列
            dynafile = dynafile[dynafile.columns[2:]]  # 从time列开始所有列
        # 求时间序列
        self.timesolts = list(dynafile['time'][:int(dynafile.shape[0] / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not dynafile['time'].isna().any():  # 时间没有空值
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        # 转3-d数组
        feature_dim = len(dynafile.columns) - 2
        df = dynafile[dynafile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = []
        for i in range(0, df.shape[0], len_time):
            data.append(df[i:i+len_time].values)
        data = np.array(data, dtype=np.float)  # (len(self.geo_ids), len_time, feature_dim)
        data = data.swapaxes(0, 1)             # (len_time, len(self.geo_ids), feature_dim)
        self._logger.info("Loaded file " + filename + '.dyna' + ', shape=' + str(data.shape))
        return data

    def _load_grid_3d(self, filename):
        """
        加载.grid文件，格式[dyna_id, type, time, row_id, column_id, properties(若干列)],
        .geo文件中的id顺序应该跟.dyna中一致,
        其中全局参数`data_col`用于指定需要加载的数据的列，不设置则默认全部加载,

        Args:
            filename(str): 数据文件名，不包含后缀

        Returns:
            np.ndarray: 数据数组, 3d-array: (len_time, num_grids, feature_dim)
        """
        # 加载数据集
        gridfile = pd.read_csv(self.data_path + filename + '.grid')
        if self.data_col != '':  # 根据指定的列加载数据集
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:  # str
                data_col = [self.data_col.copy()]
            data_col.insert(0, 'time')
            data_col.insert(1, 'row_id')
            data_col.insert(2, 'column_id')
            gridfile = gridfile[data_col]
        else:  # 不指定则加载所有列
            gridfile = gridfile[gridfile.columns[2:]]  # 从time列开始所有列
        # 求时间序列
        self.timesolts = list(gridfile['time'][:int(gridfile.shape[0] / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not gridfile['time'].isna().any():  # 时间没有空值
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        # 转3-d数组
        feature_dim = len(gridfile.columns) - 3
        df = gridfile[gridfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = []
        for i in range(0, df.shape[0], len_time):
            data.append(df[i:i + len_time].values)
        data = np.array(data, dtype=np.float)  # (len(self.geo_ids), len_time, feature_dim)
        data = data.swapaxes(0, 1)             # (len_time, len(self.geo_ids), feature_dim)
        self._logger.info("Loaded file " + filename + '.grid' + ', shape=' + str(data.shape))
        return data

    def _load_grid_4d(self, filename):
        """
        加载.grid文件，格式[dyna_id, type, time, row_id, column_id, properties(若干列)],
        .geo文件中的id顺序应该跟.dyna中一致,
        其中全局参数`data_col`用于指定需要加载的数据的列，不设置则默认全部加载

        Args:
            filename(str): 数据文件名，不包含后缀

        Returns:
            np.ndarray: 数据数组, 4d-array: (len_time, len_row, len_column, feature_dim)
        """
        # 加载数据集
        gridfile = pd.read_csv(self.data_path + filename + '.grid')
        if self.data_col != '':  # 根据指定的列加载数据集
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:  # str
                data_col = [self.data_col.copy()]
            data_col.insert(0, 'time')
            data_col.insert(1, 'row_id')
            data_col.insert(2, 'column_id')
            gridfile = gridfile[data_col]
        else:  # 不指定则加载所有列
            gridfile = gridfile[gridfile.columns[2:]]  # 从time列开始所有列
        # 求时间序列
        self.timesolts = list(gridfile['time'][:int(gridfile.shape[0] / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not gridfile['time'].isna().any():  # 时间没有空值
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        # 转4-d数组
        feature_dim = len(gridfile.columns) - 3
        df = gridfile[gridfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = []
        for i in range(self.len_row):
            tmp = []
            for j in range(self.len_column):
                index = (i * self.len_column + j) * len_time
                tmp.append(df[index:index + len_time].values)
            data.append(tmp)
        data = np.array(data, dtype=np.float)      # (len_row, len_column, len_time, feature_dim)
        data = data.swapaxes(2, 0).swapaxes(1, 2)  # (len_time, len_row, len_column, feature_dim)
        self._logger.info("Loaded file " + filename + '.grid' + ', shape=' + str(data.shape))
        return data

    def _load_grid_od_4d(self, filename):
        """
        加载.gridod文件，格式[dyna_id, type, time, origin_row_id, origin_column_id,
        destination_row_id, destination_column_id, properties(若干列)],
        .geo文件中的id顺序应该跟.dyna中一致,
        其中全局参数`data_col`用于指定需要加载的数据的列，不设置则默认全部加载

        Args:
            filename(str): 数据文件名，不包含后缀

        Returns:
            np.ndarray: 数据数组, 4d-array: (len_time, num_grids, num_grids, feature_dim)
        """
        # 加载数据集
        gridodfile = pd.read_csv(self.data_path + filename + '.gridod')
        if self.data_col != '':  # 根据指定的列加载数据集
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:  # str
                data_col = [self.data_col.copy()]
            data_col.insert(0, 'time')
            data_col.insert(1, 'origin_row_id')
            data_col.insert(2, 'origin_column_id')
            data_col.insert(3, 'destination_row_id')
            data_col.insert(4, 'destination_column_id')
            gridodfile = gridodfile[data_col]
        else:  # 不指定则加载所有列
            gridodfile = gridodfile[gridodfile.columns[2:]]  # 从time列开始所有列
        # 求时间序列
        self.timesolts = list(gridodfile['time'][:int(gridodfile.shape[0] / len(self.geo_ids) / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not gridodfile['time'].isna().any():  # 时间没有空值
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        # 转4-d数组
        feature_dim = len(gridodfile.columns) - 5
        df = gridodfile[gridodfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = np.zeros((len(self.geo_ids), len(self.geo_ids), len_time, feature_dim))
        for oi in range(self.len_row):
            for oj in range(self.len_column):
                origin_index = (oi * self.len_column + oj) * len_time * len(self.geo_ids)  # 每个起点占据len_t*n行
                for di in range(self.len_row):
                    for dj in range(self.len_column):
                        destination_index = (di * self.len_column + dj) * len_time  # 每个终点占据len_t行
                        index = origin_index + destination_index
                        # print(index, index + len_time)
                        # print((oi, oj), (di, dj))
                        # print(oi * self.len_column + oj, di * self.len_column + dj)
                        data[oi * self.len_column + oj][di * self.len_column + dj] = df[index:index + len_time].values
        data = data.transpose((2, 0, 1, 3))  # (len_time, num_grids, num_grids, feature_dim)
        self._logger.info("Loaded file " + filename + '.gridod' + ', shape=' + str(data.shape))
        return data

    def _load_grid_od_6d(self, filename):
        """
        加载.gridod文件，格式[dyna_id, type, time, origin_row_id, origin_column_id,
        destination_row_id, destination_column_id, properties(若干列)],
        .geo文件中的id顺序应该跟.dyna中一致,
        其中全局参数`data_col`用于指定需要加载的数据的列，不设置则默认全部加载

        Args:
            filename(str): 数据文件名，不包含后缀

        Returns:
            np.ndarray: 数据数组, 6d-array: (len_time, len_row, len_column, len_row, len_column, feature_dim)
        """
        # 加载数据集
        gridodfile = pd.read_csv(self.data_path + filename + '.gridod')
        if self.data_col != '':  # 根据指定的列加载数据集
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:  # str
                data_col = [self.data_col.copy()]
            data_col.insert(0, 'time')
            data_col.insert(1, 'origin_row_id')
            data_col.insert(2, 'origin_column_id')
            data_col.insert(3, 'destination_row_id')
            data_col.insert(4, 'destination_column_id')
            gridodfile = gridodfile[data_col]
        else:  # 不指定则加载所有列
            gridodfile = gridodfile[gridodfile.columns[2:]]  # 从time列开始所有列
        # 求时间序列
        self.timesolts = list(gridodfile['time'][:int(gridodfile.shape[0] / len(self.geo_ids) / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not gridodfile['time'].isna().any():  # 时间没有空值
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        # 转6-d数组
        feature_dim = len(gridodfile.columns) - 5
        df = gridodfile[gridodfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = np.zeros((self.len_row, self.len_column, self.len_row, self.len_column, len_time, feature_dim))
        for oi in range(self.len_row):
            for oj in range(self.len_column):
                origin_index = (oi * self.len_column + oj) * len_time * len(self.geo_ids)  # 每个起点占据len_t*n行
                for di in range(self.len_row):
                    for dj in range(self.len_column):
                        destination_index = (di * self.len_column + dj) * len_time  # 每个终点占据len_t行
                        index = origin_index + destination_index
                        # print(index, index + len_time)
                        data[oi][oj][di][dj] = df[index:index + len_time].values
        data = data.transpose((4, 0, 1, 2, 3, 5))  # (len_time, len_row, len_column, len_row, len_column, feature_dim)
        self._logger.info("Loaded file " + filename + '.gridod' + ', shape=' + str(data.shape))
        return data

    def _load_ext(self):
        """
        加载.ext文件，格式[ext_id, time, properties(若干列)],
        其中全局参数`ext_col`用于指定需要加载的数据的列，不设置则默认全部加载

        Returns:
            np.ndarray: 外部数据数组，shape: (timeslots, ext_dim)
        """
        # 加载数据集
        extfile = pd.read_csv(self.data_path + self.ext_file + '.ext')
        if self.ext_col != '':  # 根据指定的列加载数据集
            if isinstance(self.ext_col, list):
                ext_col = self.ext_col.copy()
            else:  # str
                ext_col = [self.ext_col.copy()]
            ext_col.insert(0, 'time')
            extfile = extfile[ext_col]
        else:  # 不指定则加载所有列
            extfile = extfile[extfile.columns[1:]]  # 从time列开始所有列
        # 求时间序列
        self.ext_timesolts = extfile['time']
        self.idx_of_ext_timesolts = dict()
        if not extfile['time'].isna().any():  # 时间没有空值
            self.ext_timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.ext_timesolts))
            self.ext_timesolts = np.array(self.ext_timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.ext_timesolts):
                self.idx_of_ext_timesolts[_ts] = idx
        # 求外部特征数组
        feature_dim = len(extfile.columns) - 1
        df = extfile[extfile.columns[-feature_dim:]].values
        self._logger.info("Loaded file " + self.ext_file + '.ext' + ', shape=' + str(df.shape))
        return df

    def _add_external_information(self, df, ext_data=None):
        """
        将外部数据和原始交通状态数据结合到高维数组中，子类必须实现这个方法来指定如何融合外部数据和交通状态数据,
        如果不想加外部数据，可以把交通状态数据`df`直接返回,
        提供3个实现好的方法适用于不同形状的交通状态数据跟外部数据结合:
        `_add_external_information_3d`/`_add_external_information_4d`/`_add_external_information_6d`

        Args:
            df(np.ndarray): 交通状态数据多维数组
            ext_data(np.ndarray): 外部数据

        Returns:
            np.ndarray: 融合后的外部数据和交通状态数据
        """
        raise NotImplementedError('Please implement the function `_add_external_information()`.')

    def _add_external_information_3d(self, df, ext_data=None):
        """
        增加外部信息（一周中的星期几/day of week，一天中的某个时刻/time of day，外部数据）

        Args:
            df(np.ndarray): 交通状态数据多维数组, (len_time, num_nodes, feature_dim)
            ext_data(np.ndarray): 外部数据

        Returns:
            np.ndarray: 融合后的外部数据和交通状态数据, (len_time, num_nodes, feature_dim_plus)
        """
        num_samples, num_nodes, feature_dim = df.shape
        is_time_nan = np.isnan(self.timesolts).any()
        data_list = [df]
        if self.add_time_in_day and not is_time_nan:
            time_ind = (self.timesolts - self.timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        if self.add_day_in_week and not is_time_nan:
            dayofweek = []
            for day in self.timesolts.astype("datetime64[D]"):
                dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
            day_in_week[np.arange(num_samples), :, dayofweek] = 1
            data_list.append(day_in_week)
        # 外部数据集
        if ext_data is not None:
            if not is_time_nan:
                indexs = []
                for ts in self.timesolts:
                    ts_index = self.idx_of_ext_timesolts[ts]
                    indexs.append(ts_index)
                select_data = ext_data[indexs]  # T * ext_dim 选出所需要的时间步的数据
                for i in range(select_data.shape[1]):
                    data_ind = select_data[:, i]
                    data_ind = np.tile(data_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                    data_list.append(data_ind)
            else:  # 没有给出具体的时间戳，只有外部数据跟原数据等长才能默认对接到一起
                if ext_data.shape[0] == df.shape[0]:
                    select_data = ext_data  # T * ext_dim
                    for i in range(select_data.shape[1]):
                        data_ind = select_data[:, i]
                        data_ind = np.tile(data_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                        data_list.append(data_ind)
        data = np.concatenate(data_list, axis=-1)
        return data

    def _add_external_information_4d(self, df, ext_data=None):
        """
        增加外部信息（一周中的星期几/day of week，一天中的某个时刻/time of day，外部数据）

        Args:
            df(np.ndarray): 交通状态数据多维数组, (len_time, len_row, len_column, feature_dim)
            ext_data(np.ndarray): 外部数据

        Returns:
            np.ndarray: 融合后的外部数据和交通状态数据, (len_time, len_row, len_column, feature_dim_plus)
        """
        num_samples, len_row, len_column, feature_dim = df.shape
        is_time_nan = np.isnan(self.timesolts).any()
        data_list = [df]
        if self.add_time_in_day and not is_time_nan:
            time_ind = (self.timesolts - self.timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, len_row, len_column, 1]).transpose((3, 1, 2, 0))
            data_list.append(time_in_day)
        if self.add_day_in_week and not is_time_nan:
            dayofweek = []
            for day in self.timesolts.astype("datetime64[D]"):
                dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            day_in_week = np.zeros(shape=(num_samples, len_row, len_column, 7))
            day_in_week[np.arange(num_samples), :, :, dayofweek] = 1
            data_list.append(day_in_week)
        # 外部数据集
        if ext_data is not None:
            if not is_time_nan:
                indexs = []
                for ts in self.timesolts:
                    ts_index = self.idx_of_ext_timesolts[ts]
                    indexs.append(ts_index)
                select_data = ext_data[indexs]  # T * ext_dim 选出所需要的时间步的数据
                for i in range(select_data.shape[1]):
                    data_ind = select_data[:, i]
                    data_ind = np.tile(data_ind, [1, len_row, len_column, 1]).transpose((3, 1, 2, 0))
                    data_list.append(data_ind)
            else:  # 没有给出具体的时间戳，只有外部数据跟原数据等长才能默认对接到一起
                if ext_data.shape[0] == df.shape[0]:
                    select_data = ext_data  # T * ext_dim
                    for i in range(select_data.shape[1]):
                        data_ind = select_data[:, i]
                        data_ind = np.tile(data_ind, [1, len_row, len_column, 1]).transpose((3, 1, 2, 0))
                        data_list.append(data_ind)
        data = np.concatenate(data_list, axis=-1)
        return data

    def _add_external_information_6d(self, df, ext_data=None):
        """
        增加外部信息（一周中的星期几/day of week，一天中的某个时刻/time of day，外部数据）

        Args:
            df(np.ndarray): 交通状态数据多维数组,
                (len_time, len_row, len_column, len_row, len_column, feature_dim)
            ext_data(np.ndarray): 外部数据

        Returns:
            np.ndarray: 融合后的外部数据和交通状态数据,
            (len_time, len_row, len_column, len_row, len_column, feature_dim)
        """
        num_samples, len_row, len_column, _, _, feature_dim = df.shape
        is_time_nan = np.isnan(self.timesolts).any()
        data_list = [df]
        if self.add_time_in_day and not is_time_nan:
            time_ind = (self.timesolts - self.timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, len_row, len_column, len_row, len_column, 1]).\
                transpose((5, 1, 2, 3, 4, 0))
            data_list.append(time_in_day)
        if self.add_day_in_week and not is_time_nan:
            dayofweek = []
            for day in self.timesolts.astype("datetime64[D]"):
                dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            day_in_week = np.zeros(shape=(num_samples, len_row, len_column, len_row, len_column, 7))
            day_in_week[np.arange(num_samples), :, :, :, :, dayofweek] = 1
            data_list.append(day_in_week)
        # 外部数据集
        if ext_data is not None:
            if not is_time_nan:
                indexs = []
                for ts in self.timesolts:
                    ts_index = self.idx_of_ext_timesolts[ts]
                    indexs.append(ts_index)
                select_data = ext_data[indexs]  # T * ext_dim 选出所需要的时间步的数据
                for i in range(select_data.shape[1]):
                    data_ind = select_data[:, i]
                    data_ind = np.tile(data_ind, [1, len_row, len_column, len_row, len_column, 1]). \
                        transpose((5, 1, 2, 3, 4, 0))
                    data_list.append(data_ind)
            else:  # 没有给出具体的时间戳，只有外部数据跟原数据等长才能默认对接到一起
                if ext_data.shape[0] == df.shape[0]:
                    select_data = ext_data  # T * ext_dim
                    for i in range(select_data.shape[1]):
                        data_ind = select_data[:, i]
                        data_ind = np.tile(data_ind, [1, len_row, len_column, len_row, len_column, 1]). \
                            transpose((5, 1, 2, 3, 4, 0))
                        data_list.append(data_ind)
        data = np.concatenate(data_list, axis=-1)
        return data

    def _generate_input_data(self, df):
        """
        根据全局参数`input_window`和`output_window`切分输入，产生模型需要的张量输入，
        即使用过去`input_window`长度的时间序列去预测未来`output_window`长度的时间序列

        Args:
            df(np.ndarray): 数据数组，shape: (len_time, ..., feature_dim)

        Returns:
            tuple: tuple contains:
                x(np.ndarray): 模型输入数据，(epoch_size, input_length, ..., feature_dim) \n
                y(np.ndarray): 模型输出数据，(epoch_size, output_length, ..., feature_dim)
        """
        num_samples = df.shape[0]
        # 预测用的过去时间窗口长度 取决于self.input_window
        x_offsets = np.sort(np.concatenate((np.arange(-self.input_window + 1, 1, 1),)))
        # 未来时间窗口长度 取决于self.output_window
        y_offsets = np.sort(np.arange(1, self.output_window + 1, 1))

        x, y = [], []
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))
        for t in range(min_t, max_t):
            x_t = df[t + x_offsets, ...]
            y_t = df[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        return x, y

    def _generate_data(self):
        """
        加载数据文件(.dyna/.grid/.od/.gridod)和外部数据(.ext)，且将二者融合，以X，y的形式返回

        Returns:
            tuple: tuple contains:
                x(np.ndarray): 模型输入数据，(num_samples, input_length, ..., feature_dim) \n
                y(np.ndarray): 模型输出数据，(num_samples, output_length, ..., feature_dim)
        """
        # 处理多数据文件问题
        if isinstance(self.data_files, list):
            data_files = self.data_files
        else:  # str
            data_files = [self.data_files]
        # 加载外部数据
        if self.load_external and os.path.exists(self.data_path + self.ext_file + '.ext'):  # 外部数据集
            ext_data = self._load_ext()
        else:
            ext_data = None
        x_list, y_list = [], []
        for filename in data_files:
            df = self._load_dyna(filename)  # (len_time, ..., feature_dim)
            if self.load_external:
                df = self._add_external_information(df, ext_data)
            x, y = self._generate_input_data(df)
            # x: (num_samples, input_length, ..., input_dim)
            # y: (num_samples, output_length, ..., output_dim)
            x_list.append(x)
            y_list.append(y)
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        self._logger.info("Dataset created")
        self._logger.info("x shape: " + str(x.shape) + ", y shape: " + str(y.shape))
        return x, y

    def _split_train_val_test(self, x, y):
        """
        划分训练集、测试集、验证集，并缓存数据集

        Args:
            x(np.ndarray): 输入数据 (num_samples, input_length, ..., feature_dim)
            y(np.ndarray): 输出数据 (num_samples, input_length, ..., feature_dim)

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        test_rate = 1 - self.train_rate - self.eval_rate
        num_samples = x.shape[0]
        num_test = round(num_samples * test_rate)
        num_train = round(num_samples * self.train_rate)
        num_val = num_samples - num_test - num_train

        # train
        x_train, y_train = x[:num_train], y[:num_train]
        # val
        x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
        # test
        x_test, y_test = x[-num_test:], y[-num_test:]
        self._logger.info("train\t" + "x: " + str(x_train.shape) + "y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + "y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + "y: " + str(y_test.shape))

        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                self.cache_file_name,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                x_val=x_val,
                y_val=y_val,
            )
            self._logger.info('Saved at ' + self.cache_file_name)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _generate_train_val_test(self):
        """
        加载数据集，并划分训练集、测试集、验证集，并缓存数据集

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        x, y = self._generate_data()
        return self._split_train_val_test(x, y)

    def _load_cache_train_val_test(self):
        """
        加载之前缓存好的训练集、测试集、验证集

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        self._logger.info('Loading ' + self.cache_file_name)
        cat_data = np.load(self.cache_file_name)
        x_train = cat_data['x_train']
        y_train = cat_data['y_train']
        x_test = cat_data['x_test']
        y_test = cat_data['y_test']
        x_val = cat_data['x_val']
        y_val = cat_data['y_val']
        self._logger.info("train\t" + "x: " + str(x_train.shape) + "y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + "y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + "y: " + str(y_test.shape))
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _get_scalar(self, x_train, y_train):
        """
        根据全局参数`scaler_type`选择数据归一化方法

        Args:
            x_train: 训练数据X
            y_train: 训练数据y

        Returns:
            Scaler: 归一化对象
        """
        if self.scaler_type == "normal":
            scaler = NormalScaler(
                maxx=max(x_train[..., :self.output_dim].max(), y_train[..., :self.output_dim].max()))
            self._logger.info('NormalScaler max: ' + str(scaler.max))
        elif self.scaler_type == "standard":
            scaler = StandardScaler(mean=x_train[..., :self.output_dim].mean(),
                                    std=x_train[..., :self.output_dim].std())
            self._logger.info('StandardScaler mean: ' + str(scaler.mean) + ', std: ' + str(scaler.std))
        elif self.scaler_type == "minmax01":
            scaler = MinMax01Scaler(
                maxx=max(x_train[..., :self.output_dim].max(), y_train[..., :self.output_dim].max()),
                minn=min(x_train[..., :self.output_dim].min(), y_train[..., :self.output_dim].min()))
            self._logger.info('MinMax01Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif self.scaler_type == "minmax11":
            scaler = MinMax11Scaler(
                maxx=max(x_train[..., :self.output_dim].max(), y_train[..., :self.output_dim].max()),
                minn=min(x_train[..., :self.output_dim].min(), y_train[..., :self.output_dim].min()))
            self._logger.info('MinMax11Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif self.scaler_type == "none":
            scaler = NoneScaler()
            self._logger.info('NoneScaler')
        else:
            raise ValueError('Scaler type error!')
        return scaler

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        # 加载数据集
        x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train, y_train, x_val, y_val, x_test, y_test = self._load_cache_train_val_test()
            else:
                x_train, y_train, x_val, y_val, x_test, y_test = self._generate_train_val_test()
        # 数据归一化
        self.feature_dim = x_train.shape[-1]
        self.scaler = self._get_scalar(x_train, y_train)
        x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_val[..., :self.output_dim] = self.scaler.transform(x_val[..., :self.output_dim])
        y_val[..., :self.output_dim] = self.scaler.transform(y_val[..., :self.output_dim])
        x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])
        if self.normal_external:
            x_train[..., self.output_dim:] = self.scaler.transform(x_train[..., self.output_dim:])
            y_train[..., self.output_dim:] = self.scaler.transform(y_train[..., self.output_dim:])
            x_val[..., self.output_dim:] = self.scaler.transform(x_val[..., self.output_dim:])
            y_val[..., self.output_dim:] = self.scaler.transform(y_val[..., self.output_dim:])
            x_test[..., self.output_dim:] = self.scaler.transform(x_test[..., self.output_dim:])
            y_test[..., self.output_dim:] = self.scaler.transform(y_test[..., self.output_dim:])
        # 把训练集的X和y聚合在一起成为list，测试集验证集同理
        # x_train/y_train: (num_samples, input_length, ..., feature_dim)
        # train_data(list): train_data[i]是一个元组，由x_train[i]和y_train[i]组成
        train_data = list(zip(x_train, y_train))
        eval_data = list(zip(x_val, y_val))
        test_data = list(zip(x_test, y_test))
        # 转Dataloader
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        """
        返回数据集特征，子类必须实现这个函数，返回必要的特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        raise NotImplementedError('Please implement the function `get_data_feature()`.')
