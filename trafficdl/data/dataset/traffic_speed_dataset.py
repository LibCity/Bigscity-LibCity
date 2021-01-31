import os
import pandas as pd
import numpy as np
import datetime
from logging import getLogger

from trafficdl.data.dataset import AbstractDataset
from trafficdl.data.utils import generate_dataloader
from trafficdl.utils import StandardScaler, ensure_dir


class TrafficSpeedDataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset', 'METR_LA')
        self.points_per_hour = self.config.get('points_per_hour', 12)
        self.input_window = self.config.get('input_window', 12)
        self.output_window = self.config.get('output_window', 12)
        self.output_dim = self.config.get('output_dim', 1)
        self.batch_size = self.config.get('batch_size', 64)
        self.num_workers = self.config.get('num_workers', 1)
        self.add_time_in_day = self.config.get('add_time_in_day', False)
        self.add_day_in_week = self.config.get('add_day_in_week', False)
        self.pad_with_last_sample = self.config.get('pad_with_last_sample', True)
        self.weight_col = self.config.get('weight_col', 'cost')
        self.calculate_weight = self.config.get('calculate_weight', False)
        self.adj_epsilon = self.config.get('adj_epsilon', 0.1)
        self.train_rate = self.config.get('train_rate', 0.7)
        self.eval_rate = self.config.get('eval_rate', 0.1)
        # TODO: 分割比例加进来
        parameters_str = str(self.dataset) + '_' + str(self.input_window) + '_' + str(self.output_window) + '_' \
                         + str(self.batch_size) + '_' + str(self.add_time_in_day) + '_' \
                         + str(self.add_day_in_week) + '_' + str(self.pad_with_last_sample)
        self.cache_file_name = os.path.join('./trafficdl/cache/dataset_cache/', 'speed_{}.npz'.format(parameters_str))
        self.cache_file_folder = './trafficdl/cache/dataset_cache/'
        ensure_dir(self.cache_file_folder)
        self.cache_dataset = self.config.get('cache_dataset', True)
        self.data_path = os.path.join('./raw_data/', self.dataset)
        self.data = None
        self.feature_name = {'X': 'float', 'y': 'float'}
        self.adj_mx = None
        self.scaler = None
        self.feature_dim = 0
        self.num_nodes = 0
        self._logger = getLogger()
        self._load_geo()
        self._load_rel()
        # TODO: 加载数据集的config.json文件

    def _load_geo(self):
        geofile = pd.read_csv(self.data_path + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.num_nodes = len(self.geo_ids)
        self.geo_to_ind = {}
        for index, id in enumerate(self.geo_ids):
            self.geo_to_ind[id] = index

    def _load_rel(self):
        relfile = pd.read_csv(self.data_path + '.rel')
        self.distance_df = relfile[~relfile[self.weight_col].isna()][
            ['origin_id', 'destination_id', self.weight_col]]
        # 把数据转换成矩阵的形式
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        self.adj_mx[:] = np.inf  # 0 ? inf
        for row in self.distance_df.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = row[2]
        # 计算权重
        if self.calculate_weight:
            self._calculate_adjacency_matrix()

    def _calculate_adjacency_matrix(self):
        # Calculates the standard deviation as theta.
        distances = self.adj_mx[~np.isinf(self.adj_mx)].flatten()
        std = distances.std()
        self.adj_mx = np.exp(-np.square(self.adj_mx / std))
        self.adj_mx[self.adj_mx < self.adj_epsilon] = 0

    def _load_dyna(self):
        dynafile = pd.read_csv(self.data_path + '.dyna')
        dynafile = dynafile[['time', 'entity_id', 'traffic_speed']]
        self.timesolts = list(dynafile[dynafile['entity_id'] == self.geo_ids[0]]['time'])
        assert(dynafile.shape[0] == len(self.timesolts) * len(self.geo_ids))
        df = pd.DataFrame(columns=self.geo_ids, index=np.array(self.timesolts, dtype='datetime64[ns]'))
        for i in range(len(self.geo_ids)):
            df[self.geo_ids[i]] = dynafile[dynafile['entity_id'] == self.geo_ids[i]]['traffic_speed'].values
        return df

    def _generate_graph_seq2seq_io_data(self, df):
        """
        根据input_window和output_window产生模型的输入输出数据的四维张量
        默认第四维的0维度([..., 0])是原始速度数据，其他维度可以是额外的特征数据，如时间
        :param df:
        :return:
        # x: (epoch_size, input_length, num_nodes, feature_dim)
        # y: (epoch_size, output_length, num_nodes, feature_dim)
        """
        num_samples, num_nodes = df.shape
        data = np.expand_dims(df.values, axis=-1)
        data_list = [data]

        if self.add_time_in_day:
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        if self.add_day_in_week:
            dayofweek = []
            for day in df.index.values.astype("datetime64[D]"):
                dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
            day_in_week[np.arange(num_samples), :, dayofweek] = 1
            data_list.append(day_in_week)

        data = np.concatenate(data_list, axis=-1)

        # 预测用的过去时间窗口长度 取决于self.input_window
        x_offsets = np.sort(np.concatenate((np.arange(-self.input_window+1, 1, 1),)))
        # 未来时间窗口长度 取决于self.output_window
        y_offsets = np.sort(np.arange(1, self.output_window+1, 1))

        x, y = [], []
        # t is the index of the last observation.
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
        for t in range(min_t, max_t):
            x_t = data[t + x_offsets, ...]
            y_t = data[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        return x, y

    def _generate_train_val_test(self):
        """
        分解train/test/eval的函数
        :return:
        """
        df = self._load_dyna()
        # x: (num_samples, input_length, num_nodes, input_dim)
        # y: (num_samples, output_length, num_nodes, output_dim)
        x, y = self._generate_graph_seq2seq_io_data(df)
        self._logger.info("Dataset created")
        self._logger.info("x shape: " + str(x.shape) + ", y shape: " + str(y.shape))

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

        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            self._logger.info("train\t" + "x: " + str(x_train.shape) + "y: " + str(y_train.shape))
            self._logger.info("eval\t" + "x: " + str(x_val.shape) + "y: " + str(y_val.shape))
            self._logger.info("test\t" + "x: " + str(x_test.shape) + "y: " + str(y_test.shape))
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

    def _load_cache_train_val_test(self):
        self._logger.info('Loading ' + self.cache_file_name)
        cat_data = np.load(self.cache_file_name)
        x_train = cat_data['x_train']
        y_train = cat_data['y_train']
        x_test = cat_data['x_test']
        y_test = cat_data['y_test']
        x_val = cat_data['x_val']
        y_val = cat_data['y_val']
        return x_train, y_train, x_val, y_val, x_test, y_test

    def get_data(self):
        '''
        获取数据 返回DataLoader
        return:
            train_dataloader (pytorch.DataLoader)
            eval_dataloader (pytorch.DataLoader)
            test_dataloader (pytorch.DataLoader)
            all the dataloaders are composed of Batch (class)
        '''
        x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train, y_train, x_val, y_val, x_test, y_test = self._load_cache_train_val_test()
            else:
                x_train, y_train, x_val, y_val, x_test, y_test = self._generate_train_val_test()
        self.feature_dim = x_train.shape[-1]
        # 特征归一化
        self.scaler = StandardScaler(mean=x_train[..., :self.output_dim].mean(), std=x_train[..., :self.output_dim].std())
        self._logger.info('Scaler mean: ' + str(self.scaler.mean) + ', std: ' + str(self.scaler.std))
        x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_val[..., :self.output_dim] = self.scaler.transform(x_val[..., :self.output_dim])
        y_val[..., :self.output_dim] = self.scaler.transform(y_val[..., :self.output_dim])
        x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])
        # 转List
        train_data = list(zip(x_train, y_train))
        eval_data = list(zip(x_val, y_val))
        test_data = list(zip(x_test, y_test))
        # 转Dataloader
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        '''
        返回数据集特征
        return:
            data_feature (dict)
        '''
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "data_loader": self.eval_dataloader,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim}

