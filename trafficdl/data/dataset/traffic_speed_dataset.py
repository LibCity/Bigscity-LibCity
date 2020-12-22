import os
import json
import pandas as pd
import numpy as np
import math
import datetime

from trafficdl.data.dataset import AbstractDataset
from trafficdl.utils.dataset import parseTime, calculateBaseTime, calculateTimeOff
from trafficdl.data.utils import generate_dataloader

class TrafficSpeedDataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        parameters_str = ''
        for key in self.config:
            parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join('./trafficdl/cache/dataset_cache/', 'speed_{}.npz'.format(parameters_str))
        self.cache_file_folder = './trafficdl/cache/dataset_cache/'
        self.data_path = os.path.join('./raw_data/', self.config['dataset'])
        self.data = None
        self.feature_name = ['X', 'y']
        self.load_geo()
        self.load_rel()

    def load_geo(self):
        geofile = pd.read_csv(self.data_path + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.geo_to_ind = {}
        for index, id in enumerate(self.geo_ids):
            self.geo_to_ind[id] = index

    def load_rel(self):
        relfile = pd.read_csv(self.data_path + '.rel')
        self.distance_df = relfile[~relfile['cost'].isna()][['origin_id', 'destination_id', 'cost']]
        self.calcule_adj_mx()

    def calcule_adj_mx(self):
        self.adj_mx = self.get_adjacency_matrix(self.distance_df, self.geo_ids)
        # 验证计算的结果跟数据集的link_weight列是否一致
        # for i in range(relfile.shape[0]):
        #     link_weight = relfile['link_weight'][i]
        #     fromid = relfile['origin_id'][i]
        #     toid = relfile['destination_id'][i]
        #     cal_weight = self.adj_mx[self.geo_to_ind[fromid]][self.geo_to_ind[toid]]
        #     if abs(link_weight - cal_weight) > 0.001:
        #         print(i)
        #         print(relfile.iloc[i])
        #         print(cal_weight)
        #         print('---------------------')

    def get_adjacency_matrix(self, distance_df, sensor_ids, normalized_k=0.1):
        """
        :param distance_df: data frame with three columns: [from, to, distance].
        :param sensor_ids: list of sensor ids.
        :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
        :return:
        """
        num_sensors = len(sensor_ids)
        dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
        dist_mx[:] = np.inf

        # Fills cells in the matrix with distances.
        for row in distance_df.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            dist_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = row[2]

        # Calculates the standard deviation as theta.
        distances = dist_mx[~np.isinf(dist_mx)].flatten()
        std = distances.std()
        adj_mx = np.exp(-np.square(dist_mx / std))
        # Make the adjacent matrix symmetric by taking the max.
        # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
        # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
        adj_mx[adj_mx < normalized_k] = 0
        return adj_mx

    def load_dyna(self):
        dynafile = pd.read_csv(self.data_path + '.dyna')
        dynafile = dynafile[['time', 'entity_id', 'traffic_speed']]
        self.timesolts = list(dynafile[dynafile['entity_id'] == self.geo_ids[0]]['time'])
        assert(dynafile.shape[0] == len(self.timesolts) * len(self.geo_ids))
        df = pd.DataFrame(columns=self.geo_ids, index=np.array(self.timesolts, dtype='datetime64[ns]'))
        for i in range(len(self.geo_ids)):
            df[self.geo_ids[i]] = dynafile[dynafile['entity_id'] == self.geo_ids[0]]['traffic_speed'].values
        return df

    def generate_graph_seq2seq_io_data(
            self, df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None):
        """
        Generate samples from
        :param df:
        :param x_offsets:
        :param y_offsets:
        :param add_time_in_day:
        :param add_day_in_week:
        :param scaler:
        :return:
        # x: (epoch_size, input_length, num_nodes, input_dim)
        # y: (epoch_size, output_length, num_nodes, output_dim)
        """
        num_samples, num_nodes = df.shape
        data = np.expand_dims(df.values, axis=-1)
        data_list = [data]
        if add_time_in_day:
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        if add_day_in_week:
            dayofweek = []
            for day in df.index.values.astype("datetime64[D]"):
                dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
            day_in_week[np.arange(num_samples), :, dayofweek] = 1
            data_list.append(day_in_week)

        data = np.concatenate(data_list, axis=-1)  # (34272, 207, 9)
        # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
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
        return x, y  # 四维数组  (*, 12, 207, num_feature)

    def generate_train_val_test(self):
        df = self.load_dyna()
        # 0 is the latest observed sample.
        x_offsets = np.sort(np.concatenate((np.arange(-11, 1, 1),)))
        # Predict the next one hour
        y_offsets = np.sort(np.arange(1, 13, 1))
        # x: (num_samples, input_length, num_nodes, input_dim)
        # y: (num_samples, output_length, num_nodes, output_dim)
        x, y = self.generate_graph_seq2seq_io_data(
            df,
            x_offsets=x_offsets,
            y_offsets=y_offsets,
            add_time_in_day=True,
            add_day_in_week=False,
        )
        # Write the data into npz file.
        # num_test = 6831, using the last 6831 examples as testing.
        # for the rest: 7/8 is used for training, and 1/8 is used for validation.
        train_rate = self.config['train_rate']
        eval_rate = self.config['eval_rate']
        num_samples = x.shape[0]
        num_test = round(num_samples * (1 - train_rate - eval_rate))
        num_train = round(num_samples * train_rate)
        num_val = num_samples - num_test - num_train

        # train
        x_train, y_train = x[:num_train], y[:num_train]
        # val
        x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
        # test
        x_test, y_test = x[-num_test:], y[-num_test:]

        if self.config['cache_dataset']:
            if not os.path.exists(self.cache_file_folder):
                os.makedirs(self.cache_file_folder)
            print("train", "x: ", x_train.shape, "y: ", y_train.shape)
            print("eval", "x: ", x_val.shape, "y: ", y_val.shape)
            print("test", "x: ", x_test.shape, "y: ", y_test.shape)
            np.savez_compressed(
                self.cache_file_name,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                x_val=x_val,
                y_val=y_val,
                x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
                y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
            )
        return x_train, y_train, x_val, y_val, x_test, y_test, x_offsets, y_offsets

    def load_cache_train_val_test(self):
        cat_data = np.load(self.cache_file_name)
        x_train = cat_data['x_train']
        y_train = cat_data['y_train']
        x_test = cat_data['x_test']
        y_test = cat_data['y_test']
        x_val = cat_data['x_val']
        y_val = cat_data['y_val']
        x_offsets = cat_data['x_offsets']
        y_offsets = cat_data['y_offsets']
        return x_train, y_train, x_val, y_val, x_test, y_test, x_offsets, y_offsets

    def get_data(self):
        '''
        return:
            train_dataloader (pytorch.DataLoader)
            eval_dataloader (pytorch.DataLoader)
            test_dataloader (pytorch.DataLoader)
            all the dataloaders are composed of Batch (class)
        '''
        x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
        if self.data == None:
            self.data = {}
            if self.config['cache_dataset'] and os.path.exists(self.cache_file_name):
                x_train, y_train, x_val, y_val, x_test, y_test, x_offsets, y_offsets = self.load_cache_train_val_test()
            else:
                x_train, y_train, x_val, y_val, x_test, y_test, x_offsets, y_offsets = self.generate_train_val_test()

        self.scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())
        x_train[..., 0] = self.scaler.transform(x_train[..., 0])
        y_train[..., 0] = self.scaler.transform(y_train[..., 0])
        x_val[..., 0] = self.scaler.transform(x_val[..., 0])
        y_val[..., 0] = self.scaler.transform(y_val[..., 0])
        x_test[..., 0] = self.scaler.transform(x_test[..., 0])
        y_test[..., 0] = self.scaler.transform(y_test[..., 0])

        train_data = list(zip(x_train, y_train))
        eval_data = list(zip(x_val, y_val))
        test_data = list(zip(x_test, y_test))
        return generate_dataloader(train_data, eval_data, test_data,
                                   self.feature_name, self.config['batch_size'],
                                   self.config['num_workers'])

    def get_data_feature(self):
        '''
        如果模型使用了 embedding 层，一般是需要根据数据集的 loc_size、tim_size、uid_size 等特征来确定 embedding 层的大小的
        故该方法返回一个 dict，包含表示层能够提供的数据集特征
        return:
            data_feature (dict)
        '''
        raise NotImplementedError("get_data_feature not implemented")


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
