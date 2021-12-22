from libcity.data.dataset import TrafficStatePointDataset
import os
import numpy as np
import argparse
import configparser
import torch
from libcity.data.batch import Batch
import pandas as pd
import csv
from libcity.utils import  MinMax11Scaler

def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def load_graphdata_normY_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, batch_size, shuffle=True, percent=1.0):
    '''
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注： 从文件读入的数据，x,y都是归一化后的值
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''
    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks) + '.npz')

    print('load file:', filename)

    file_data = np.load(filename)
    train_x = file_data['train_x']  # (10181, 307, 3, 12)
    train_x = train_x[:, :, 0:1, :]
    train_target = file_data['train_target']  # (10181, 307, 12)
    train_timestamp = file_data['train_timestamp']  # (10181, 1)

    train_x_length = train_x.shape[0]
    scale = int(train_x_length*percent)
    print('ori length:', train_x_length, ', percent:', percent, ', scale:', scale)
    train_x = train_x[:scale]
    train_target = train_target[:scale]
    train_timestamp = train_timestamp[:scale]

    val_x = file_data['val_x']
    val_x = val_x[:, :, 0:1, :]
    val_target = file_data['val_target']
    val_timestamp = file_data['val_timestamp']

    test_x = file_data['test_x']
    test_x = test_x[:, :, 0:1, :]
    test_target = file_data['test_target']
    test_timestamp = file_data['test_timestamp']

    _max = file_data['mean']  # (1, 1, 3, 1)
    _min = file_data['std']  # (1, 1, 3, 1)

    # 统一对y进行归一化，变成[-1,1]之间的值
    train_target_norm = max_min_normalization(train_target, _max[:, :, 0, :], _min[:, :, 0, :])
    test_target_norm = max_min_normalization(test_target, _max[:, :, 0, :], _min[:, :, 0, :])
    val_target_norm = max_min_normalization(val_target, _max[:, :, 0, :], _min[:, :, 0, :])

    #  ------- train_loader -------
    train_decoder_input_start = train_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    train_decoder_input_start = np.squeeze(train_decoder_input_start, 2)  # (B,N,T(1))
    train_decoder_input = np.concatenate((train_decoder_input_start, train_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor) # (B, N, F, T)
    train_decoder_input_tensor = torch.from_numpy(train_decoder_input).type(torch.FloatTensor)  # (B, N, T)
    train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_decoder_input_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    #  ------- val_loader -------
    val_decoder_input_start = val_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    val_decoder_input_start = np.squeeze(val_decoder_input_start, 2)  # (B,N,T(1))
    val_decoder_input = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor) # (B, N, F, T)
    val_decoder_input_tensor = torch.from_numpy(val_decoder_input).type(torch.FloatTensor)  # (B, N, T)
    val_target_tensor = torch.from_numpy(val_target_norm).type(torch.FloatTensor)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_decoder_input_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    #  ------- test_loader -------
    test_decoder_input_start = test_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    test_decoder_input_start = np.squeeze(test_decoder_input_start, 2)  # (B,N,T(1))
    test_decoder_input = np.concatenate((test_decoder_input_start, test_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor) # (B, N, F, T)
    test_decoder_input_tensor = torch.from_numpy(test_decoder_input).type(torch.FloatTensor) # (B, N, T)
    test_target_tensor = torch.from_numpy(test_target_norm).type(torch.FloatTensor)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_decoder_input_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # print
    print('train:', train_x_tensor.size(), train_decoder_input_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_decoder_input_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_decoder_input_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max, _min



def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    """
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    """

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    """
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
    label_start_idx: int, the first index of predicting target, 预测值开始的那个点
    num_for_predict: int,
                     the number of points will be predicted for each sample
    points_per_hour: int, default 12, number of points per hour
    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)
    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)
    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)
    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    """
    week_sample, day_sample, hour_sample = None, None, None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target


def MinMaxnormalization(train, val, test):
    """
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    """

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same

    _max = train.max(axis=(0, 1, 3), keepdims=True)
    _min = train.min(axis=(0, 1, 3), keepdims=True)

    print('_max.shape:', _max.shape)
    print('_min.shape:', _min.shape)

    def normalize(x):
        x = 1. * (x - _min) / (_max - _min)
        x = 2. * x - 1.
        return x

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_max': _max, '_min': _min}, train_norm, val_norm, test_norm


class ASTGNNDatasets(TrafficStatePointDataset):
    def __init__(self, config):
        super().__init__(config)
        points_per_hour = int(config['points_per_hour'])
        num_for_predict = int(config['num_for_predict'])
        num_of_weeks = int(config['num_of_weeks'])
        num_of_days = int(config['num_of_days'])
        num_of_hours = int(config['num_of_hours'])
        graph_signal_matrix_filename = config['graph_signal_matrix_filename']
        self.id_filename = config['id_filename']
        # data = np.load(graph_signal_matrix_filename)
        with open(self.id_filename, 'r') as f:
            f.readline()  # 略过表头那一行
            reader = csv.reader(f)
            self.id_dict={int(i[0]): idx for idx, i in enumerate(reader)}

        all_data = self.read_and_generate_dataset_encoder_decoder(graph_signal_matrix_filename, num_of_weeks, num_of_days, num_of_hours, num_for_predict, points_per_hour=points_per_hour, save=True)
    ##  数据预处理
        adj_filename = config['adj_filename']
        epochs = int(config['epochs'])
        fine_tune_epochs = int(config['fine_tune_epochs'])
        print('total training epoch, fine tune epoch:', epochs, ',' , fine_tune_epochs, flush=True)
        batch_size = int(config['batch_size'])
        print('batch_size:', batch_size, flush=True)
        direction = int(config['direction'])
        # direction = 1 means: if i connected to j, adj[i,j]=1;
        # direction = 2 means: if i connected to j, then adj[i,j]=adj[j,i]=1
        if direction == 2:
            self.adj_mx, distance_mx = self.get_adjacency_matrix_2direction(adj_filename)
        if direction == 1:
            self.adj_mx, distance_mx = self.get_adjacency_matrix(adj_filename)
        self.train_loader, self.train_target_tensor, self.val_loader, self.val_target_tensor, self.test_loader, self.test_target_tensor, self._max, self._min = load_graphdata_normY_channel1(
        graph_signal_matrix_filename, num_of_hours,
        num_of_days, num_of_weeks, batch_size)
        self.scaler = MinMax11Scaler(maxx=self._max, minn=self._min)


    def get_adjacency_matrix(self,distance_df_filename):


        A = np.zeros((len(self.id_dict), len(self.id_dict)),
                        dtype=np.float32)

        distaneA = np.zeros((len(self.id_dict), len(self.id_dict)),
                                dtype=np.float32)

        with open(distance_df_filename, 'r') as f:
            f.readline()  # 略过表头那一行
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 5:
                    continue
                i, j, distance = int(row[2]), int(row[3]), float(row[4])
                A[self.id_dict[i], self.id_dict[j]] = 1
                distaneA[self.id_dict[i], self.id_dict[j]] = distance
        return A, distaneA

    def get_adjacency_matrix_2direction(self,distance_df_filename):
        """
        Parameters
        ----------
        distance_df_filename: str, path of the csv file contains edges information

        num_of_vertices: int, the number of vertices

        Returns
        ----------
        A: np.ndarray, adjacency matrix

        """
        A = np.zeros((len(self.id_dict), len(self.id_dict)),
                        dtype=np.float32)

        distaneA = np.zeros((len(self.id_dict), len(self.id_dict)),
                                dtype=np.float32)

        with open(distance_df_filename, 'r') as f:
            f.readline()  # 略过表头那一行
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 5:
                    continue
                i, j, distance = int(row[2]), int(row[3]), float(row[4])
                A[self.id_dict[i], self.id_dict[j]] = 1
                A[self.id_dict[j], self.id_dict[i]] = 1
                distaneA[self.id_dict[i], self.id_dict[j]] = distance
                distaneA[self.id_dict[j], self.id_dict[i]] = distance
        return A, distaneA


    def read_and_generate_dataset_encoder_decoder(self,graph_signal_matrix_filename,
                                              num_of_weeks, num_of_days,
                                              num_of_hours, num_for_predict,
                                              points_per_hour=12, save=False):

        dynafile = pd.read_csv(graph_signal_matrix_filename)
        dynafile = dynafile[dynafile.columns[2:]]  # 从time列开始所有列
            # 求时间序列

        timesolts = list(dynafile['time'][:int(dynafile.shape[0] / len(self.id_dict))])
        idx_of_timesolts = dict()
        if not dynafile['time'].isna().any():
            timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), timesolts))
            timesolts = np.array(timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(timesolts):
                idx_of_timesolts[_ts] = idx
        feature_dim = len(dynafile.columns) - 2
        df = dynafile[dynafile.columns[-feature_dim:]]
        len_time = len(timesolts)
        data = []
        for i in range(0, df.shape[0], len_time):
            data.append(df[i:i+len_time].values)
        data = np.array(data, dtype=np.float)  # (len(self.geo_ids), len_time, feature_dim)
        data = data.swapaxes(0, 1)  
        data_seq=data
        all_samples = []
        for idx in range(data_seq.shape[0]):
            sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                        num_of_hours, idx, num_for_predict,
                                        points_per_hour)
            if (sample[0] is None) and (sample[1] is None) and (sample[2] is None):
                continue

            week_sample, day_sample, hour_sample, target = sample

            sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]

            if num_of_weeks > 0:
                week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
                sample.append(week_sample)

            if num_of_days > 0:
                day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
                sample.append(day_sample)

            if num_of_hours > 0:
                hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
                sample.append(hour_sample)

            target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
            sample.append(target)

            time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
            sample.append(time_sample)

            all_samples.append(
                sample)  # sampe：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),
            # (1,N,F,Th),(1,N,Tpre),(1,1)]

        split_line1 = int(len(all_samples) * 0.6)
        split_line2 = int(len(all_samples) * 0.8)

        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line1])]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
        validation_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[split_line1: split_line2])]
        testing_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[split_line2:])]

        train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T'), concat multiple time series segments (for
        # week, day, hour) together
        val_x = np.concatenate(validation_set[:-2], axis=-1)
        test_x = np.concatenate(testing_set[:-2], axis=-1)

        train_target = training_set[-2]  # (B,N,T)
        val_target = validation_set[-2]
        test_target = testing_set[-2]

        train_timestamp = training_set[-1]  # (B,1)
        val_timestamp = validation_set[-1]
        test_timestamp = testing_set[-1]

        # max-min normalization on x
        (stats, train_x_norm, val_x_norm, test_x_norm) = MinMaxnormalization(train_x, val_x, test_x)

        all_data = {
            'train': {
                'x': train_x_norm,
                'target': train_target,
                'timestamp': train_timestamp,
            },
            'val': {
                'x': val_x_norm,
                'target': val_target,
                'timestamp': val_timestamp,
            },
            'test': {
                'x': test_x_norm,
                'target': test_target,
                'timestamp': test_timestamp,
            },
            'stats': {
                '_max': stats['_max'],
                '_min': stats['_min'],
            }
        }
        print('train x:', all_data['train']['x'].shape)
        print('train target:', all_data['train']['target'].shape)
        print('train timestamp:', all_data['train']['timestamp'].shape)
        print()
        print('val x:', all_data['val']['x'].shape)
        print('val target:', all_data['val']['target'].shape)
        print('val timestamp:', all_data['val']['timestamp'].shape)
        print()
        print('test x:', all_data['test']['x'].shape)
        print('test target:', all_data['test']['target'].shape)
        print('test timestamp:', all_data['test']['timestamp'].shape)
        print()
        print('train data max :', stats['_max'].shape, stats['_max'])
        print('train data min :', stats['_min'].shape, stats['_min'])

        if save:
            file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
            dirpath = os.path.dirname(graph_signal_matrix_filename)
            filename = os.path.join(dirpath,
                                    file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks))
            print('save file:', filename)
            np.savez_compressed(filename,
                                train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                                train_timestamp=all_data['train']['timestamp'],
                                val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                                val_timestamp=all_data['val']['timestamp'],
                                test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                                test_timestamp=all_data['test']['timestamp'],
                                mean=all_data['stats']['_max'], std=all_data['stats']['_min']
                                )
        return all_data
    def get_data_feature(self):
        return {"adj_mx": self.adj_mx,"scaler":self.scaler}

    def get_data(self):
        train_data=[]
        val_data=[]
        test_data=[]
        for batch_data in self.train_loader:
            encoder_inputs, decoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)
            labels = labels.unsqueeze(-1)
            batch=Batch(
                {
                    'encoder_inputs':'float',
                    'decoder_inputs':'float',
                    'y':'float'
                }
            )
            batch['encoder_inputs']=encoder_inputs.cpu()
            batch['decoder_inputs']=decoder_inputs.cpu()
            batch['y']=labels.cpu()
            train_data.append(batch)
        for batch_data in self.val_loader:
            encoder_inputs, decoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)

            labels = labels.unsqueeze(-1)
            batch=Batch(
                {
                    'encoder_inputs':'float',
                    'decoder_inputs':'float',
                    'y':'float'
                }
            )
            batch['encoder_inputs']=encoder_inputs.cpu()
            batch['decoder_inputs']=decoder_inputs.cpu()
            batch['y']=labels.cpu()
            val_data.append(batch)
        for batch_data in self.test_loader:
            encoder_inputs, decoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)

            labels = labels.unsqueeze(-1)
            batch=Batch(
                {
                    'encoder_inputs':'float',
                    'decoder_inputs':'float',
                    'y':'float'
                }
            )
            batch['encoder_inputs']=encoder_inputs.cpu()
            batch['decoder_inputs']=decoder_inputs.cpu()
            batch['y']=labels.cpu()
            test_data.append(batch)
        return train_data,val_data,test_data