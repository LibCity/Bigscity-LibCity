import os
import pandas as pd
import numpy as np
import datetime
import torch
import copy
from libcity.utils import ensure_dir
from torch.utils.data import DataLoader
from libcity.data.dataset import TrafficStatePointDataset
from libcity.data.list_dataset import ListDataset
from libcity.data.batch import Batch



class ASTGNNDataset(TrafficStatePointDataset):
    def __init__(self, config):
        super().__init__(config)
        self.batch_size = config.get("batch_size", 4)
        self.num_of_weeks = config.get("num_of_weeks", 0)
        self.num_of_days = config.get("num_of_days", 0)
        self.num_of_hours = config.get("num_of_hours", 1)
        self.points_per_hour = config.get("points_per_hour", 12)
        self.num_for_predict = config.get("num_for_predict", 12)
        self.device = config.get('device', torch.device('cpu'))
        self.feature_name = {'En': 'float', 'De': 'float', 'y': 'float'}
        

    def search_data(self, sequence_length, num_of_depend, label_start_idx,
                    num_for_predict, units, points_per_hour):
        '''
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
        '''

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

    def get_sample_indices(self, data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
        '''
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
        '''
        week_sample, day_sample, hour_sample = None, None, None

        if label_start_idx + num_for_predict > data_sequence.shape[0]:
            return week_sample, day_sample, hour_sample, None

        if num_of_weeks > 0:
            week_indices = self.search_data(data_sequence.shape[0], num_of_weeks,
                                    label_start_idx, num_for_predict,
                                    7 * 24, points_per_hour)
            if not week_indices:
                return None, None, None, None

            week_sample = np.concatenate([data_sequence[i: j]
                                        for i, j in week_indices], axis=0)

        if num_of_days > 0:
            day_indices = self.search_data(data_sequence.shape[0], num_of_days,
                                    label_start_idx, num_for_predict,
                                    24, points_per_hour)
            if not day_indices:
                return None, None, None, None

            day_sample = np.concatenate([data_sequence[i: j]
                                        for i, j in day_indices], axis=0)

        if num_of_hours > 0:
            hour_indices = self.search_data(data_sequence.shape[0], num_of_hours,
                                    label_start_idx, num_for_predict,
                                    1, points_per_hour)
            if not hour_indices:
                return None, None, None, None

            hour_sample = np.concatenate([data_sequence[i: j]
                                        for i, j in hour_indices], axis=0)

        target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

        return week_sample, day_sample, hour_sample, target
    
    def MinMaxnormalization(self, train, val, test):
        '''
        Parameters
        ----------
        train, val, test: np.ndarray (B,N,F,T)
        Returns
        ----------
        stats: dict, two keys: mean and std
        train_norm, val_norm, test_norm: np.ndarray,
                                        shape is the same as original
        '''

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
        all_data = np.load(self.cache_file_name)
        train_x = all_data['train_x']
        train_target = all_data['train_target']
        train_timestamp = all_data['train_timestamp']
        val_x = all_data['val_x']
        val_target = all_data['val_target']
        val_timestamp = all_data['val_timestamp']
        test_x = all_data['test_x']
        test_target = all_data['test_target']
        test_timestamp = all_data['test_timestamp']
        _max = all_data['mean']
        _min = all_data['std']
        self._logger.info("train\t" + "x: " + str(train_x.shape) + ", target: " + str(train_target.shape)+ ", timestamp: " + str(train_timestamp.shape))
        self._logger.info("val\t" + "x: " + str(val_x.shape) + ", target: " + str(val_target.shape)+ ", timestamp: " + str(val_timestamp.shape))
        self._logger.info("test\t" + "x: " + str(test_x.shape) + ", target: " + str(test_target.shape)+ ", timestamp: " + str(test_timestamp.shape))
        return train_x, train_target, train_timestamp, val_x, val_target, val_timestamp,test_x, test_target, test_timestamp, _max, _min
    
    def _generate_train_val_test(self):
        # 处理多数据文件问题
        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:  # str
            data_files = [self.data_files].copy()
        all_samples = []
        for filename in data_files:
            data_seq = self._load_dyna(filename)
            for idx in range(data_seq.shape[0]):
                sample = self.get_sample_indices(data_seq, self.num_of_weeks, self.num_of_days,
                                            self.num_of_hours, idx, self.num_for_predict,
                                            self.points_per_hour)
                if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
                    continue

                week_sample, day_sample, hour_sample, target = sample

                sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]

                if self.num_of_weeks > 0:
                    week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
                    sample.append(week_sample)

                if self.num_of_days > 0:
                    day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
                    sample.append(day_sample)

                if self.num_of_hours > 0:
                    hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
                    sample.append(hour_sample)

                target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
                sample.append(target)

                time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
                sample.append(time_sample)

                all_samples.append(sample)
        
        split_line1 = int(len(all_samples) * self.train_rate)
        split_line2 = int(len(all_samples) * (self.train_rate + self.eval_rate))

        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line1])]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
        validation_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[split_line1: split_line2])]
        testing_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[split_line2:])]

        train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T'), concat multiple time series segments (for week, day, hour) together
        val_x = np.concatenate(validation_set[:-2], axis=-1)
        test_x = np.concatenate(testing_set[:-2], axis=-1)

        train_target = training_set[-2]  # (B,N,T)
        val_target = validation_set[-2]
        test_target = testing_set[-2]

        train_timestamp = training_set[-1]  # (B,1)
        val_timestamp = validation_set[-1]
        test_timestamp = testing_set[-1]

        # max-min normalization on x
        (stats, train_x_norm, val_x_norm, test_x_norm) = self.MinMaxnormalization(train_x, val_x, test_x)

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
        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(self.cache_file_name,
                            train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                            train_timestamp=all_data['train']['timestamp'],
                            val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                            val_timestamp=all_data['val']['timestamp'],
                            test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                            test_timestamp=all_data['test']['timestamp'],
                            mean=all_data['stats']['_max'], std=all_data['stats']['_min']
                            )
            self._logger.info('Saved at ' + self.cache_file_name)
        return train_x_norm, train_target, train_timestamp, val_x_norm, val_target, val_timestamp,test_x_norm, test_target, test_timestamp, stats['_max'], stats['_min']
            
    
    def get_data(self):
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                train_x, train_target, train_timestamp, val_x, val_target, val_timestamp, test_x, test_target, test_timestamp, _max, _min = self._load_cache_train_val_test()
            else:
                train_x, train_target, train_timestamp, val_x, val_target, val_timestamp, test_x, test_target, test_timestamp, _max, _min = self._generate_train_val_test()
        def max_min_normalization(x, _max, _min):
            x = 1. * (x - _min)/(_max - _min)
            x = x * 2. - 1.
            return x
        def collator(indices):
            batch = Batch(self.feature_name)
            for item in indices:
                batch.append(copy.deepcopy(item))
            return batch
        # pre process
        train_x = train_x[:, :, 0:1, :]
        val_x = val_x[:, :, 0:1, :]
        test_x = test_x[:, :, 0:1, :]
        
        train_target_norm = max_min_normalization(train_target, _max[:, :, 0, :], _min[:, :, 0, :])
        test_target_norm = max_min_normalization(test_target, _max[:, :, 0, :], _min[:, :, 0, :])
        val_target_norm = max_min_normalization(val_target, _max[:, :, 0, :], _min[:, :, 0, :])
        #  ------- train_loader -------
        train_decoder_input_start = train_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
        train_decoder_input_start = np.squeeze(train_decoder_input_start, 2)  # (B,N,T(1))
        train_decoder_input = np.concatenate((train_decoder_input_start, train_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

        # train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor)  # (B, N, F, T)
        # train_decoder_input_tensor = torch.from_numpy(train_decoder_input).type(torch.FloatTensor)  # (B, N, T)
        # train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor)  # (B, N, T)

        # train_data = list(zip(train_x_tensor, train_decoder_input_tensor, train_target_tensor))
        train_data = list(zip(train_x, train_decoder_input, train_target_norm))
        train_dataset = ListDataset(train_data)
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, collate_fn=collator,
                                  shuffle=True)
        
        # train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_decoder_input_tensor, train_target_tensor)
        # self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        #  ------- val_loader -------
        val_decoder_input_start = val_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
        val_decoder_input_start = np.squeeze(val_decoder_input_start, 2)  # (B,N,T(1))
        val_decoder_input = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

        # val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor)  # (B, N, F, T)
        # val_decoder_input_tensor = torch.from_numpy(val_decoder_input).type(torch.FloatTensor)  # (B, N, T)
        # val_target_tensor = torch.from_numpy(val_target_norm).type(torch.FloatTensor)  # (B, N, T)

        # eval_data = list(zip(val_x_tensor, val_decoder_input_tensor, val_target_tensor))
        eval_data = list(zip(val_x, val_decoder_input, val_target_norm))
        eval_dataset = ListDataset(eval_data)
        self.eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, collate_fn=collator,
                                  shuffle=False)
        # eval_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_decoder_input_tensor, val_target_tensor)
        # self.eval_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size)

        #  ------- test_loader -------
        test_decoder_input_start = test_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
        test_decoder_input_start = np.squeeze(test_decoder_input_start, 2)  # (B,N,T(1))
        test_decoder_input = np.concatenate((test_decoder_input_start, test_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

        # test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor)  # (B, N, F, T)
        # test_decoder_input_tensor = torch.from_numpy(test_decoder_input).type(torch.FloatTensor)  # (B, N, T)
        # test_target_tensor = torch.from_numpy(test_target_norm).type(torch.FloatTensor)  # (B, N, T)

        # test_data = list(zip(val_x_tensor, val_decoder_input_tensor, val_target_tensor))
        test_data = list(zip(test_x, test_decoder_input, test_target_norm))
        test_dataset = ListDataset(test_data)
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, collate_fn=collator,
                                  shuffle=False)

        # test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_decoder_input_tensor, test_target_tensor)
        # self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size)

        # print
        # print('train:', train_x_tensor.size(), train_decoder_input_tensor.size(), train_target_tensor.size())
        # print('val:', val_x_tensor.size(), val_decoder_input_tensor.size(), val_target_tensor.size())
        # print('test:', test_x_tensor.size(), test_decoder_input_tensor.size(), test_target_tensor.size())

        
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader
