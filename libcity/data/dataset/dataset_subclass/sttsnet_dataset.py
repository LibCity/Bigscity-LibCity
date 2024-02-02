import os
from copy import copy as cp
from datetime import datetime

import numpy as np
import pandas as pd

from libcity.data.dataset import TrafficStateGridDataset
from libcity.data.utils import generate_dataloader
from libcity.utils import ensure_dir


def string2timestamp(strings, T=48):
    """
    strings: list, eg. ['2017080912','2017080913']
    return: list, eg. [Timestamp('2017-08-09 05:30:00'), Timestamp('2017-08-09 06:00:00')]
    """
    timestamps = []

    time_per_slot = 24.0 / T
    num_per_T = T // 24
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:])
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(slot * time_per_slot),
                                                minute=(slot % num_per_T) * int(60.0 * time_per_slot))))

    return timestamps


class STMatrix(object):
    """docstring for STMatrix"""

    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.data_1 = data[:, 0, :, :]
        self.data_2 = data[:, 1, :, :]
        self.timestamps = timestamps
        self.T = T
        func = string2timestamp
        self.pd_timestamps = func(timestamps, T=self.T)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i - 1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i - 1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):
        return self.data[self.get_index[timestamp]]

    def get_matrix_1(self, timestamp):  # in_flow
        ori_matrix = self.data_1[self.get_index[timestamp]]
        new_matrix = ori_matrix[np.newaxis, :]
        # print("new_matrix shape:",new_matrix.shape) #(1, 32, 32)
        return new_matrix

    def get_matrix_2(self, timestamp):  # out_flow
        ori_matrix = self.data_2[self.get_index[timestamp]]
        new_matrix = ori_matrix[np.newaxis, :]
        # print("new_matrix shape:",new_matrix.shape) #(1, 32, 32)
        return new_matrix

    def save(self, fname):
        pass

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset_3D(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_closeness + 1),
                   [PeriodInterval * self.T * j for j in range(1, len_period + 1)],
                   [TrendInterval * self.T * j for j in range(1, len_trend + 1)]]

        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)
        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

            if Flag is False:
                i += 1
                continue

            # closeness
            c_1_depends = list(depends[0])  # in_flow
            c_1_depends.sort(reverse=True)
            # print('----- c_1_depends:',c_1_depends)

            c_2_depends = list(depends[0])  # out_flow
            c_2_depends.sort(reverse=True)
            # print('----- c_2_depends:',c_2_depends)

            x_c_1 = [self.get_matrix_1(self.pd_timestamps[i] - j * offset_frame) for j in
                     c_1_depends]  # [(1,32,32),(1,32,32),(1,32,32)] in
            x_c_2 = [self.get_matrix_2(self.pd_timestamps[i] - j * offset_frame) for j in
                     c_2_depends]  # [(1,32,32),(1,32,32),(1,32,32)] out

            x_c_1_all = np.vstack(x_c_1)  # x_c_1_all.shape  (3, 32, 32)
            x_c_2_all = np.vstack(x_c_2)  # x_c_1_all.shape  (3, 32, 32)

            x_c_1_new = x_c_1_all[np.newaxis, :]  # (1, 3, 32, 32)
            x_c_2_new = x_c_2_all[np.newaxis, :]  # (1, 3, 32, 32)

            x_c = np.vstack([x_c_1_new, x_c_2_new])  # (2, 3, 32, 32)

            # period
            p_depends = list(depends[1])
            if (len(p_depends) > 0):
                p_depends.sort(reverse=True)
                # print('----- p_depends:',p_depends)

                x_p_1 = [self.get_matrix_1(self.pd_timestamps[i] - j * offset_frame) for j in p_depends]
                x_p_2 = [self.get_matrix_2(self.pd_timestamps[i] - j * offset_frame) for j in p_depends]

                x_p_1_all = np.vstack(x_p_1)  # [(3,32,32),(3,32,32),...]
                x_p_2_all = np.vstack(x_p_2)  # [(3,32,32),(3,32,32),...]

                x_p_1_new = x_p_1_all[np.newaxis, :]  # (1, 3, 32, 32)
                x_p_2_new = x_p_2_all[np.newaxis, :]  # (1, 3, 32, 32)

                x_p = np.vstack([x_p_1_new, x_p_2_new])  # (2, 3, 32, 32)

            # trend
            t_depends = list(depends[2])
            if (len(t_depends) > 0):
                t_depends.sort(reverse=True)

                x_t_1 = [self.get_matrix_1(self.pd_timestamps[i] - j * offset_frame) for j in t_depends]
                x_t_2 = [self.get_matrix_2(self.pd_timestamps[i] - j * offset_frame) for j in t_depends]

                x_t_1_all = np.vstack(x_t_1)  # [(3,32,32),(3,32,32),...]
                x_t_2_all = np.vstack(x_t_2)  # [(3,32,32),(3,32,32),...]

                x_t_1_new = x_t_1_all[np.newaxis, :]  # (1, 3, 32, 32)
                x_t_2_new = x_t_2_all[np.newaxis, :]  # (1, 3, 32, 32)

                x_t = np.vstack([x_t_1_new, x_t_2_new])  # (2, 3, 32, 32)

            y = self.get_matrix(self.pd_timestamps[i])

            if len_closeness > 0:
                XC.append(x_c)
            if len_period > 0:
                XP.append(x_p)
            if len_trend > 0:
                XT.append(x_t)
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1

        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)
        return XC, XP, XT, Y, timestamps_Y


class MinMaxNormalization(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X


class STTSNetDataset(TrafficStateGridDataset):

    def __init__(self, config):
        super().__init__(config)
        days_test = config.get("days_test", 28)
        # number of time intervals in one day
        self.T = int(24 * 60 * 60 / self.time_intervals)
        # 通过 self.len_test 划分 test 数据集
        # 剩下的部分通过 train_rate 和 eval_rate 来划分 train 和 eval 部分，所以 train_rate 和 eval_rate 和要为 1
        self.len_test = self.T * days_test
        self.nb_flow = config.get("nb_flow", 2)
        self.prediction_offset = config.get("prediction_offset", 0)
        self.len_closeness = config.get("len_closeness", 10)
        self.len_period = config.get("len_period", 0)
        self.len_trend = config.get("len_trend", 4)
        self.adjust_ext_timestamp = config.get('adjust_ext_timestamp', False)
        # feature_name
        if self.load_external:
            self.feature_name = {'xc': 'float', 'xt': 'float', 'x_ext': 'float', 'y': 'float'}
        else:
            self.feature_name = {'xc': 'float', 'xt': 'float', 'y': 'float'}

    def _load_dyna(self, filename):
        """
        return a 4D tensor of shape (number_of_timeslots, 2, 16, 8), of which `data[i]` is a 3D tensor of shape (2,
        16, 8) at the timeslot `date[i]`, `data[i][0]` is a `16x8` new-flow matrix and `data[i][1]` is a `16x8`
        end-flow matrix.
        """
        return np.transpose(super()._load_grid_4d(filename), (0, 3, 1, 2))

    def _generate_data(self):
        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:  # str
            data_files = [self.data_files].copy()
        self.data_all = []
        self.timestamps_all = []
        for filename in data_files:
            df = self._load_dyna(filename)  # (number_of_timeslots, 2, 16, 8)
            df = df[:, :self.nb_flow]
            df[df < 0] = 0.
            self.data_all.append(df)
            # 时间戳数据
            self.timestamps_all.append(np.char.replace(np.char.replace(np.datetime_as_string(self.timesolts, unit='h'), '-',
                                                                   ''), 'T', '').astype(bytes))
        return self.load_data()

    def load_data(self):
        """

        @return:
        X_train: a list size is 3 [xc(7776, 2, 10, 16, 8), xt(7776, 2, 4, 16, 8), x_ext(7776, 24)]
        Y_train: ndarray(7776, 2, 16, 8)
        X_test: a list size is 3 [xc(672, 2, 10, 16, 8), xt(672, 2, 4, 16, 8), x_ext(672, 24)]
        Y_test: ndarray shape = {tuple: 4} (672, 2, 16, 8)
        mmn: normalization
        metadata_dim: ext dim
        timestamp_train: a list size is 7776 data type is bytes timestamps data eg. 0000 = {bytes_: ()} b'2014012900'
        timestamp_test: a list size is 672 data type is bytes timestamps data eg. 000 = {bytes_: ()} b'2014120400'
        """

        def adjust_timeslots(timeslot):
            """
            adjust timeslots extend by one hour
            """
            timeslot_str = timeslot.decode("utf-8")
            interval = timeslot_str[-2:]
            new_interval = f'{int(interval) + 1:02}'
            return bytes(timeslot_str[:-2] + new_interval, encoding='utf8')

        data_train = np.vstack(cp(self.data_all))[:-self.len_test]
        mmn = MinMaxNormalization()
        mmn.fit(data_train)
        data_all_mmn = [mmn.transform(d) for d in self.data_all]
        XC, XP, XT = [], [], []
        Y = []
        timestamps_Y = []
        for data, timestamps in zip(data_all_mmn, self.timestamps_all):
            st = STMatrix(data, timestamps, self.T, CheckComplete=False)
            _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset_3D(len_closeness=self.len_closeness, len_period=
            self.len_period, len_trend=self.len_trend)
            XC.append(_XC)
            XP.append(_XP)
            XT.append(_XT)
            Y.append(_Y)
            timestamps_Y += _timestamps_Y
        XC = np.vstack(XC)
        XP = np.vstack(XP)
        XT = np.vstack(XT)
        Y = np.vstack(Y)
        len_test = self.len_test
        XC_train, XP_train, XT_train, Y_train = XC[:-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
        XC_test, XP_test, XT_test, Y_test = XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
        timestamp_train, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[-len_test:]
        X_train = []
        X_test = []
        len_closeness = self.len_closeness
        len_period = self.len_period
        len_trend = self.len_trend
        for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
            if l > 0:
                X_train.append(X_)
        for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
            if l > 0:
                X_test.append(X_)
        # add ext data
        metadata_dim = None
        if self.load_external:
            ext_df = self._load_ext()
            # 筛选出符合时间的ext数据
            final_ext_df = []
            if self.adjust_ext_timestamp:
                timeslots = [adjust_timeslots(t) for t in timestamps_Y]
            else:
                timeslots = timestamps_Y
            for slot in timeslots:
                final_ext_df.append(ext_df[self.idx_of_ext_timesolts[slot]])
            # WindSpeed Temperature 0-1 scale
            arr = np.array(final_ext_df)
            arr[:, -1] = 1. * (arr[:, -1] - arr[:, -1].min()) / (arr[:, -1].max() - arr[:, -1].min())
            arr[:, -2] = 1. * (arr[:, -2] - arr[:, -2].min()) / (arr[:, -2].max() - arr[:, -2].min())
            metadata_dim = arr.shape[1]
            meta_feature_train, meta_feature_test = arr[:-len_test], arr[-len_test:]
            # 最后 meta_feature_train 有 24 个 特征值 温度 + 风速 + 13个天气 + holiday + 8个time_feature = 16 + 8 = 24
            X_train.append(meta_feature_train)
            X_test.append(meta_feature_test)
        # y to real value
        Y_train = mmn.inverse_transform(Y_train)  # X is MaxMinNormalized, Y is real value
        Y_test = mmn.inverse_transform(Y_test)
        return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test

    def _load_ext(self):
        # 加载数据集
        extfile = pd.read_csv(self.data_path + self.ext_file + '.ext')
        if self.ext_col != '':  # 根据指定的列加载数据集
            if isinstance(self.ext_col, list):
                ext_col = self.ext_col.copy()
            else:  # str
                ext_col = [self.ext_col].copy()
            ext_col.insert(0, 'time')
            extfile = extfile[ext_col]
        else:  # 不指定则加载所有列
            extfile = extfile[extfile.columns[1:]]  # 从time列开始所有列
        # 求时间序列
        self.ext_timesolts = extfile['time']
        self.idx_of_ext_timesolts = dict()
        if not extfile['time'].isna().any():  # 时间没有空值
            self.ext_timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.ext_timesolts))
            self.ext_timestamps = np.char.replace(np.char.replace(np.char.replace(self.ext_timesolts, ':00:00', ''), ' '
                                                                  , ''), '-', '').astype(bytes)
            for idx, _ts in enumerate(self.ext_timestamps):
                self.idx_of_ext_timesolts[_ts] = idx
        # 求外部特征数组
        feature_dim = len(extfile.columns) - 1
        df = extfile[extfile.columns[-feature_dim:]].values
        self._logger.info("Loaded file " + self.ext_file + '.ext' + ', shape=' + str(df.shape))
        return df

    def split_train_val_test(self, x, y, x_test, y_test):
        """
        将数据根据 train_rate eval_rate 参数来切分数据
        test 部分已经通过参数 self.len_test 提前切分完成，剩下的部分通过 train_rate 和 eval_rate 来划分 train 和 eval 部分，
        所以 train_rate 和 eval_rate 和要为 1

        @param y_test: ndarray shape = {tuple: 4} (672, 2, 16, 8)
        @param x_test: a list size is 3 [xc(672, 2, 10, 16, 8), xt(672, 2, 4, 16, 8), x_ext(672, 24)]
        @param x: a list size is 3 [xc(7776, 2, 10, 16, 8), xt(7776, 2, 4, 16, 8), x_ext(7776, 24)]
        @param y: ndarray(7776, 2, 16, 8)
        @return:
        """
        num_samples = y.shape[0]
        num_train = round(num_samples * self.train_rate)
        if self.load_external:
            xc, xt, x_ext = x
            # train
            xc_train, xt_train, x_ext_train, y_train = xc[:num_train], xt[:num_train], x_ext[:num_train], y[:num_train]
            # val
            xc_val, xt_val, x_ext_val, y_val = xc[num_train:], xt[num_train:], x_ext[num_train:], y[num_train:]
            # test
            xc_test, xt_test, x_ext_test = x_test

            self._logger.info("train\t" + "xc: " + str(xc_train.shape) + ", xt: " + str(xt_train.shape) +
                              ", x_ext: " + str(x_ext_train.shape) + ", y: " + str(y_train.shape))
            self._logger.info("eval\t" + "xc: " + str(xc_val.shape) + ", xt: " + str(xt_val.shape) +
                              ", x_ext: " + str(x_ext_val.shape) + ", y: " + str(y_val.shape))
            self._logger.info("test\t" + "xc: " + str(xc_test.shape) + ", xt: " + str(xt_test.shape) +
                              ", x_ext: " + str(x_ext_test.shape) + ", y: " + str(y_test.shape))
        else:
            xc, xt = x
            # train
            xc_train, xt_train, x_ext_train, y_train = xc[:num_train], xt[:num_train], None, y[:num_train]
            # val
            xc_val, xt_val, x_ext_val, y_val = xc[num_train:], xt[num_train:], None, y[num_train:]
            # test
            xc_test, xt_test = x_test
            x_ext_test = None

            self._logger.info("train\t" + "xc: " + str(xc_train.shape) + ", xt: " + str(xt_train.shape) +
                              ", y: " + str(y_train.shape))
            self._logger.info("eval\t" + "xc: " + str(xc_val.shape) + ", xt: " + str(xt_val.shape) +
                              ", y: " + str(y_val.shape))
            self._logger.info("test\t" + "xc: " + str(xc_test.shape) + ", xt: " + str(xt_test.shape) +
                              ", y: " + str(y_test.shape))

        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                self.cache_file_name,
                xc_train=xc_train,
                xt_train=xt_train,
                x_ext_train=x_ext_train,
                y_train=y_train,
                xc_val=xc_val,
                xt_val=xt_val,
                x_ext_val=x_ext_val,
                y_val=y_val,
                xc_test=xc_test,
                xt_test=xt_test,
                x_ext_test=x_ext_test,
                y_test=y_test,
            )
            self._logger.info('Saved at ' + self.cache_file_name)
        return xc_train, xt_train, x_ext_train, y_train, xc_val, xt_val, x_ext_val, y_val, xc_test, xt_test, \
            x_ext_test, y_test

    def _generate_train_val_test(self):
        # 获取数据
        x_train, y_train, x_test, y_test, mmn, metadata_dim, timestamp_train, timestamp_test = self._generate_data()
        # 划分数据
        return self.split_train_val_test(x_train, y_train, x_test, y_test)

    def _load_cache_train_val_test(self):
        self._logger.info('Loading ' + self.cache_file_name)
        cache_data = np.load(self.cache_file_name, allow_pickle=True)
        xc_train = cache_data['xc_train']
        xt_train = cache_data['xt_train']
        x_ext_train = cache_data['x_ext_train']
        y_train = cache_data['y_train']
        xc_val = cache_data['xc_val']
        xt_val = cache_data['xt_val']
        x_ext_val = cache_data['x_ext_val']
        y_val = cache_data['y_val']
        xc_test = cache_data['xc_test']
        xt_test = cache_data['xt_test']
        x_ext_test = cache_data['x_ext_test']
        y_test = cache_data['y_test']
        if self.load_external:
            self._logger.info("train\t" + "xc: " + str(xc_train.shape) + ", xt: " + str(xt_train.shape) +
                              ", x_ext: " + str(x_ext_train.shape) + ", y: " + str(y_train.shape))
            self._logger.info("eval\t" + "xc: " + str(xc_val.shape) + ", xt: " + str(xt_val.shape) +
                              ", x_ext: " + str(x_ext_val.shape) + ", y: " + str(y_val.shape))
            self._logger.info("test\t" + "xc: " + str(xc_test.shape) + ", xt: " + str(xt_test.shape) +
                              ", x_ext: " + str(x_ext_test.shape) + ", y: " + str(y_test.shape))
        else:
            self._logger.info("train\t" + "xc: " + str(xc_train.shape) + ", xt: " + str(xt_train.shape) +
                              ", y: " + str(y_train.shape))
            self._logger.info("eval\t" + "xc: " + str(xc_val.shape) + ", xt: " + str(xt_val.shape) +
                              ", y: " + str(y_val.shape))
            self._logger.info("test\t" + "xc: " + str(xc_test.shape) + ", xt: " + str(xt_test.shape) +
                              ", y: " + str(y_test.shape))
        return xc_train, xt_train, x_ext_train, y_train, xc_val, xt_val, x_ext_val, y_val, xc_test, xt_test, \
            x_ext_test, y_test

    def get_data(self):
        # 加载数据集
        xc_train, xt_train, x_ext_train, y_train, xc_val, xt_val, x_ext_val, y_val, xc_test, xt_test, \
            x_ext_test, y_test = [], [], [], [], [], [], [], [], [], [], [], []
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                xc_train, xt_train, x_ext_train, y_train, xc_val, xt_val, x_ext_val, y_val, xc_test, xt_test, \
                    x_ext_test, y_test = self._load_cache_train_val_test()
            else:
                xc_train, xt_train, x_ext_train, y_train, xc_val, xt_val, x_ext_val, y_val, xc_test, xt_test, \
                    x_ext_test, y_test = self._generate_train_val_test()
        if self.load_external:
            train_data = list(zip(xc_train, xt_train, x_ext_train, y_train))
            eval_data = list(zip(xc_val, xt_val, x_ext_val, y_val))
            test_data = list(zip(xc_test, xt_test, x_ext_test, y_test))
        else:
            train_data = list(zip(xc_train, xt_train, y_train))
            eval_data = list(zip(xc_val, xt_val, y_val))
            test_data = list(zip(xc_test, xt_test, y_test))
        # 转Dataloader
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample)
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "ext_dim": self.ext_dim,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches,
                "time_intervals": self.time_intervals, "load_external": self.load_external}
