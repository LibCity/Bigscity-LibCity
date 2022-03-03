import numpy as np
import os
import pandas as pd

from libcity.data.dataset import TrafficStateDataset
from libcity.data.utils import generate_dataloader
from libcity.utils import StandardScaler, NormalScaler, NoneScaler, MinMax01Scaler, MinMax11Scaler, ensure_dir


class STDNDataset(TrafficStateDataset):

    def __init__(self, config):
        super().__init__(config)
        # lstm_seq_len = input_window
        self.input_window = self.config.get('input_window', 7)
        self.output_window = self.config.get('output_window', 1)
        self.att_lstm_num = self.config.get('att_lstm_num', 3)
        self.att_lstm_seq_len = self.config.get('att_lstm_seq_len', 3)
        self.hist_feature_daynum = self.config.get('hist_feature_daynum', 7)
        self.last_feature_num = self.config.get('last_feature_num', 48)
        self.points_per_hour = 3600 // self.time_intervals
        self.timeslot_daynum = self.points_per_hour * 24
        self.cnn_nbhd_size = self.config.get('cnn_nbhd_size', 3)
        self.nbhd_size = self.config.get('nbhd_size', 2)
        self.scaler = None
        self.flow_scaler = None
        self.feature_name = {'X': 'float', 'y': 'float', 'flatten_att_nbhd_inputs': 'float',
                             'flatten_att_flow_inputs': 'float', 'att_lstm_inputs': 'float', 'nbhd_inputs': 'float',
                             'flow_inputs': 'float', 'lstm_inputs': 'float'}
        self.batch_size = self.config.get('batch_size', 1)

    def _load_geo(self):
        super()._load_grid_geo()

    def _load_rel(self):
        super()._load_grid_rel()

    def _load_grid(self, filename):
        return super()._load_grid_4d(filename)

    def _load_gridod(self, filename):
        gridodfile = pd.read_csv(self.data_path + filename + '.gridod')
        # if self.data_col != '':  # 根据指定的列加载数据集
        #     if isinstance(self.data_col, list):
        #         data_col = self.data_col.copy()
        #     else:  # str
        #         data_col = [self.data_col.copy()]
        #     data_col.insert(0, 'time')
        #     data_col.insert(1, 'origin_row_id')
        #     data_col.insert(2, 'origin_column_id')
        #     data_col.insert(3, 'destination_row_id')
        #     data_col.insert(4, 'destination_column_id')
        #     gridodfile = gridodfile[data_col]
        # else:  # 不指定则加载所有列
        #     gridodfile = gridodfile[gridodfile.columns[2:]]  # 从time列开始所有列
        gridodfile = gridodfile[gridodfile.columns[2:]]
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

    def _sample_stdn(self, volume_df, flow_df):
        cnn_att_features = []
        lstm_att_features = []
        flow_att_features = []
        for i in range(self.att_lstm_num):
            cnn_att_features.append([])
            lstm_att_features.append([])
            flow_att_features.append([])
            for j in range(self.att_lstm_seq_len):
                cnn_att_features[i].append([])
                flow_att_features[i].append([])

        # cnn_features是一个长度为lstm_seq_len的列表，
        # cnn_features[i]也是一个列表，排列顺序为时间t, 横坐标x, 纵坐标y的顺序
        # 记录了t - (lstm_seq_len - i)时间以(x,y)为中心的7 * 7区域的volume数据
        cnn_features = []
        # flow_features是一个长度为lstm_seq_len的列表，
        # flow_features[i]也是一个列表，排列顺序为时间t, 横坐标x, 纵坐标y的顺序
        # 记录了t - (lstm_seq_len - i)时间以(x,y)为中心的7 * 7区域的flow数据
        flow_features = []
        for i in range(self.input_window):
            cnn_features.append([])
            flow_features.append([])

        time_start = (self.hist_feature_daynum + self.att_lstm_num) * self.timeslot_daynum + self.att_lstm_seq_len
        time_end = volume_df.shape[0]
        volume_type = volume_df.shape[-1]

        short_term_lstm_features = []
        for t in range(time_start, time_end):
            for x in range(self.len_row):
                for y in range(self.len_column):
                    short_term_lstm_samples = []
                    for seqn in range(self.input_window):
                        real_t = t - (self.input_window - seqn)

                        # cnn_feature表示在real_t时间以(x, y)为中心的7 * 7区域的volume数据
                        cnn_feature = np.zeros((2 * self.cnn_nbhd_size + 1, 2 * self.cnn_nbhd_size + 1, volume_type))
                        for cnn_nbhd_x in range(x - self.cnn_nbhd_size, x + self.cnn_nbhd_size + 1):
                            for cnn_nbhd_y in range(y - self.cnn_nbhd_size, y + self.cnn_nbhd_size + 1):
                                if not (0 <= cnn_nbhd_x < self.len_row and 0 <= cnn_nbhd_y < self.len_column):
                                    continue
                                cnn_feature[cnn_nbhd_x - (x - self.cnn_nbhd_size),
                                cnn_nbhd_y - (y - self.cnn_nbhd_size), :] = volume_df[real_t, cnn_nbhd_x, cnn_nbhd_y, :]
                        cnn_features[seqn].append(cnn_feature)

                        flow_feature_curr_out = flow_df[real_t, x, y, :, :, 0]
                        flow_feature_curr_in = flow_df[real_t, :, :, x, y, 0]
                        flow_feature_last_out_to_curr = flow_df[real_t - 1, x, y, :, :, 1]
                        flow_feature_curr_in_from_last = flow_df[real_t - 1, :, :, x, y, 1]
                        flow_feature = np.zeros(flow_feature_curr_in.shape + (4,))
                        flow_feature[:, :, 0] = flow_feature_curr_out
                        flow_feature[:, :, 1] = flow_feature_curr_in
                        flow_feature[:, :, 2] = flow_feature_last_out_to_curr
                        flow_feature[:, :, 3] = flow_feature_curr_in_from_last
                        # local_flow_feature表示在real_t时间以(x, y)为中心的7 * 7区域的flow数据
                        local_flow_feature = np.zeros((2 * self.cnn_nbhd_size + 1, 2 * self.cnn_nbhd_size + 1, 4))
                        for cnn_nbhd_x in range(x - self.cnn_nbhd_size, x + self.cnn_nbhd_size + 1):
                            for cnn_nbhd_y in range(y - self.cnn_nbhd_size, y + self.cnn_nbhd_size + 1):
                                if not (0 <= cnn_nbhd_x < self.len_row and 0 <= cnn_nbhd_y < self.len_column):
                                    continue
                                local_flow_feature[cnn_nbhd_x - (x - self.cnn_nbhd_size),
                                cnn_nbhd_y - (y - self.cnn_nbhd_size), :] = flow_feature[cnn_nbhd_x, cnn_nbhd_y, :]
                        flow_features[seqn].append(local_flow_feature)

                        hist_feature = volume_df[
                                       real_t - self.hist_feature_daynum * self.timeslot_daynum: real_t: self.timeslot_daynum,
                                       x, y, :].flatten()
                        last_feature = volume_df[real_t - self.last_feature_num: real_t, x, y, :].flatten()
                        nbhd_feature = np.zeros((2 * self.nbhd_size + 1, 2 * self.nbhd_size + 1, volume_type))
                        for nbhd_x in range(x - self.nbhd_size, x + self.nbhd_size + 1):
                            for nbhd_y in range(y - self.nbhd_size, y + self.nbhd_size + 1):
                                if not (0 <= nbhd_x < self.len_row and 0 <= nbhd_y < self.len_column):
                                    continue
                                nbhd_feature[nbhd_x - (x - self.nbhd_size), nbhd_y - (y - self.nbhd_size),
                                :] = volume_df[real_t, nbhd_x, nbhd_y, :]
                        nbhd_feature = nbhd_feature.flatten()
                        feature_vec = np.concatenate((hist_feature, last_feature))
                        feature_vec = np.concatenate((feature_vec, nbhd_feature))
                        short_term_lstm_samples.append(feature_vec)
                    short_term_lstm_features.append(np.array(short_term_lstm_samples))

                    for att_lstm_cnt in range(self.att_lstm_num):
                        long_term_lstm_samples = []
                        att_t = t - (self.att_lstm_num - att_lstm_cnt) * self.timeslot_daynum + (
                                self.att_lstm_seq_len - 1) / 2 + 1
                        att_t = int(att_t)
                        for seqn in range(self.att_lstm_seq_len):
                            real_t = att_t - (self.att_lstm_seq_len - seqn)

                            cnn_feature = np.zeros(
                                (2 * self.cnn_nbhd_size + 1, 2 * self.cnn_nbhd_size + 1, volume_type))
                            for cnn_nbhd_x in range(x - self.cnn_nbhd_size, x + self.cnn_nbhd_size + 1):
                                for cnn_nbhd_y in range(y - self.cnn_nbhd_size, y + self.cnn_nbhd_size + 1):
                                    if not (0 <= cnn_nbhd_x < self.len_row and 0 <= cnn_nbhd_y < self.len_column):
                                        continue
                                    cnn_feature[cnn_nbhd_x - (x - self.cnn_nbhd_size),
                                    cnn_nbhd_y - (y - self.cnn_nbhd_size), :] = volume_df[real_t, cnn_nbhd_x,
                                                                                cnn_nbhd_y, :]
                            cnn_att_features[att_lstm_cnt][seqn].append(cnn_feature)

                            flow_feature_curr_out = flow_df[real_t, x, y, :, :, 0]
                            flow_feature_curr_in = flow_df[real_t, :, :, x, y, 0]
                            flow_feature_last_out_to_curr = flow_df[real_t - 1, x, y, :, :, 1]
                            flow_feature_curr_in_from_last = flow_df[real_t - 1, :, :, x, y, 1]
                            flow_feature = np.zeros(flow_feature_curr_in.shape + (4,))
                            flow_feature[:, :, 0] = flow_feature_curr_out
                            flow_feature[:, :, 1] = flow_feature_curr_in
                            flow_feature[:, :, 2] = flow_feature_last_out_to_curr
                            flow_feature[:, :, 3] = flow_feature_curr_in_from_last
                            local_flow_feature = np.zeros((2 * self.cnn_nbhd_size + 1, 2 * self.cnn_nbhd_size + 1, 4))
                            for cnn_nbhd_x in range(x - self.cnn_nbhd_size, x + self.cnn_nbhd_size + 1):
                                for cnn_nbhd_y in range(y - self.cnn_nbhd_size, y + self.cnn_nbhd_size + 1):
                                    if not (0 <= cnn_nbhd_x < self.len_row and 0 <= cnn_nbhd_y < self.len_column):
                                        continue
                                    local_flow_feature[cnn_nbhd_x - (x - self.cnn_nbhd_size),
                                    cnn_nbhd_y - (y - self.cnn_nbhd_size), :] = flow_feature[cnn_nbhd_x, cnn_nbhd_y, :]
                            flow_att_features[att_lstm_cnt][seqn].append(local_flow_feature)

                            hist_feature = volume_df[
                                           real_t - self.hist_feature_daynum * self.timeslot_daynum: real_t: self.timeslot_daynum,
                                           x, y, :].flatten()
                            last_feature = volume_df[real_t - self.last_feature_num: real_t, x, y, :].flatten()
                            nbhd_feature = np.zeros((2 * self.nbhd_size + 1, 2 * self.nbhd_size + 1, volume_type))
                            for nbhd_x in range(x - self.nbhd_size, x + self.nbhd_size + 1):
                                for nbhd_y in range(y - self.nbhd_size, y + self.nbhd_size + 1):
                                    if not (0 <= nbhd_x < self.len_row and 0 <= nbhd_y < self.len_column):
                                        continue
                                    nbhd_feature[nbhd_x - (x - self.nbhd_size), nbhd_y - (y - self.nbhd_size),
                                    :] = volume_df[real_t, nbhd_x, nbhd_y, :]
                            nbhd_feature = nbhd_feature.flatten()
                            feature_vec = np.concatenate((hist_feature, last_feature))
                            feature_vec = np.concatenate((feature_vec, nbhd_feature))
                            long_term_lstm_samples.append(feature_vec)
                        lstm_att_features[att_lstm_cnt].append(np.array(long_term_lstm_samples))

        output_cnn_att_features = []
        output_flow_att_features = []
        for i in range(self.att_lstm_num):
            lstm_att_features[i] = np.array(lstm_att_features[i])
            for j in range(self.att_lstm_seq_len):
                cnn_att_features[i][j] = np.array(cnn_att_features[i][j])
                flow_att_features[i][j] = np.array(flow_att_features[i][j])
                output_cnn_att_features.append(cnn_att_features[i][j])
                output_flow_att_features.append(flow_att_features[i][j])
        output_cnn_att_features = np.stack(output_cnn_att_features, axis=0)
        output_cnn_att_features = np.swapaxes(output_cnn_att_features, 0, 1)
        output_cnn_att_features = np.reshape(output_cnn_att_features,
                                             (-1, self.len_row, self.len_column, *output_cnn_att_features.shape[1:]))
        output_flow_att_features = np.stack(output_flow_att_features, axis=0)
        output_flow_att_features = np.swapaxes(output_flow_att_features, 0, 1)
        output_flow_att_features = np.reshape(output_flow_att_features,
                                              (-1, self.len_row, self.len_column, *output_flow_att_features.shape[1:]))
        lstm_att_features = np.stack(lstm_att_features, axis=0)
        lstm_att_features = np.swapaxes(lstm_att_features, 0, 1)
        lstm_att_features = np.reshape(lstm_att_features,
                                       (-1, self.len_row, self.len_column, *lstm_att_features.shape[1:]))
        for i in range(self.input_window):
            cnn_features[i] = np.array(cnn_features[i])
            flow_features[i] = np.array(flow_features[i])
        cnn_features = np.stack(cnn_features, axis=0)
        cnn_features = np.swapaxes(cnn_features, 0, 1)
        cnn_features = np.reshape(cnn_features,
                                  (-1, self.len_row, self.len_column, *cnn_features.shape[1:]))
        flow_features = np.stack(flow_features, axis=0)
        flow_features = np.swapaxes(flow_features, 0, 1)
        flow_features = np.reshape(flow_features,
                                   (-1, self.len_row, self.len_column, *flow_features.shape[1:]))
        short_term_lstm_features = np.array(short_term_lstm_features)
        short_term_lstm_features = np.reshape(short_term_lstm_features,
                                              (-1, self.len_row, self.len_column, *short_term_lstm_features.shape[1:]))
        return output_cnn_att_features, output_flow_att_features, lstm_att_features, cnn_features, flow_features, short_term_lstm_features  # , inputs, labels

    def _generate_input_data_stdn(self, volume_df, flow_df):
        flatten_att_nbhd_input, flatten_att_flow_input, att_lstm_input, nbhd_input, flow_input, lstm_input = self._sample_stdn(
            volume_df, flow_df)
        num_samples = lstm_input.shape[0]
        x_offsets = np.sort(np.concatenate((np.arange(-self.input_window + 1, 1, 1),)))
        y_offsets = np.sort(np.arange(1, self.output_window + 1, 1))

        flatten_att_nbhd_inputs = []
        flatten_att_flow_inputs = []
        att_lstm_inputs = []
        nbhd_inputs = []
        flow_inputs = []
        lstm_inputs = []
        x = []
        y = []

        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))
        for t in range(min_t, max_t):
            flatten_att_nbhd_inputs_t = flatten_att_nbhd_input[t + y_offsets, ...]
            flatten_att_flow_inputs_t = flatten_att_flow_input[t + y_offsets, ...]
            att_lstm_inputs_t = att_lstm_input[t + y_offsets, ...]
            nbhd_inputs_t = nbhd_input[t + y_offsets, ...]
            flow_inputs_t = flow_input[t + y_offsets, ...]
            lstm_inputs_t = lstm_input[t + y_offsets, ...]
            x_t = volume_df[t + x_offsets, ...]
            y_t = volume_df[t + y_offsets, ...]

            flatten_att_nbhd_inputs.append(flatten_att_nbhd_inputs_t)
            flatten_att_flow_inputs.append(flatten_att_flow_inputs_t)
            att_lstm_inputs.append(att_lstm_inputs_t)
            nbhd_inputs.append(nbhd_inputs_t)
            flow_inputs.append(flow_inputs_t)
            lstm_inputs.append(lstm_inputs_t)
            x.append(x_t)
            y.append(y_t)
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        flatten_att_nbhd_inputs = np.stack(flatten_att_nbhd_inputs, axis=0)
        flatten_att_flow_inputs = np.stack(flatten_att_flow_inputs, axis=0)
        att_lstm_inputs = np.stack(att_lstm_inputs, axis=0)
        nbhd_inputs = np.stack(nbhd_inputs, axis=0)
        flow_inputs = np.stack(flow_inputs, axis=0)
        lstm_inputs = np.stack(lstm_inputs, axis=0)
        return x, y, flatten_att_nbhd_inputs, flatten_att_flow_inputs, att_lstm_inputs, nbhd_inputs, flow_inputs, lstm_inputs

    def _generate_data(self):
        volume_df = self._load_grid(self.data_files[0])
        flow_df = self._load_gridod(self.data_files[0])

        x, y, flatten_att_nbhd_inputs, flatten_att_flow_inputs, att_lstm_inputs, nbhd_inputs, flow_inputs, lstm_inputs = self._generate_input_data_stdn(
            volume_df, flow_df)

        x = np.concatenate([x])
        y = np.concatenate([y])
        flatten_att_nbhd_inputs = np.concatenate([flatten_att_nbhd_inputs])
        flatten_att_flow_inputs = np.concatenate([flatten_att_flow_inputs])
        att_lstm_inputs = np.concatenate([att_lstm_inputs])
        nbhd_inputs = np.concatenate([nbhd_inputs])
        flow_inputs = np.concatenate([flow_inputs])
        lstm_inputs = np.concatenate([lstm_inputs])

        self._logger.info("Dataset created")
        self._logger.info(
            "x shape: " + str(x.shape) + ", y shape: " + str(y.shape) + ", flatten_att_nbhd_inputs shape: " + str(
                flatten_att_nbhd_inputs.shape) + ", flatten_att_flow_inputs shape: " + str(
                flatten_att_flow_inputs.shape) + ", att_lstm_inputs shape: " + str(
                att_lstm_inputs.shape) + ", nbhd_inputs shape: " + str(
                nbhd_inputs.shape) + ", flow_inputs shape: " + str(flow_inputs.shape) + ", lstm_inputs shape: " + str(
                lstm_inputs.shape))
        return x, y, flatten_att_nbhd_inputs, flatten_att_flow_inputs, att_lstm_inputs, nbhd_inputs, flow_inputs, lstm_inputs

    def _split_train_val_test_stdn(self, x, y, flatten_att_nbhd_inputs, flatten_att_flow_inputs, att_lstm_inputs,
                                   nbhd_inputs, flow_inputs, lstm_inputs):
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
        x_train = x[:num_train]
        y_train = y[:num_train]
        flatten_att_nbhd_inputs_train = flatten_att_nbhd_inputs[:num_train]
        flatten_att_flow_inputs_train = flatten_att_flow_inputs[:num_train]
        att_lstm_inputs_train = att_lstm_inputs[:num_train]
        nbhd_inputs_train = nbhd_inputs[:num_train]
        flow_inputs_train = flow_inputs[:num_train]
        lstm_inputs_train = lstm_inputs[:num_train]
        # val
        x_val = x[num_train: num_train + num_val]
        y_val = y[num_train: num_train + num_val]
        flatten_att_nbhd_inputs_val = flatten_att_nbhd_inputs[num_train: num_train + num_val]
        flatten_att_flow_inputs_val = flatten_att_flow_inputs[num_train: num_train + num_val]
        att_lstm_inputs_val = att_lstm_inputs[num_train: num_train + num_val]
        nbhd_inputs_val = nbhd_inputs[num_train: num_train + num_val]
        flow_inputs_val = flow_inputs[num_train: num_train + num_val]
        lstm_inputs_val = lstm_inputs[num_train: num_train + num_val]
        # test
        x_test = x[-num_test:]
        y_test = y[-num_test:]
        flatten_att_nbhd_inputs_test = flatten_att_nbhd_inputs[-num_test:]
        flatten_att_flow_inputs_test = flatten_att_flow_inputs[-num_test:]
        att_lstm_inputs_test = att_lstm_inputs[-num_test:]
        nbhd_inputs_test = nbhd_inputs[-num_test:]
        flow_inputs_test = flow_inputs[-num_test:]
        lstm_inputs_test = lstm_inputs[-num_test:]
        self._logger.info(
            "train\t" + "x: " + str(x_train.shape) + "y: " + str(y_train.shape) + "flatten_att_nbhd_inputs: " + str(
                flatten_att_nbhd_inputs_train.shape) + "flatten_att_flow_inputs: " + str(
                flatten_att_flow_inputs_train.shape) + "att_lstm_inputs: " + str(
                att_lstm_inputs_train.shape) + "nbhd_inputs: " + str(nbhd_inputs_train.shape) + "flow_inputs: " + str(
                flow_inputs_train.shape) + "lstm_inputs: " + str(lstm_inputs_train.shape))
        self._logger.info(
            "eval\t" + "x: " + str(x_val.shape) + "y: " + str(y_val.shape) + "flatten_att_nbhd_inputs: " + str(
                flatten_att_nbhd_inputs_val.shape) + "flatten_att_flow_inputs: " + str(
                flatten_att_flow_inputs_val.shape) + "att_lstm_inputs: " + str(
                att_lstm_inputs_val.shape) + "nbhd_inputs: " + str(nbhd_inputs_val.shape) + "flow_inputs: " + str(
                flow_inputs_val.shape) + "lstm_inputs: " + str(lstm_inputs_val.shape))
        self._logger.info(
            "test\t" + "x: " + str(x_test.shape) + "y: " + str(y_test.shape) + "flatten_att_nbhd_inputs: " + str(
                flatten_att_nbhd_inputs_test.shape) + "flatten_att_flow_inputs: " + str(
                flatten_att_flow_inputs_test.shape) + "att_lstm_inputs: " + str(
                att_lstm_inputs_test.shape) + "nbhd_inputs: " + str(nbhd_inputs_test.shape) + "flow_inputs: " + str(
                flow_inputs_test.shape) + "lstm_inputs: " + str(lstm_inputs_test.shape))

        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                self.cache_file_name,
                x_train=x_train,
                y_train=y_train,
                flatten_att_nbhd_inputs_train=flatten_att_nbhd_inputs_train,
                flatten_att_flow_inputs_train=flatten_att_flow_inputs_train,
                att_lstm_inputs_train=att_lstm_inputs_train,
                nbhd_inputs_train=nbhd_inputs_train,
                flow_inputs_train=flow_inputs_train,
                lstm_inputs_train=lstm_inputs_train,
                x_test=x_test,
                y_test=y_test,
                flatten_att_nbhd_inputs_test=flatten_att_nbhd_inputs_test,
                flatten_att_flow_inputs_test=flatten_att_flow_inputs_test,
                att_lstm_inputs_test=att_lstm_inputs_test,
                nbhd_inputs_test=nbhd_inputs_test,
                flow_inputs_test=flow_inputs_test,
                lstm_inputs_test=lstm_inputs_test,
                x_val=x_val,
                y_val=y_val,
                flatten_att_nbhd_inputs_val=flatten_att_nbhd_inputs_val,
                flatten_att_flow_inputs_val=flatten_att_flow_inputs_val,
                att_lstm_inputs_val=att_lstm_inputs_val,
                nbhd_inputs_val=nbhd_inputs_val,
                flow_inputs_val=flow_inputs_val,
                lstm_inputs_val=lstm_inputs_val,
            )
            self._logger.info('Saved at ' + self.cache_file_name)
        return x_train, y_train, flatten_att_nbhd_inputs_train, flatten_att_flow_inputs_train, att_lstm_inputs_train, nbhd_inputs_train, flow_inputs_train, lstm_inputs_train, \
               x_val, y_val, flatten_att_nbhd_inputs_val, flatten_att_flow_inputs_val, att_lstm_inputs_val, nbhd_inputs_val, flow_inputs_val, lstm_inputs_val, \
               x_test, y_test, flatten_att_nbhd_inputs_test, flatten_att_flow_inputs_test, att_lstm_inputs_test, nbhd_inputs_test, flow_inputs_test, lstm_inputs_test

    def _generate_train_val_test(self):
        """
        加载数据集，并划分训练集、测试集、验证集，并缓存数据集
        """
        x, y, flatten_att_nbhd_inputs, flatten_att_flow_inputs, att_lstm_inputs, nbhd_inputs, flow_inputs, lstm_inputs = self._generate_data()
        return self._split_train_val_test_stdn(x, y, flatten_att_nbhd_inputs, flatten_att_flow_inputs, att_lstm_inputs,
                                               nbhd_inputs, flow_inputs, lstm_inputs)

    def _load_cache_train_val_test(self):
        """
        加载之前缓存好的训练集、测试集、验证集
        """
        self._logger.info('Loading ' + self.cache_file_name)
        cat_data = np.load(self.cache_file_name)
        x_train = cat_data['x_train']
        y_train = cat_data['y_train']
        flatten_att_nbhd_inputs_train = cat_data['flatten_att_nbhd_inputs_train']
        flatten_att_flow_inputs_train = cat_data['flatten_att_flow_inputs_train']
        att_lstm_inputs_train = cat_data['att_lstm_inputs_train']
        nbhd_inputs_train = cat_data['nbhd_inputs_train']
        flow_inputs_train = cat_data['flow_inputs_train']
        lstm_inputs_train = cat_data['lstm_inputs_train']
        x_test = cat_data['x_test']
        y_test = cat_data['y_test']
        flatten_att_nbhd_inputs_test = cat_data['flatten_att_nbhd_inputs_test']
        flatten_att_flow_inputs_test = cat_data['flatten_att_flow_inputs_test']
        att_lstm_inputs_test = cat_data['att_lstm_inputs_test']
        nbhd_inputs_test = cat_data['nbhd_inputs_test']
        flow_inputs_test = cat_data['flow_inputs_test']
        lstm_inputs_test = cat_data['lstm_inputs_test']
        x_val = cat_data['x_val']
        y_val = cat_data['y_val']
        flatten_att_nbhd_inputs_val = cat_data['flatten_att_nbhd_inputs_val']
        flatten_att_flow_inputs_val = cat_data['flatten_att_flow_inputs_val']
        att_lstm_inputs_val = cat_data['att_lstm_inputs_val']
        nbhd_inputs_val = cat_data['nbhd_inputs_val']
        flow_inputs_val = cat_data['flow_inputs_val']
        lstm_inputs_val = cat_data['lstm_inputs_val']
        self._logger.info(
            "train\t" + "x: " + str(x_train.shape) + "y: " + str(y_train.shape) + "flatten_att_nbhd_inputs: " + str(
                flatten_att_nbhd_inputs_train.shape) + "flatten_att_flow_inputs: " + str(
                flatten_att_flow_inputs_train.shape) + "att_lstm_inputs: " + str(
                att_lstm_inputs_train.shape) + "nbhd_inputs: " + str(nbhd_inputs_train.shape) + "flow_inputs: " + str(
                flow_inputs_train.shape) + "lstm_inputs: " + str(lstm_inputs_train.shape))
        self._logger.info(
            "eval\t" + "x: " + str(x_val.shape) + "y: " + str(y_val.shape) + "flatten_att_nbhd_inputs: " + str(
                flatten_att_nbhd_inputs_val.shape) + "flatten_att_flow_inputs: " + str(
                flatten_att_flow_inputs_val.shape) + "att_lstm_inputs: " + str(
                att_lstm_inputs_val.shape) + "nbhd_inputs: " + str(nbhd_inputs_val.shape) + "flow_inputs: " + str(
                flow_inputs_val.shape) + "lstm_inputs: " + str(lstm_inputs_val.shape))
        self._logger.info(
            "test\t" + "x: " + str(x_test.shape) + "y: " + str(y_test.shape) + "flatten_att_nbhd_inputs: " + str(
                flatten_att_nbhd_inputs_test.shape) + "flatten_att_flow_inputs: " + str(
                flatten_att_flow_inputs_test.shape) + "att_lstm_inputs: " + str(
                att_lstm_inputs_test.shape) + "nbhd_inputs: " + str(nbhd_inputs_test.shape) + "flow_inputs: " + str(
                flow_inputs_test.shape) + "lstm_inputs: " + str(lstm_inputs_test.shape))
        return x_train, y_train, flatten_att_nbhd_inputs_train, flatten_att_flow_inputs_train, att_lstm_inputs_train, nbhd_inputs_train, flow_inputs_train, lstm_inputs_train, \
               x_val, y_val, flatten_att_nbhd_inputs_val, flatten_att_flow_inputs_val, att_lstm_inputs_val, nbhd_inputs_val, flow_inputs_val, lstm_inputs_val, \
               x_test, y_test, flatten_att_nbhd_inputs_test, flatten_att_flow_inputs_test, att_lstm_inputs_test, nbhd_inputs_test, flow_inputs_test, lstm_inputs_test

    def _get_scalar_stdn(self, x_train, y_train, flow_inputs_train):
        if self.scaler_type == "normal":
            volume_scaler = NormalScaler(maxx=max(x_train.max(), y_train.max()))
            flow_scaler = NormalScaler(maxx=flow_inputs_train.max())
            self._logger.info(
                'NormalScaler volume max: ' + str(volume_scaler.max) + ' flow max: ' + str(flow_scaler.max))
        elif self.scaler_type == "standard":
            volume_scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
            flow_scaler = StandardScaler(mean=flow_inputs_train.mean(), std=flow_inputs_train.std())
            self._logger.info('StandardScaler volume mean: ' + str(volume_scaler.mean) + ', volume std: ' + str(
                volume_scaler.std) + ', flow mean: ' + str(flow_scaler.mean) + ', flow std: ' + str(flow_scaler.std))
        elif self.scaler_type == "minmax01":
            volume_scaler = MinMax01Scaler(maxx=max(x_train.max(), y_train.max()),
                                           minn=min(x_train.min(), y_train.min()))
            flow_scaler = MinMax01Scaler(maxx=flow_inputs_train.max(), minn=flow_inputs_train.min())
            self._logger.info('MinMax01Scaler volume max: ' + str(volume_scaler.max) + ', volume min: ' + str(
                volume_scaler.min) + ', flow max: ' + str(flow_scaler.max) + ', flow min: ' + str(flow_scaler.min))
        elif self.scaler_type == "minmax11":
            volume_scaler = MinMax11Scaler(maxx=max(x_train.max(), y_train.max()),
                                           minn=min(x_train.min(), y_train.min()))
            flow_scaler = MinMax11Scaler(maxx=flow_inputs_train.max(), minn=flow_inputs_train.min())
            self._logger.info('MinMax11Scaler volume max: ' + str(volume_scaler.max) + ', volume min: ' + str(
                volume_scaler.min) + ', flow max: ' + str(flow_scaler.max) + ', flow min: ' + str(flow_scaler.min))
        elif self.scaler_type == "none":
            volume_scaler = NoneScaler()
            flow_scaler = NoneScaler()
            self._logger.info('NoneScaler')
        else:
            raise ValueError('Scaler type error!')
        return volume_scaler, flow_scaler

    def get_data(self):
        x_train, y_train, flatten_att_nbhd_inputs_train, flatten_att_flow_inputs_train, att_lstm_inputs_train, nbhd_inputs_train, flow_inputs_train, lstm_inputs_train = [], [], [], [], [], [], [], []
        x_val, y_val, flatten_att_nbhd_inputs_val, flatten_att_flow_inputs_val, att_lstm_inputs_val, nbhd_inputs_val, flow_inputs_val, lstm_inputs_val = [], [], [], [], [], [], [], []
        x_test, y_test, flatten_att_nbhd_inputs_test, flatten_att_flow_inputs_test, att_lstm_inputs_test, nbhd_inputs_test, flow_inputs_test, lstm_inputs_test = [], [], [], [], [], [], [], []
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train, y_train, flatten_att_nbhd_inputs_train, flatten_att_flow_inputs_train, att_lstm_inputs_train, nbhd_inputs_train, flow_inputs_train, lstm_inputs_train, \
                x_val, y_val, flatten_att_nbhd_inputs_val, flatten_att_flow_inputs_val, att_lstm_inputs_val, nbhd_inputs_val, flow_inputs_val, lstm_inputs_val, \
                x_test, y_test, flatten_att_nbhd_inputs_test, flatten_att_flow_inputs_test, att_lstm_inputs_test, nbhd_inputs_test, flow_inputs_test, lstm_inputs_test = self._load_cache_train_val_test()
            else:
                x_train, y_train, flatten_att_nbhd_inputs_train, flatten_att_flow_inputs_train, att_lstm_inputs_train, nbhd_inputs_train, flow_inputs_train, lstm_inputs_train, \
                x_val, y_val, flatten_att_nbhd_inputs_val, flatten_att_flow_inputs_val, att_lstm_inputs_val, nbhd_inputs_val, flow_inputs_val, lstm_inputs_val, \
                x_test, y_test, flatten_att_nbhd_inputs_test, flatten_att_flow_inputs_test, att_lstm_inputs_test, nbhd_inputs_test, flow_inputs_test, lstm_inputs_test = self._generate_train_val_test()
        self.feature_dim = x_train.shape[-1]
        self.feature_vec_len = lstm_inputs_train.shape[-1]
        self.nbhd_type = nbhd_inputs_train.shape[-1]
        self.scaler, self.flow_scaler = self._get_scalar_stdn(x_train, y_train, flow_inputs_train)
        x_train = self.scaler.transform(x_train)
        y_train = self.scaler.transform(y_train)
        flatten_att_nbhd_inputs_train = self.scaler.transform(flatten_att_nbhd_inputs_train)
        att_lstm_inputs_train = self.scaler.transform(att_lstm_inputs_train)
        nbhd_inputs_train = self.scaler.transform(nbhd_inputs_train)
        lstm_inputs_train = self.scaler.transform(lstm_inputs_train)
        x_val = self.scaler.transform(x_val)
        y_val = self.scaler.transform(y_val)
        flatten_att_nbhd_inputs_val = self.scaler.transform(flatten_att_nbhd_inputs_val)
        att_lstm_inputs_val = self.scaler.transform(att_lstm_inputs_val)
        nbhd_inputs_val = self.scaler.transform(nbhd_inputs_val)
        lstm_inputs_val = self.scaler.transform(lstm_inputs_val)
        x_test = self.scaler.transform(x_test)
        y_test = self.scaler.transform(y_test)
        flatten_att_nbhd_inputs_test = self.scaler.transform(flatten_att_nbhd_inputs_test)
        att_lstm_inputs_test = self.scaler.transform(att_lstm_inputs_test)
        nbhd_inputs_test = self.scaler.transform(nbhd_inputs_test)
        lstm_inputs_test = self.scaler.transform(lstm_inputs_test)

        flatten_att_flow_inputs_train = self.flow_scaler.transform(flatten_att_flow_inputs_train)
        flow_inputs_train = self.flow_scaler.transform(flow_inputs_train)
        flatten_att_flow_inputs_val = self.flow_scaler.transform(flatten_att_flow_inputs_val)
        flow_inputs_val = self.flow_scaler.transform(flow_inputs_val)
        flatten_att_flow_inputs_test = self.flow_scaler.transform(flatten_att_flow_inputs_test)
        flow_inputs_test = self.flow_scaler.transform(flow_inputs_test)

        train_data = list(
            zip(x_train, y_train, flatten_att_nbhd_inputs_train, flatten_att_flow_inputs_train, att_lstm_inputs_train,
                nbhd_inputs_train, flow_inputs_train, lstm_inputs_train))
        eval_data = list(
            zip(x_val, y_val, flatten_att_nbhd_inputs_val, flatten_att_flow_inputs_val, att_lstm_inputs_val,
                nbhd_inputs_val, flow_inputs_val, lstm_inputs_val))
        test_data = list(
            zip(x_test, y_test, flatten_att_nbhd_inputs_test, flatten_att_flow_inputs_test, att_lstm_inputs_test,
                nbhd_inputs_test, flow_inputs_test, lstm_inputs_test))
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        return {"scaler": self.scaler, "adj_mx": self.adj_mx,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "len_row": self.len_row, "len_column": self.len_column,
                "feature_vec_len": self.feature_vec_len, "nbhd_type": self.nbhd_type}
