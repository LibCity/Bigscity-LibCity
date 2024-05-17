import copy
import os
import datetime

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from libcity.data.batch import Batch
from libcity.data.dataset import TrafficStatePointDataset
from libcity.data.list_dataset import ListDataset
from libcity.utils import ensure_dir


class SimSTDataset(TrafficStatePointDataset):
    def __init__(self, config):
        self.normalized_k = config.get("normalized_k", 0)
        super().__init__(config)
        self.in_neighbor_num = config.get("in_neighbor_num", 0)
        self.infer_bs = config.get("infer_bs", 64)
        self.y_start = config.get("y_start", 1)
        self.feature_name = {'X': 'float', 'y': 'float', 'node_idx': 'int'}

    def _load_rel(self):
        def get_adjacency_matrix(distance_df, sensor_id_to_ind, normalized_k=0):
            num_sensors = len(sensor_ids)
            dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
            dist_mx[:] = np.inf

            # Fills cells in the matrix with distances
            for row in distance_df.values:
                if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
                    continue
                dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

            # Calculates the standard deviation as theta
            distances = dist_mx[~np.isinf(dist_mx)].flatten()
            std = distances.std()
            adj_mx = np.exp(-np.square(dist_mx / std))

            # Sets entries that lower than a threshold to zero for sparsity
            adj_mx[adj_mx < normalized_k] = 0
            return adj_mx

        distance_df = pd.read_csv(self.data_path + self.rel_file + '.rel',
                                  dtype={'origin_id': 'str', 'destination_id': 'str'})
        distance_df = distance_df.drop(['rel_id', 'type'], axis=1)
        t = distance_df['origin_id'] == distance_df['destination_id']
        if np.sum(t) != 0:
            for i in range(len(t)):
                if t[i] == True:
                    distance_df = distance_df.drop([i])

        # Builds sensor id to index map
        sensor_id_to_ind = {}
        sensor_ids = distance_df['origin_id'].tolist() + distance_df['destination_id'].tolist()
        sensor_ids = list(set(sensor_ids))
        sensor_ids = [int(i) for i in sensor_ids]

        sensor_ids = [str(i) for i in range(len(sensor_ids))]
        for i, sensor_id in enumerate(sensor_ids):
            sensor_id_to_ind[sensor_id] = i

        # Generate adjacency matrix
        adj_mx = get_adjacency_matrix(distance_df, sensor_id_to_ind, self.normalized_k)
        adj_mx += + np.eye(len(sensor_ids))
        self.adj_mx = adj_mx

    def _add_external_information(self, df, ext_data=None):
        return self._add_external_information_3d(df, ext_data)

    def _add_external_information_3d(self, df, ext_data=None):
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
            day_in_week = np.tile(dayofweek, [1, num_nodes, 1]).transpose((2, 1, 0))
            day_in_week = day_in_week / 7
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

    def _generate_train_val_test(self):

        def generate_graph_seq2seq_io_data(df, x_offsets, y_offsets, add_time_in_day, add_day_in_week, timeslots):
            num_samples, num_nodes = df.shape
            # data = np.expand_dims(df.values, axis=-1)
            data = np.expand_dims(df, axis=-1)
            feature_list = [data]
            if add_time_in_day:
                time_ind = (timeslots - timeslots.astype("datetime64[D]")) / np.timedelta64(1, "D")
                time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                feature_list.append(time_in_day)

                # time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
                # time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                # feature_list.append(time_in_day)
            if add_day_in_week:
                dayofweek = []
                for day in timeslots.astype("datetime64[D]"):
                    dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
                day_in_week = np.tile(dayofweek, [1, num_nodes, 1]).transpose((2, 1, 0))
                day_in_week = day_in_week / 7
                feature_list.append(day_in_week)

                # dow = df.index.dayofweek
                # dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
                # dow_tiled = dow_tiled / 7
                # feature_list.append(dow_tiled)

            data = np.concatenate(feature_list, axis=-1)
            x, y = [], []
            min_t = abs(min(x_offsets))
            max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
            for t in range(min_t, max_t):  # t is the index of the last observation.
                x.append(data[t + x_offsets, ...])
                y.append(data[t + y_offsets, ...])
            x = np.stack(x, axis=0)
            y = np.stack(y, axis=0)
            return x, y

        # 获取 df
        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:  # str
            data_files = [self.data_files].copy()
        df_list = []
        for filename in data_files:
            df = self._load_dyna(filename)
            df_list.append(df)
        df = np.concatenate(df_list)
        # 
        seq_length_x, seq_length_y = self.input_window, self.output_window
        x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
        y_offsets = np.sort(np.arange(self.y_start, (seq_length_y + 1), 1))
        df = df.squeeze()
        num_samples = df.shape[0]
        num_train = int(num_samples * self.train_rate)
        num_val = int(num_samples * (self.train_rate + self.eval_rate))

        x_train, y_train = generate_graph_seq2seq_io_data(df[:num_train], x_offsets, y_offsets, self.add_time_in_day,
                                                          self.add_day_in_week, self.timesolts[:num_train])
        x_val, y_val = generate_graph_seq2seq_io_data(df[num_train: num_val], x_offsets, y_offsets,
                                                      self.add_time_in_day, self.add_day_in_week,
                                                      self.timesolts[num_train: num_val])
        x_test, y_test = generate_graph_seq2seq_io_data(df[num_val:], x_offsets, y_offsets, self.add_time_in_day,
                                                        self.add_day_in_week, self.timesolts[num_val:])

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

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """

        def adjust_dataset(xs, ys, batch_size, neighbor_record, pad_with_last_sample):
            """
            调整数据集结构

            :param xs: samples T N F
            :param ys: samples T N F
            :param batch_size: batch size
            :param neighbor_record: neighbor_record
            :param pad_with_last_sample: pad_with_last_sample
            :return:
                x, y: if batch_size > 64 return (samples * N, F, 1, T) else return (samples, 1, N, T)
                node_idx: if batch_size > 64 return (samples * N,) else return (samples,)
            """
            # samples T N F -> samples N F T
            xs = np.transpose(xs, (0, 2, 3, 1))
            ys = np.transpose(ys, (0, 2, 3, 1))
            # y 集截取出来F的第一个维度，也就是流量特征，忽略后面的 tod 和 dow   -> samples N 1 T
            ys = ys[:, :, :1, :]
            node_num = xs.shape[1]
            # add neighbor_record
            if neighbor_record != None:
                aug_xs = []
                for s in xs:
                    all_nb_ts = []
                    for n in range(node_num):
                        feature_num = len(neighbor_record[n])
                        nb_ts = []
                        for f in range(feature_num):
                            ids = neighbor_record[n][f]
                            nb_num = len(ids)
                            ts = np.zeros(12)
                            for i in range(nb_num):
                                ts = ts + s[ids[i]][0]
                            ts = ts / nb_num
                            nb_ts.append(ts)
                        all_nb_ts.append(nb_ts)
                    all_nb_ts = np.array(all_nb_ts)
                    new_x = np.concatenate([s, all_nb_ts], axis=1)
                    aug_xs.append(new_x)
                xs = np.array(aug_xs)
            if batch_size > 64:
                node_idx = np.arange(node_num).reshape(1, -1).repeat(xs.shape[0], axis=0).reshape(-1)
                xs = xs.reshape(-1, xs.shape[2], 1, xs.shape[3])
                ys = ys.reshape(-1, ys.shape[2], 1, ys.shape[3])
            else:
                node_idx = np.arange(xs.shape[0])
                xs = np.transpose(xs, (0, 2, 1, 3))
                ys = np.transpose(ys, (0, 2, 1, 3))

            if pad_with_last_sample:
                num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
                x_padding = np.repeat(xs[-1:], num_padding, axis=0)
                y_padding = np.repeat(ys[-1:], num_padding, axis=0)
                idx_padding = np.repeat(node_idx[-1], num_padding)
                xs = np.concatenate([xs, x_padding], axis=0)
                ys = np.concatenate([ys, y_padding], axis=0)
                node_idx = np.concatenate([node_idx, idx_padding])
            # samples F N T -> samples T N F
            xs = np.transpose(xs, (0, 3, 2, 1))
            ys = np.transpose(ys, (0, 3, 2, 1))
            return xs, ys, node_idx

        def build_neighbor_record(adj_mx, node_num, in_neighbor_num=0):
            """
            通过 adj_mx 构建 neighbor_record 字典

            :param adj_mx: adj_mx
            :param node_num: number of nodes
            :param in_neighbor_num: highlighted neighbors number
            :return: neighbor_record
            """
            pdf = adj_mx
            tpdf = np.transpose(pdf)
            pdfs = [pdf]
            tpdfs = [tpdf]

            k = in_neighbor_num
            neighbor_record = {}
            for n in range(node_num):
                nb1 = np.nonzero(pdfs[0][n])[0]
                tnb1 = np.nonzero(tpdfs[0][n])[0]
                nb1 = np.delete(nb1, np.argwhere(nb1 == n))
                tnb1 = np.delete(tnb1, np.argwhere(tnb1 == n))

                w = pdf[n, nb1]
                tw = tpdf[n, tnb1]
                w_idx = w.argsort()[-k:]
                tw_idx = tw.argsort()[-k:]
                n_id = list(nb1[w_idx])
                tn_id = list(tnb1[tw_idx])

                while len(n_id) < k:
                    n_id.append(n)
                while len(tn_id) < k:
                    tn_id.append(n)

                if len(nb1) == 0: nb1 = [n]
                if len(tnb1) == 0: tnb1 = [n]

                neighbor_record[n] = []
                for i in range(k - 1, -1, -1):
                    neighbor_record[n].append([n_id[i]])
                    neighbor_record[n].append([tn_id[i]])
                neighbor_record[n].append(list(nb1))
                neighbor_record[n].append(list(tnb1))
            return neighbor_record

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
        self.ext_dim = self.feature_dim - self.output_dim
        self.scaler = self._get_scalar(self.scaler_type,
                                       x_train[..., :self.output_dim], y_train[..., :self.output_dim])
        self.ext_scaler = self._get_scalar(self.ext_scaler_type,
                                           x_train[..., self.output_dim:], y_train[..., self.output_dim:])
        x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_val[..., :self.output_dim] = self.scaler.transform(x_val[..., :self.output_dim])
        y_val[..., :self.output_dim] = self.scaler.transform(y_val[..., :self.output_dim])
        x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])
        if self.normal_external:
            x_train[..., self.output_dim:] = self.ext_scaler.transform(x_train[..., self.output_dim:])
            y_train[..., self.output_dim:] = self.ext_scaler.transform(y_train[..., self.output_dim:])
            x_val[..., self.output_dim:] = self.ext_scaler.transform(x_val[..., self.output_dim:])
            y_val[..., self.output_dim:] = self.ext_scaler.transform(y_val[..., self.output_dim:])
            x_test[..., self.output_dim:] = self.ext_scaler.transform(x_test[..., self.output_dim:])
            y_test[..., self.output_dim:] = self.ext_scaler.transform(y_test[..., self.output_dim:])

        # build neighbor record
        neighbor_record = build_neighbor_record(self.adj_mx, self.num_nodes, self.in_neighbor_num)
        # adjust dataset
        x_train, y_train, node_idx_train = adjust_dataset(x_train, y_train, self.batch_size, neighbor_record,
                                                          self.pad_with_last_sample)
        x_val, y_val, node_idx_val = adjust_dataset(x_val, y_val, self.infer_bs, neighbor_record,
                                                    self.pad_with_last_sample)
        x_test, y_test, node_idx_test = adjust_dataset(x_test, y_test, self.infer_bs, neighbor_record,
                                                       self.pad_with_last_sample)
        self._logger.info("final dataset shape")
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))

        train_data = list(zip(x_train, y_train, node_idx_train))
        eval_data = list(zip(x_val, y_val, node_idx_val))
        test_data = list(zip(x_test, y_test, node_idx_test))

        # 转Dataloader
        train_dataset = ListDataset(train_data)
        eval_dataset = ListDataset(eval_data)
        test_dataset = ListDataset(test_data)

        def collator(indices):
            batch = Batch(self.feature_name)
            for item in indices:
                batch.append(copy.deepcopy(item))
            return batch

        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, collate_fn=collator,
                                           shuffle=True)

        self.eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.infer_bs,
                                          num_workers=self.num_workers, collate_fn=collator,
                                          shuffle=False)

        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.infer_bs,
                                          num_workers=self.num_workers, collate_fn=collator,
                                          shuffle=False)
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader
