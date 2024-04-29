import os

import numpy as np

from libcity.data.dataset import TrafficStatePointDataset
from libcity.data.utils import generate_dataloader
from libcity.utils import ensure_dir


def search_multihop_neighbor(adj, hops=5):
    node_cnt = adj.shape[0]
    hop_arr = np.zeros((adj.shape[0], adj.shape[0]))
    for h_idx in range(node_cnt):  # refer node idx(n)
        tmp_h_node, tmp_neibor_step = [h_idx], [h_idx]  # save spatial corr node  # 0 step(self) first
        hop_arr[h_idx, :] = -1  # if the value exceed maximum hop, it is set to (hops + 1)
        hop_arr[h_idx, h_idx] = 0  # at begin, the hop of self->self is set to 0
        for hop_idx in range(hops):  # how many spatial steps
            tmp_step_node = []  # neighbor nodes in the previous k step
            tmp_step_node_kth = []  # neighbor nodes in the kth step
            for tmp_nei_node in tmp_neibor_step:
                tmp_neibor_step = list(
                    (np.argwhere(adj[tmp_nei_node] == 1).flatten()))  # find the one step neighbor first
                tmp_step_node += tmp_neibor_step
                tmp_step_node_kth += set(tmp_step_node) - set(
                    tmp_h_node)  # the nodes that have appeared in the first k-1 step are no longer needed
                tmp_h_node += tmp_neibor_step
            tmp_neibor_step = tmp_step_node_kth.copy()
            all_spatial_node = list(set(tmp_neibor_step))  # the all spatial node in kth step
            hop_arr[h_idx, all_spatial_node] = hop_idx + 1
    return hop_arr[:, :, np.newaxis]


class STPGCNDataset(TrafficStatePointDataset):
    def __init__(self, config):
        super().__init__(config)
        self.parameters_str += "_STPGCN"
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'point_based_{}.npz'.format(self.parameters_str))

        self.feature_name = {'X': 'float', 'y': 'float', 'pos_w': 'int', 'pos_d': 'int'}
        self.points_per_hour = config.get('points_per_hour', 12)
        self.alpha = config.get('alpha', 4)
        self.beta = config.get('beta', 2)
        self.t_size = self.beta + 1
        self.spatial_distance = search_multihop_neighbor(self.adj_mx, hops=self.alpha)
        self.range_mask = self.interaction_range_mask(t_size=self.t_size)

    def interaction_range_mask(self, t_size=3):
        hop_arr = self.spatial_distance
        hop_arr[hop_arr != -1] = 1
        hop_arr[hop_arr == -1] = 0
        return np.concatenate([hop_arr.squeeze()] * t_size, axis=-1)  # V,tV

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，num_nodes是点的个数，feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"scaler": self.scaler, "ext_dim": self.ext_dim, "spatial_distance": self.spatial_distance,
                "range_mask": self.range_mask, "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches}

    def generate_data(self):
        """
        加载数据文件

        Returns:
            df: (num_samples, num_nodes, feature)
        """
        # 处理多数据文件问题
        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:  # str
            data_files = [self.data_files].copy()
        # 加载外部数据
        if self.load_external and os.path.exists(self.data_path + self.ext_file + '.ext'):  # 外部数据集
            ext_data = self._load_ext()
        else:
            ext_data = None
        all_df = []
        for filename in data_files:
            df = self._load_dyna(filename)  # (len_time, ..., feature_dim)
            if self.load_external:
                df = self._add_external_information(df, ext_data)
            all_df.append(df)
        df = np.concatenate(all_df)
        self._logger.info("Dataset created")
        self._logger.info("final df shape: " + str(df.shape))
        return df

    def handle_data(self, df, data_range):
        def search_recent_data(train, label_start_idx, num_prediction):
            if label_start_idx + num_prediction > len(train): return None
            start_idx, end_idx = label_start_idx - num_prediction, label_start_idx - num_prediction + num_prediction
            if start_idx < 0 or end_idx < 0: return None
            return (start_idx, end_idx), (label_start_idx, label_start_idx + num_prediction)

        def get_time_pos(num_prediction, idx, points_per_hour):
            idx = np.array(range(num_prediction)) + idx
            pos_w = (idx // (points_per_hour * 24)) % 7  # day of week
            pos_d = idx % (points_per_hour * 24)  # time of day
            return pos_w, pos_d

        # param prepare
        points_per_hour = self.points_per_hour
        num_prediction = self.output_window

        # generate idx_lst
        idx_lst = []
        start = data_range[0]
        end = data_range[1] if data_range[1] != -1 else df.shape[0]
        for label_start_idx in range(start, end):
            recent = search_recent_data(df, label_start_idx, num_prediction)
            if recent:
                idx_lst.append(recent)
        x_list, y_list, pos_w_list, pos_d_list = [], [], [], []
        for index in range(len(idx_lst)):
            recent_idx = idx_lst[index]
            # y
            start, end = recent_idx[1][0], recent_idx[1][1]
            t_y = df[start:end]
            y_list.append(t_y)
            # x
            start, end = recent_idx[0][0], recent_idx[0][1]
            t_x = df[start:end]
            x_list.append(t_x)
            # w d
            pos_w, pos_d = get_time_pos(num_prediction, start, points_per_hour)
            pos_w_list.append(pos_w)
            pos_d_list.append(pos_d)
        return np.array(x_list), np.array(y_list), np.array(pos_w_list), np.array(pos_d_list)

    def split_data(self, df):
        num_samples = df.shape[0]
        val_start_idx = int(num_samples * self.train_rate)
        test_start_idx = int(num_samples * (self.train_rate + self.eval_rate))
        train_x, train_y, train_pos_w, train_pos_d = self.handle_data(df, (0 + self.output_window,
                                                                           val_start_idx - self.output_window + 1))
        val_x, val_y, val_pos_w, val_pos_d = self.handle_data(df, (val_start_idx + self.output_window,
                                                                   test_start_idx - self.output_window + 1))
        test_x, test_y, test_pos_w, test_pos_d = self.handle_data(df, (test_start_idx + self.output_window, -1))

        self._logger.info("train\t" + "x: " + str(train_x.shape) + ", y: " + str(train_y.shape))
        self._logger.info("eval\t" + "x: " + str(val_x.shape) + ", y: " + str(val_y.shape))
        self._logger.info("test\t" + "x: " + str(test_x.shape) + ", y: " + str(test_y.shape))

        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                self.cache_file_name,
                x_train=train_x,
                y_train=train_y,
                x_test=test_x,
                y_test=test_y,
                x_val=val_x,
                y_val=val_y,
                train_pos_w=train_pos_w,
                train_pos_d=train_pos_d,
                val_pos_w=val_pos_w,
                val_pos_d=val_pos_d,
                test_pos_w=test_pos_w,
                test_pos_d=test_pos_d,
            )
            self._logger.info('Saved at ' + self.cache_file_name)
        return train_x, train_y, val_x, val_y, test_x, test_y, train_pos_w, train_pos_d, val_pos_w, val_pos_d, \
            test_pos_w, test_pos_d

    def _load_cache_train_val_test(self):
        self._logger.info('Loading ' + self.cache_file_name)
        cat_data = np.load(self.cache_file_name)
        x_train = cat_data['x_train']
        y_train = cat_data['y_train']
        x_test = cat_data['x_test']
        y_test = cat_data['y_test']
        x_val = cat_data['x_val']
        y_val = cat_data['y_val']
        train_pos_w = cat_data['train_pos_w']
        train_pos_d = cat_data['train_pos_d']
        val_pos_w = cat_data['val_pos_w']
        val_pos_d = cat_data['val_pos_d']
        test_pos_w = cat_data['test_pos_w']
        test_pos_d = cat_data['test_pos_d']
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))
        return x_train, y_train, x_val, y_val, x_test, y_test, train_pos_w, train_pos_d, val_pos_w, val_pos_d, \
            test_pos_w, test_pos_d

    def _generate_train_val_test(self):
        df = self.generate_data()
        return self.split_data(df)

    def get_data(self):
        # 加载数据集
        x_train, y_train, x_val, y_val, x_test, y_test, train_pos_w, train_pos_d, val_pos_w, val_pos_d, \
            test_pos_w, test_pos_d = [], [], [], [], [], [], [], [], [], [], [], []
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train, y_train, x_val, y_val, x_test, y_test, train_pos_w, train_pos_d, val_pos_w, val_pos_d, \
                    test_pos_w, test_pos_d = self._load_cache_train_val_test()
            else:
                x_train, y_train, x_val, y_val, x_test, y_test, train_pos_w, train_pos_d, val_pos_w, val_pos_d, \
                    test_pos_w, test_pos_d = self._generate_train_val_test()
        # 归一化
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
        # 转 dataloader
        train_data = list(zip(x_train, y_train, train_pos_w, train_pos_d))
        eval_data = list(zip(x_val, y_val, val_pos_w, val_pos_d))
        test_data = list(zip(x_test, y_test, test_pos_w, test_pos_d))
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample)
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader
