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
        self.points_per_hour = self.time_intervals // 60
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
                "output_dim": self.output_dim, "num_batches": self.num_batches, "points_per_hour": self.points_per_hour}

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

    def _split_train_val_test(self, x, y):
        def get_time_pos(idx):
            idx = np.array(range(self.input_window)) + idx
            pos_w = (idx // (self.points_per_hour * 24)) % 7  # day of week
            pos_d = idx % (self.points_per_hour * 24)  # time of day
            return pos_w, pos_d

        test_rate = 1 - self.train_rate - self.eval_rate
        num_samples = x.shape[0]
        num_test = round(num_samples * test_rate)
        num_train = round(num_samples * self.train_rate)
        num_val = num_samples - num_test - num_train

        # generate pos_w pos_d
        pos_w_list, pos_d_list = [], []
        for idx in range(num_samples):
            pos_w, pos_d = get_time_pos(idx)
            pos_w_list.append(pos_w)
            pos_d_list.append(pos_d)

        # (num_samples, T)
        pos_w_list = np.stack(pos_w_list, axis=0)
        pos_d_list = np.stack(pos_d_list, axis=0)

        # train
        x_train, y_train = x[:num_train], y[:num_train]
        train_pos_w, train_pos_d = pos_w_list[:num_train], pos_d_list[:num_train]
        # val
        x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
        val_pos_w, val_pos_d = pos_w_list[num_train: num_train + num_val], pos_d_list[num_train: num_train + num_val]
        # test
        x_test, y_test = x[-num_test:], y[-num_test:]
        test_pos_w, test_pos_d = pos_w_list[-num_test:], pos_d_list[-num_test:]

        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))

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
                train_pos_w=train_pos_w,
                train_pos_d=train_pos_d,
                val_pos_w=val_pos_w,
                val_pos_d=val_pos_d,
                test_pos_w=test_pos_w,
                test_pos_d=test_pos_d,
            )
            self._logger.info('Saved at ' + self.cache_file_name)
        return x_train, y_train, x_val, y_val, x_test, y_test, train_pos_w, train_pos_d, val_pos_w, val_pos_d, \
            test_pos_w, test_pos_d

    def _generate_train_val_test(self):
        x, y = super()._generate_data()
        return self._split_train_val_test(x, y)

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
