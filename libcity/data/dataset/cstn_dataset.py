import os

import numpy as np

from libcity.data.dataset import TrafficStateGridOdDataset
from libcity.data.utils import generate_dataloader
from libcity.utils import ensure_dir


class CSTNDataset(TrafficStateGridOdDataset):

    def __init__(self, config):
        super().__init__(config)
        self.feature_name = {'X': 'float', 'W': 'float', 'y': 'float'}

    def _generate_ext_data(self, ext_data):
        num_samples = ext_data.shape[0]
        offsets = np.sort(np.concatenate((np.arange(-self.input_window - self.output_window + 1, 1, 1),)))
        min_t = abs(min(offsets))
        max_t = abs(num_samples - abs(max(offsets)))
        W = []
        for t in range(min_t, max_t):
            W_t = ext_data[t + offsets, ...]
            W.append(W_t)
        W = np.stack(W, axis=0)
        return W

    def _generate_data(self):
        """
        加载数据文件(.gridod)和外部数据(.ext)，以X, W, y的形式返回

        Returns:
            tuple: tuple contains:
                X(np.ndarray): 模型输入数据，(num_samples, input_length, ..., feature_dim) \n
                W(np.ndarray): 模型外部数据，(num_samples, input_length, ext_dim)
                y(np.ndarray): 模型输出数据，(num_samples, output_length, ..., feature_dim)
        """
        # 处理多数据文件问题
        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:
            data_files = [self.data_files].copy()

        # 加载外部数据
        ext_data = self._load_ext()  # （len_time, ext_dim)
        W = self._generate_ext_data(ext_data)

        # 加载基本特征数据
        X_list, y_list = [], []
        for filename in data_files:
            df = self._load_dyna(filename)  # (len_time, ..., feature_dim)
            X, y = self._generate_input_data(df)
            # x: (num_samples, input_length, input_dim)
            # y: (num_samples, output_length, ..., output_dim)
            X_list.append(X)
            y_list.append(y)
        X = np.concatenate(X_list)
        y = np.concatenate(y_list)

        df = self._load_dyna(data_files[0]).squeeze()
        self._logger.info("Dataset created")
        self._logger.info("X shape: {}, W shape: {}, y shape: ".format(str(X.shape), str(W.shape), y.shape))
        return X, W, y

    def _split_train_val_test(self, X, W, y):
        test_rate = 1 - self.train_rate - self.eval_rate
        num_samples = X.shape[0]
        num_test = round(num_samples * test_rate)
        num_train = round(num_samples * self.train_rate)
        num_eval = num_samples - num_test - num_train
        # train
        x_train, w_train, y_train = X[:num_train], W[:num_train], y[:num_train]

        # eval
        x_eval, w_eval, y_eval = X[num_train: num_train + num_eval], \
                                 W[num_train: num_train + num_eval], y[num_train: num_train + num_eval]
        # test
        x_test, w_test, y_test = X[-num_test:], W[-num_test:], y[-num_test:]

        # log
        self._logger.info(
            "train\tX: {}, W: {}, y: {}".format(str(x_train.shape), str(w_train.shape), str(y_train.shape)))
        self._logger.info("eval\tX: {}, W: {}, y: {}".format(str(x_eval.shape), str(w_eval.shape), str(y_eval.shape)))
        self._logger.info("test\tX: {}, W: {}, y: {}".format(str(x_test.shape), str(w_test.shape), str(y_test.shape)))
        return x_train, w_train, y_train, x_eval, w_eval, y_eval, x_test, w_test, y_test

    def _load_cache_train_val_test(self):
        self._logger.info('Loading ' + self.cache_file_name)
        cat_data = np.load(self.cache_file_name)
        x_train, w_train, y_train, x_eval, w_eval, y_eval, x_test, w_test, y_test = \
            cat_data['x_train'], cat_data['w_train'], cat_data['y_train'], cat_data['x_eval'], cat_data['w_eval'], \
            cat_data['y_eval'], cat_data['x_test'], cat_data['w_test'], cat_data['y_test']

        self._logger.info(
            "train\tX: {}, W: {}, y: {}".format(str(x_train.shape), str(w_train.shape), str(y_train.shape)))
        self._logger.info("eval\tX: {}, W: {}, y: {}".format(str(x_eval.shape), str(w_eval.shape), str(y_eval.shape)))
        self._logger.info("test\tX: {}, W: {}, y: {}".format(str(x_test.shape), str(w_test.shape), str(y_test.shape)))

        return x_train, w_train, y_train, x_eval, w_eval, y_eval, x_test, w_test, y_test

    def _generate_train_val_test(self):
        X, W, y = self._generate_data()
        x_train, w_train, y_train, x_eval, w_eval, y_eval, x_test, w_test, y_test = self._split_train_val_test(X, W, y)

        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                self.cache_file_name,
                x_train=x_train,
                w_train=w_train,
                y_train=y_train,
                x_test=x_test,
                w_test=w_test,
                y_test=y_test,
                x_eval=x_eval,
                w_eval=w_eval,
                y_eval=y_eval,
            )
            self._logger.info('Saved at ' + self.cache_file_name)

        return x_train, w_train, y_train, x_eval, w_eval, y_eval, x_test, w_test, y_test

    def get_data(self):
        # 加载数据集
        x_train, w_train, y_train, x_eval, w_eval, y_eval, x_test, w_test, y_test = [], [], [], [], [], [], [], [], []
        if self.data is None:
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train, w_train, y_train, x_eval, w_eval, y_eval, x_test, w_test, y_test = self._load_cache_train_val_test()
            else:
                x_train, w_train, y_train, x_eval, w_eval, y_eval, x_test, w_test, y_test = self._generate_train_val_test()

        # 数据归一化
        self.feature_dim = x_train.shape[-1]
        self.ext_dim = w_train.shape[-1]
        self.scaler = self._get_scalar(self.scaler_type, x_train, y_train)
        x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
        w_train[..., :self.output_dim] = self.scaler.transform(w_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_eval[..., :self.output_dim] = self.scaler.transform(x_eval[..., :self.output_dim])
        w_eval[..., :self.output_dim] = self.scaler.transform(w_eval[..., :self.output_dim])
        y_eval[..., :self.output_dim] = self.scaler.transform(y_eval[..., :self.output_dim])
        x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
        w_test[..., :self.output_dim] = self.scaler.transform(w_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])

        train_data = list(zip(x_train, w_train, y_train))
        eval_data = list(zip(x_eval, w_eval, y_eval))
        test_data = list(zip(x_test, w_test, y_test))

        # 转Dataloader
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample)
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是网格的个数，
        len_row是网格的行数，len_column是网格的列数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"scaler": self.scaler,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim, "ext_dim": self.ext_dim,
                "output_dim": self.output_dim, "len_row": self.len_row, "len_column": self.len_column,
                "num_batches": self.num_batches}
