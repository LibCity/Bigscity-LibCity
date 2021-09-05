import os
import numpy as np
from libcity.data.dataset import TrafficStatePointDataset
from libcity.utils import ensure_dir
from libcity.data.utils import generate_dataloader

"""
需要有.ext文件！
"""


class CRANNDataset(TrafficStatePointDataset):

    def __init__(self, config):
        super().__init__(config)
        self.feature_name = {'x_time': 'float', 'x_space': 'float', 'x_ext': 'float', 'y': 'float'}
        self.n_timesteps = self.config.get('n_timesteps', 24*14)
        self.dim_x = self.config.get('dim_x', 5)
        self.dim_y = self.config.get('dim_y', 6)

    def _generate_data(self):
        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:  # str
            data_files = [self.data_files].copy()
        df = self._load_dyna(data_files[0]).squeeze()  # (T, N)
        ext_data = self._load_ext()  # (T, F)

        data_time = []
        data_space = []
        data_ext = []
        n_data = len(df) - 1 - (self.n_timesteps + self.output_window)
        indexes1 = np.arange(n_data)
        indexes2 = indexes1 + (self.n_timesteps + self.output_window) - (self.input_window + self.output_window)
        for index1, index2 in zip(indexes1, indexes2):
            data_time.append(ext_data[index1:index1 + (self.n_timesteps + self.output_window), 0:1])
            data_space.append(df[index2:index2 + (self.input_window + self.output_window), 0:self.dim_x * self.dim_y])
            data_ext.append(ext_data[(index2 + self.input_window):
                                     index2 + (self.input_window + self.output_window), 1:])

        data_time = np.array(data_time).\
            reshape((-1, (self.n_timesteps + self.output_window), 1))
        data_space = np.array(data_space).\
            reshape((-1, (self.input_window + self.output_window), self.dim_x, self.dim_y))
        data_ext = np.array(data_ext).reshape((-1, self.output_window, ext_data.shape[1] - 1))

        x_time, y_time = data_time[:, :self.n_timesteps, :], data_time[:, self.n_timesteps:, :]
        x_space, y_space = data_space[:, :self.input_window, :, :], data_space[:, self.input_window:, :, :]
        x_ext = data_ext
        self._logger.info("Dataset created")
        self._logger.info("x_time shape: " + str(x_time.shape) + ", x_space shape: " + str(x_space.shape)
                          + ", x_exo shape: " + str(x_ext.shape) + ", y shape: " + str(y_space.shape))
        return x_time, x_space, x_ext, y_space

    def _split_train_val_test(self, x_time, x_space, x_ext, y):
        test_rate = 1 - self.train_rate - self.eval_rate
        num_samples = x_time.shape[0]
        num_test = round(num_samples * test_rate)
        num_train = round(num_samples * self.train_rate)
        num_val = num_samples - num_test - num_train

        # train
        x_time_train, x_space_train, x_ext_train, y_train = \
            x_time[:num_train], x_space[:num_train], x_ext[:num_train], y[:num_train]
        # val
        x_time_val, x_space_val, x_ext_val, y_val = \
            x_time[num_train: num_train + num_val], x_space[num_train: num_train + num_val], \
            x_ext[num_train: num_train + num_val], y[num_train: num_train + num_val]
        # test
        x_time_test, x_space_test, x_ext_test, y_test = \
            x_time[-num_test:], x_space[-num_test:], x_ext[-num_test:], y[-num_test:]
        self._logger.info("train\t" + "x_time: " + str(x_time_train.shape) + ", x_space: " + str(x_space_train.shape)
                          + ", x_ext: " + str(x_ext_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x_time: " + str(x_time_val.shape) + ", x_space: " + str(x_space_val.shape)
                          + ", x_ext: " + str(x_ext_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x_time: " + str(x_time_test.shape) + ", x_space: " + str(x_space_test.shape)
                          + ", x_ext: " + str(x_ext_test.shape) + ", y: " + str(y_test.shape))

        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                self.cache_file_name,
                x_time_train=x_time_train,
                x_space_train=x_space_train,
                x_ext_train=x_ext_train,
                x_time_val=x_time_val,
                x_space_val=x_space_val,
                x_ext_val=x_ext_val,
                x_time_test=x_time_test,
                x_space_test=x_space_test,
                x_ext_test=x_ext_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
            )
            self._logger.info('Saved at ' + self.cache_file_name)
        return x_time_train, x_space_train, x_ext_train, y_train, x_time_val, x_space_val, x_ext_val, y_val, \
               x_time_test, x_space_test, x_ext_test, y_test

    def _generate_train_val_test(self):
        x_time, x_space, x_ext, y = self._generate_data()
        return self._split_train_val_test(x_time, x_space, x_ext, y)

    def _load_cache_train_val_test(self):
        self._logger.info('Loading ' + self.cache_file_name)
        cat_data = np.load(self.cache_file_name)
        x_time_train = cat_data['x_time_train']
        x_space_train = cat_data['x_space_train']
        x_ext_train = cat_data['x_ext_train']
        y_train = cat_data['y_train']
        x_time_test = cat_data['x_time_test']
        x_space_test = cat_data['x_space_test']
        x_ext_test = cat_data['x_ext_test']
        y_test = cat_data['y_test']
        x_time_val = cat_data['x_time_val']
        x_space_val = cat_data['x_space_val']
        x_ext_val = cat_data['x_ext_val']
        y_val = cat_data['y_val']
        self._logger.info("train\t" + "x_time: " + str(x_time_train.shape) + ", x_space: " + str(x_space_train.shape)
                          + ", x_ext: " + str(x_ext_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x_time: " + str(x_time_val.shape) + ", x_space: " + str(x_space_val.shape)
                          + ", x_ext: " + str(x_ext_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x_time: " + str(x_time_test.shape) + ", x_space: " + str(x_space_test.shape)
                          + ", x_ext: " + str(x_ext_test.shape) + ", y: " + str(y_test.shape))
        return x_time_train, x_space_train, x_ext_train, y_train, x_time_val, x_space_val, x_ext_val, y_val, \
               x_time_test, x_space_test, x_ext_test, y_test

    def get_data(self):
        # 加载数据集
        x_time_train, x_space_train, x_ext_train, y_train, x_time_val, x_space_val, x_ext_val, y_val, \
            x_time_test, x_space_test, x_ext_test, y_test = [], [], [], [], [], [], [], [], [], [], [], []
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_time_train, x_space_train, x_ext_train, y_train, x_time_val, x_space_val, x_ext_val, y_val, \
                    x_time_test, x_space_test, x_ext_test, y_test = self._load_cache_train_val_test()
            else:
                x_time_train, x_space_train, x_ext_train, y_train, x_time_val, x_space_val, x_ext_val, y_val, \
                    x_time_test, x_space_test, x_ext_test, y_test = self._generate_train_val_test()
        # 数据归一化
        self.feature_dim = x_time_train.shape[-1]
        self.ext_dim = x_ext_train.shape[-1]
        self.scaler = self._get_scalar(self.scaler_type, x_space_train, y_train)
        x_time_train[..., :self.output_dim] = self.scaler.transform(x_time_train[..., :self.output_dim])
        x_space_train[..., :self.output_dim] = self.scaler.transform(x_space_train[..., :self.output_dim])
        x_ext_train[..., :self.output_dim] = self.scaler.transform(x_ext_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_time_val[..., :self.output_dim] = self.scaler.transform(x_time_val[..., :self.output_dim])
        x_space_val[..., :self.output_dim] = self.scaler.transform(x_space_val[..., :self.output_dim])
        x_ext_val[..., :self.output_dim] = self.scaler.transform(x_ext_val[..., :self.output_dim])
        y_val[..., :self.output_dim] = self.scaler.transform(y_val[..., :self.output_dim])
        x_time_test[..., :self.output_dim] = self.scaler.transform(x_time_test[..., :self.output_dim])
        x_space_test[..., :self.output_dim] = self.scaler.transform(x_space_test[..., :self.output_dim])
        x_ext_test[..., :self.output_dim] = self.scaler.transform(x_ext_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])
        train_data = list(zip(x_time_train, x_space_train, x_ext_train, y_train))
        eval_data = list(zip(x_time_val, x_space_val, x_ext_val, y_val))
        test_data = list(zip(x_time_test, x_space_test, x_ext_test, y_test))
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample)
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader
