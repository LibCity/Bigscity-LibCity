import os
import numpy as np

from trafficdl.data.dataset import TrafficStatePointDataset
from trafficdl.utils import ensure_dir


class CONVGCNDataset(TrafficStatePointDataset):

    def __init__(self, config):
        super().__init__(config)
        self.TG = self.config.get('TG', 30)  # time interval
        self.time_lag = self.config.get('time_lag', 11)
        self.TG_in_one_day = self.config.get('TG_in_one_day', 36)  # time interval of one day
        self.TG_in_one_week = self.config.get('TG_in_one_week', 180)  # time interval of one week
        self.parameters_str = \
            str(self.dataset) \
            + '_' + str(self.TG) + '_' + str(self.time_lag) + '_' + str(self.output_window) \
            + '_' + str(self.TG_in_one_day) \
            + '_' + str(self.TG_in_one_week)
        self.cache_file_name = os.path.join('./trafficdl/cache/dataset_cache/',
                                            'point_based_{}.npz'.format(self.parameters_str))

    def _generate_input_data(self, df):
        """
        根据全局参数len_closeness/len_period/len_trend切分输入，产生模型需要的输入

        Args:
            df(np.ndarray): 输入数据, shape: (len_time, ..., feature_dim)

        Returns:
            tuple: tuple contains:
                sources(np.ndarray): 模型输入数据, shape: (num_samples, Tw+Td+Th, ..., feature_dim) \n
                targets(np.ndarray): 模型输出数据, shape: (num_samples, Tp, ..., feature_dim)
        """
        # (TG_all,num_nodes, 2)  2 for inflow/outflow
        data = np.array(df)

        Xin = np.zeros(
            (data.shape[0] - self.time_lag + 1 - self.output_window - self.TG_in_one_week, data.shape[1],
             (self.time_lag - 1) * 3))
        Xout = np.zeros(
            (data.shape[0] - self.time_lag + 1 - self.output_window - self.TG_in_one_week, data.shape[1],
             (self.time_lag - 1) * 3))
        Yin = []
        Yout = []

        for index in range(self.TG_in_one_week, data.shape[0] - self.time_lag + 1 - self.output_window):
            for i in range(self.num_nodes):
                temp1 = data[index - self.TG_in_one_week: index + self.time_lag - 1 - self.TG_in_one_week, i,
                        0].tolist()
                temp2 = data[index - self.TG_in_one_week: index + self.time_lag - 1 - self.TG_in_one_week, i,
                        1].tolist()
                temp1.extend(data[index - self.TG_in_one_day: index + self.time_lag - 1 - self.TG_in_one_day, i, 0])
                temp2.extend(data[index - self.TG_in_one_day: index + self.time_lag - 1 - self.TG_in_one_day, i, 1])
                temp1.extend(data[index: index + self.time_lag - 1, i, 0])
                temp2.extend(data[index: index + self.time_lag - 1, i, 1])
                # Xin[index - self.TG_in_one_week].append(temp1)
                for k in range(len(temp1)):
                    Xin[index - self.TG_in_one_week, i, k] = temp1[k]
                    Xout[index - self.TG_in_one_week, i, k] = temp2[k]
                # Xout[index - self.TG_in_one_week].append(temp2)
            Yin.append(data[index + self.time_lag - 1: index + self.time_lag - 1 + self.output_window, :, 0])  # input
            Yout.append(data[index + self.time_lag - 1: index + self.time_lag - 1 + self.output_window, :, 0])  # output

        X = np.concatenate((Xin[:, :, :, np.newaxis], Xout[:, :, :, np.newaxis]), axis=3)
        Yin = np.array(Yin)
        Yout = np.array(Yout)

        Y = np.concatenate((Yin[:, :, :, np.newaxis], Yout[:, :, :, np.newaxis]), axis=3)
        X = X.swapaxes(1, 2)
        return X, Y

    def _split_train_val_test(self, x, y):

        test_rate = 1 - self.train_rate - self.eval_rate
        num_samples = x.shape[0]
        num_test = round(num_samples * test_rate)
        num_train = round(num_samples * self.train_rate)
        num_val = num_samples - num_test - num_train

        # train
        x_train, y_train = x[:num_train], y[:num_train]
        # val
        x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
        # test
        x_test, y_test = x[-num_test:], y[-num_test:]
        self._logger.info("train\t" + "x: " + str(x_train.shape) + "y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + "y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + "y: " + str(y_test.shape))

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
