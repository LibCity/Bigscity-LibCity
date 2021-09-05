import os
import numpy as np

from libcity.data.dataset import TrafficStatePointDataset


class RESLSTMDataset(TrafficStatePointDataset):

    def __init__(self, config):
        super().__init__(config)
        self.time_lag = self.config.get('time_lag', 6)
        self.TG = self.config.get('TG', 15)  # time interval(min)
        self.TG_in_one_day = self.config.get('TG_in_one_day', 72)  # time interval of one day
        self.TG_in_one_week = self.config.get('TG_in_one_week', 360)  # time interval of one week
        self.parameters_str = \
            str(self.dataset) + '_' + str(self.time_lag) + '_' + str(self.output_window) + '_' \
            + str(self.train_rate) + '_' + str(self.eval_rate) + '_' + str(self.scaler_type) + '_' \
            + str(self.batch_size) + '_' + str(self.load_external) + '_' \
            + str(self.TG) + '_' + str(self.TG_in_one_day) + '_' + str(self.TG_in_one_week)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'point_based_{}.npz'.format(self.parameters_str))

    def _generate_input_data(self, df):
        data = np.array(df)
        X = np.zeros(
            (data.shape[0] - self.time_lag + 1 - self.output_window - self.TG_in_one_week, data.shape[1],
             (self.time_lag - 1) * 3, data.shape[-1]))
        Y = []

        for index in range(self.TG_in_one_week, data.shape[0] - self.time_lag + 1 - self.output_window):
            for i in range(self.num_nodes):
                temp = []
                for j in range(data.shape[-1]):
                    temp.append(data[index - self.TG_in_one_week: index + self.time_lag - 1 - self.TG_in_one_week, i,
                                j].tolist())
                    temp[-1].extend(
                        data[index - self.TG_in_one_day: index + self.time_lag - 1 - self.TG_in_one_day, i, j])
                    temp[-1].extend(data[index: index + self.time_lag - 1, i, j])

                for j in range(data.shape[-1]):
                    for k in range((self.time_lag - 1) * 3):
                        X[index - self.TG_in_one_week, i, k, j] = temp[j][k]

            Y.append(data[index + self.time_lag - 1: index + self.time_lag - 1 + self.output_window, :, :])

        X = X.swapaxes(1, 2)
        Y = np.array(Y)
        return X, Y
