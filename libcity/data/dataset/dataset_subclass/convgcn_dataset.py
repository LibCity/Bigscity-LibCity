import os
import numpy as np

from libcity.data.dataset import TrafficStatePointDataset


class CONVGCNDataset(TrafficStatePointDataset):

    def __init__(self, config):
        super().__init__(config)
        self.TG = self.config.get('TG', 30)  # time interval(min)
        self.time_lag = self.config.get('time_lag', 11)
        self.TG_in_one_day = self.config.get('TG_in_one_day', 36)  # time interval of one day
        self.TG_in_one_week = self.config.get('TG_in_one_week', 180)  # time interval of one week
        self.parameters_str = \
            str(self.dataset) \
            + '_' + str(self.TG) + '_' + str(self.time_lag) + '_' + str(self.output_window) \
            + '_' + str(self.TG_in_one_day) \
            + '_' + str(self.TG_in_one_week)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'point_based_{}.npz'.format(self.parameters_str))

    def _generate_input_data(self, df):
        # (TG_all,num_nodes, 2)
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
                temp1 = data[
                        index - self.TG_in_one_week: index + self.time_lag - 1 - self.TG_in_one_week,
                        i, 0].tolist()
                temp2 = data[
                        index - self.TG_in_one_week: index + self.time_lag - 1 - self.TG_in_one_week,
                        i, 1].tolist()
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
