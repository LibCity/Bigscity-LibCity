import os
import numpy as np

from trafficdl.data.dataset import TrafficStatePointDataset


class ASTGCNDataset(TrafficStatePointDataset):

    def __init__(self, config):
        super().__init__(config)
        self.points_per_hour = self.config.get('points_per_hour', 2)
        self.offset_frame = np.timedelta64(60 // int(self.points_per_hour), 'm')
        self.len_closeness = self.config.get('len_closeness', 3)
        self.len_period = self.config.get('len_period', 4)
        self.len_trend = self.config.get('len_trend', 0)
        assert (self.len_closeness + self.len_period + self.len_trend > 0)
        self.interval_period = self.config.get('interval_period', 1)  # period的长度/天
        self.interval_trend = self.config.get('interval_trend', 7)    # trend的长度/天
        self.feature_name = {'X': 'float', 'y': 'float'}
        self.parameters_str = \
            str(self.dataset) + '_' + str(self.len_closeness) \
            + '_' + str(self.len_period) + '_' + str(self.len_trend) \
            + '_' + str(self.interval_period) + '_' + str(self.interval_trend) \
            + '_' + str(self.output_window) + '_' + str(self.train_rate) \
            + '_' + str(self.eval_rate) + '_' + str(self.scaler_type) \
            + '_' + str(self.batch_size) + '_' + str(self.add_time_in_day) \
            + '_' + str(self.add_day_in_week) + '_' + str(self.pad_with_last_sample)
        self.cache_file_name = os.path.join('./trafficdl/cache/dataset_cache/',
                                            'point_based_{}.npz'.format(self.parameters_str))

    def _search_data(self, sequence_length, label_start_idx, num_for_predict, num_of_depend, units):
        """
        根据全局参数`len_closeness`/`len_period`/`len_trend`找到数据索引的位置
        :param sequence_length: int, 历史数据的总长度
        :param label_start_idx: int, 预测开始的时间片的索引
        :param num_for_predict: int, 预测的时间片序列长度
        :param num_of_depend: int, len_trend/len_period/len_closeness
        :param units: int, trend/period/closeness的长度(以小时为单位)
        :return: 起点-终点区间段的数组，list[(start_idx, end_idx)]
        """
        if self.points_per_hour < 0:
            raise ValueError("points_per_hour should be greater than 0!")
        if label_start_idx + num_for_predict > sequence_length:
            return None
        x_idx = []
        for i in range(1, num_of_depend + 1):
            # 从label_start_idx向左偏移，i是区间数，units*points_per_hour是区间长度(时间片为单位)
            start_idx = label_start_idx - self.points_per_hour * units * i
            end_idx = start_idx + num_for_predict
            if start_idx >= 0:
                x_idx.append((start_idx, end_idx))  # 每一段的长度是num_for_predict
            else:  # i越大越可能有问题，所以遇到错误直接范湖
                return None
        if len(x_idx) != num_of_depend:
            return None
        return x_idx[::-1]  # 倒序，因为原顺序是从右到左，倒序则从左至右

    def _get_sample_indices(self, data_sequence, label_start_idx):
        """
        根据全局参数`len_closeness`/`len_period`/`len_trend`找到数据
        预测目标数据段: [label_start_idx: label_start_idx+output_window)
        :param data_sequence: np.ndarray (len_time, ..., feature_dim)
        :param label_start_idx: the first index of predicting target, 预测开始的时间片的索引
        :return:
        trend_sample: np.ndarray (len_trend * self.output_window, ..., feature_dim)
        period_sample: np.ndarray (len_period * self.output_window, ..., feature_dim)
        closeness_sample: np.ndarray (len_closeness * self.output_window, ..., feature_dim)
        target: np.ndarray (self.output_window, ..., feature_dim)
        """
        trend_sample, period_sample, closeness_sample = None, None, None
        if label_start_idx + self.output_window > data_sequence.shape[0]:
            return trend_sample, period_sample, closeness_sample, None

        if self.len_trend > 0:
            trend_indices = self._search_data(data_sequence.shape[0],  label_start_idx, self.output_window,
                                              self.len_trend, self.interval_trend * 24)
            if not trend_indices:
                return None, None, None, None
            # (len_trend * self.output_window, ..., feature_dim)
            trend_sample = np.concatenate([data_sequence[i: j] for i, j in trend_indices], axis=0)

        if self.len_period > 0:
            period_indices = self._search_data(data_sequence.shape[0], label_start_idx, self.output_window,
                                               self.len_period, self.interval_period * 24)
            if not period_indices:
                return None, None, None, None
            # (len_period * self.output_window, ..., feature_dim)
            period_sample = np.concatenate([data_sequence[i: j] for i, j in period_indices], axis=0)

        if self.len_closeness > 0:
            closeness_indices = self._search_data(data_sequence.shape[0], label_start_idx, self.output_window,
                                                  self.len_closeness, 1)
            if not closeness_indices:
                return None, None, None, None
            # (len_closeness * self.output_window, ..., feature_dim)
            closeness_sample = np.concatenate([data_sequence[i: j] for i, j in closeness_indices], axis=0)

        target = data_sequence[label_start_idx: label_start_idx + self.output_window]
        # (self.output_window, ..., feature_dim)
        return trend_sample, period_sample, closeness_sample, target

    def _generate_input_data(self, df):
        """
        根据全局参数`len_closeness`/`len_period`/`len_trend`切分输入，产生模型需要的输入
        :param df: ndarray (len_time, ..., feature_dim)
        :return:
        # x: (num_samples, Tw+Td+Th, ..., feature_dim)
        # y: (num_samples, Tp, ..., feature_dim)
        """
        trend_samples, period_samples, closeness_samples, targets = [], [], [], []
        for idx in range(df.shape[0]):
            sample = self._get_sample_indices(df, idx)
            if (sample[0] is None) and (sample[1] is None) and (sample[2] is None):
                continue
            trend_sample, period_sample, closeness_sample, target = sample
            if self.len_trend > 0:
                trend_sample = np.expand_dims(trend_sample, axis=0)  # (1,Tw,N,F)
                trend_samples.append(trend_sample)
            if self.len_period > 0:
                period_sample = np.expand_dims(period_sample, axis=0)    # (1,Td,N,F)
                period_samples.append(period_sample)
            if self.len_closeness > 0:
                closeness_sample = np.expand_dims(closeness_sample, axis=0)  # (1,Th,N,F)
                closeness_samples.append(closeness_sample)
            target = np.expand_dims(target, axis=0)  # (1,Tp,N,F)
            targets.append(target)

        sources = []
        if len(closeness_samples) > 0:
            closeness_samples = np.concatenate(closeness_samples, axis=0)  # (num_samples,Th,N,F)
            sources.append(closeness_samples)
            self._logger.info('closeness: ' + str(closeness_samples.shape))
        if len(period_samples) > 0:
            period_samples = np.concatenate(period_samples, axis=0)    # (num_samples,Td,N,F)
            sources.append(period_samples)
            self._logger.info('period: ' + str(period_samples.shape))
        if len(trend_samples) > 0:
            trend_samples = np.concatenate(trend_samples, axis=0)  # (num_samples,Tw,N,F)
            sources.append(trend_samples)
            self._logger.info('trend: ' + str(trend_samples.shape))
        sources = np.concatenate(sources, axis=1)  # (num_samples,Tw+Td+Th,N,F)
        targets = np.concatenate(targets, axis=0)  # (num_samples,Tp,N,F)
        return sources, targets

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
                     feature_dim是输入数据的维度，output_dim是模型输出的维度
        :return: data_feature (dict)
        """
        return {"scaler": self.scaler, "adj_mx": self.adj_mx,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim,
                "len_closeness": self.len_closeness * self.output_window,
                "len_period": self.len_period * self.output_window,
                "len_trend": self.len_trend * self.output_window}
