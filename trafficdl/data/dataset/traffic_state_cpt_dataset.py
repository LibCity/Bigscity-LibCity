import os
import numpy as np
import datetime

from trafficdl.data.dataset import TrafficStateDataset
from trafficdl.data.utils import generate_dataloader
from trafficdl.utils import ensure_dir


class TrafficStateCPTDataset(TrafficStateDataset):
    """
    交通状态预测数据集的另一个基类
    部分交通预测模型通过对接近度(closeness)/周期(period)/趋势(trend)进行建模实现预测。
    默认使用`len_closeness`/`len_period`/`len_trend`的数据预测当前时刻的数据，即一个X，一个y。（一般是单步预测）
    **数据原始的时间戳不能为空！**
    一般对外部数据进行单独建模，因此数据为[X, y, X_ext(可选), y_ext(可选)]。
    默认使用`train_rate`和`eval_rate`在样本数量(num_samples)维度上直接切分训练集、测试集、验证集。
    """

    def __init__(self, config):
        super().__init__(config)
        self.points_per_hour = self.config.get('points_per_hour', 2)
        self.offset_frame = np.timedelta64(60 // int(self.points_per_hour), 'm')
        self.len_closeness = self.config.get('len_closeness', 3)
        self.len_period = self.config.get('len_period', 4)
        self.len_trend = self.config.get('len_trend', 0)
        assert (self.len_closeness + self.len_period + self.len_trend > 0)
        self.pad_forward_period = self.config.get('pad_forward_period', 0)
        self.pad_back_period = self.config.get('pad_back_period', 0)
        self.pad_forward_trend = self.config.get('pad_forward_trend', 0)
        self.pad_back_trend = self.config.get('pad_back_trend', 0)
        self.interval_period = self.config.get('interval_period', 1)
        self.interval_trend = self.config.get('interval_trend', 7)
        self.feature_name = {'X': 'float', 'y': 'float', 'X_ext': 'float', 'y_ext': 'float'}

    def _generate_input_data(self, df):
        """
        根据全局参数`len_closeness`/`len_period`/`len_trend`切分输入，产生模型需要的输入

        Structure:
                  |<---------------------- it --------------------->|
                  |                     |<----------- ip ---------->|
           -+--+--+--+--+--+-    -+--+--+--+--+--+-    -+--+--+--+--+-
            |  |  |  |  |  | .... |  |  |  |  |  | .... |  |  |  |  | < Y data
           -+--+--+--+--+--+-    -+--+--+--+--+--+-    -+--+--+--+--+-
            |<bt >|  |< ft>|      |<bp >|  |< fp>|      |<- lc ->|
             ^  ^  ^  ^  ^         ^  ^  ^  ^  ^          ^  ^  ^
            Trend data x lt       Period data x lp     Closeness data

        `interval_period`是period的长度，一般是一天，单位是天
        `interval_trend`是trend的长度，一般是一周，单位是天
        `pad_**`则是向前或向后扩展多长的距离
        用三段的输入一起与预测输出，单步预测
        :param df: ndarray (len_time, ..., feature_dim)
        :return:
        # x: (num_samples, T_c+T_p+T_t, ..., feature_dim)
        # y: (num_samples, 1, ..., feature_dim)
        # ts_x: (num_samples, T_c+T_p+T_t)
        # ts_y: (num_samples, )
        """
        # 求三段相对于预测位置（即y）的偏移距离
        tday = self.points_per_hour * 24  # 每天的时间片数
        r_c = range(1, self.len_closeness + 1)
        rl_p = [range(self.interval_period * tday * i - self.pad_forward_period,
                      self.interval_period * tday * i + self.pad_back_period + 1)
                for i in range(1, self.len_period + 1)]
        rl_t = [range(self.interval_trend * tday * i - self.pad_forward_trend,
                      self.interval_trend * tday * i + self.pad_back_trend + 1)
                for i in range(1, self.len_trend + 1)]
        # print('time index', r_c, rl_p, rl_t)
        offset_mat = \
            [
                [e for e in r_c],
                [e for r_p in rl_p for e in r_p],
                [e for r_t in rl_t for e in r_t]
            ]  # [[closeness], [period], [trend]]
        # 计算最大值，即最大的偏移距离
        largest_interval = max([k[-1] if len(k) != 0 else 0 for k in offset_mat])
        # print('offset_mat', offset_mat)
        # 从最大的偏移处开始算，因为要向前偏移这么多
        x, y, ts_x, ts_y = [], [], [], []
        ts_dumped = []  # 错误时间戳
        for cur_ts in self.timesolts[largest_interval:]:
            # 求当前时间片cur_ts向左偏移offset_mat之后得到的时间片
            # ts_mat和offset_mat形状相同([[closeness], [period], [trend]])，存储的是具体的时间戳
            ts_mat = \
                [
                    [
                        cur_ts - offset * self.offset_frame  # offset_frame 每个时间片的时长（min）
                        for offset in offset_seq
                    ]
                    for offset_seq in offset_mat  # offset_seq: 记录偏移距离的数组
                ]
            # print(ts_mat)
            # 验证时间戳矩阵里的时间戳是否是合法的
            flag = True
            for ts_seq in ts_mat:  # ts_seq: 时间戳数组
                for ts in ts_seq:  # ts: 时间戳
                    if ts not in self.idx_of_timesolts.keys():
                        ts_dumped.append((cur_ts, ts))
                        flag = False
                        break
                if not flag:
                    break
            if not flag:  # 有异常时间戳则进入下一次循环
                continue
            # 获取时间戳矩阵对应位置的数据
            dat_mat = [[df[self.idx_of_timesolts[ts]] for ts in ts_seq] for ts_seq in ts_mat]
            x_c, x_p, x_t = np.array(dat_mat[0]), np.array(dat_mat[1]), np.array(dat_mat[2])
            # (t_c, ..., feature_dim), (t_p, ..., feature_dim), (t_t, ..., feature_dim)
            x_exist = [x_ for x_ in [x_c, x_p, x_t] if len(x_) > 0]
            x_input = np.concatenate(x_exist, axis=0)  # (t_c+t_p+t_t, ..., feature_dim)
            x.append(x_input)
            cur_index = self.idx_of_timesolts[cur_ts]
            y_input = df[cur_index:cur_index+1]  # 预测目标即cur_ts处的数据, (1, ..., feature_dim)
            y.append(y_input)
            ts_x.append(ts_mat[0] + ts_mat[1] + ts_mat[2])  # 对应的时间片
            ts_y.append(cur_ts)                             # 对应的时间片
        x = np.asarray(x)  # (num_samples, T_c+T_p+T_t, ..., feature_dim)
        y = np.asarray(y)  # (num_samples, 1, ..., feature_dim)
        ts_x = np.asarray(ts_x)  # (num_samples, T_c+T_p+T_t)
        ts_y = np.asarray(ts_y)  # (num_samples, )
        self._logger.info("Dumped " + str(len(ts_dumped)) + " data.")
        return x, y, ts_x, ts_y

    def _get_external_array(self, timestamp_list, ext_data=None, previous_ext=False):
        """
        根据时间戳数组，获取对应时间的外部特征
        :param timestamp_list: 时间戳序列
        :param ext_data: 外部数据
        :param previous_ext: 是否是用过去时间段的外部数据，因为对于预测的时间段Y，
                            一般没有真实的外部数据，所以用前一个时刻的数据，**多步预测则用提前多步的数据**
        :return: ndarray (len(timestamp_list), ext_dim)
        """
        data_list = []
        if self.add_time_in_day:
            time_ind = (timestamp_list - timestamp_list.astype("datetime64[D]")) / np.timedelta64(1, "D")
            data_list.append(time_ind.reshape(time_ind.shape[0], 1))
        if self.add_day_in_week:
            dayofweek = []
            for day in timestamp_list.astype("datetime64[D]"):
                dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            day_in_week = np.zeros(shape=(len(timestamp_list), 7))
            day_in_week[np.arange(len(timestamp_list)), dayofweek] = 1
            data_list.append(day_in_week)
        # 外部数据集
        if ext_data is not None:
            indexs = []
            for ts in timestamp_list:
                if previous_ext:
                    ts_index = self.idx_of_ext_timesolts[ts - self.offset_frame]
                else:
                    ts_index = self.idx_of_ext_timesolts[ts]
                indexs.append(ts_index)
            select_data = ext_data[indexs]  # len(timestamp_list) * ext_dim 选出所需要的时间步的数据
            data_list.append(select_data)
        if len(data_list) > 0:
            data = np.concatenate(data_list, axis=1)
        else:
            data = np.zeros((len(timestamp_list), 0))
        return data  # (len(timestamp_list), ext_dim)

    def _load_data(self):
        """
        加载数据文件(.dyna/.grid/.od/.gridod)
        :return:
        x: (num_samples, T_c+T_p+T_t, ..., feature_dim)
        y: (num_samples, 1, ..., feature_dim)
        ts_x: (num_samples, T_c+T_p+T_t)
        ts_y: (num_samples, )
        """
        # 处理多数据文件问题
        if isinstance(self.data_files, list):
            data_files = self.data_files
        else:  # str
            data_files = [self.data_files]
        x_list, y_list, ts_x_list, ts_y_list = [], [], [], []
        for filename in data_files:
            df = self._load_dyna(filename)  # (len_time, ..., feature_dim)
            x, y, ts_x, ts_y = self._generate_input_data(df)
            x_list.append(x)  # x: (num_samples, T_c+T_p+T_t, ..., feature_dim)
            y_list.append(y)  # y: (num_samples, 1, ..., feature_dim)
            ts_x_list.append(ts_x)  # ts_x: (num_samples, T_c+T_p+T_t)
            ts_y_list.append(ts_y)  # ts_y: (num_samples, )
        x = np.concatenate(x_list)  # (num_samples_plus, T_c+T_p+T_t, ..., feature_dim)
        y = np.concatenate(y_list)  # (num_samples_plus, 1, ..., feature_dim)
        ts_x = np.concatenate(ts_x_list)  # (num_samples_plus, T_c+T_p+T_t)
        ts_y = np.concatenate(ts_y_list)  # (num_samples_plus, )
        return x, y, ts_x, ts_y

    def _load_ext_data(self, ts_x, ts_y):
        """
        加载外部数据(.ext)
        :param ts_x: (num_samples, T_c+T_p+T_t)
        :param ts_y: (num_samples, )
        :return:
        ext_x: (num_samples, T_c+T_p+T_t, ext_dim)
        ext_y: (num_samples, ext_dim)
        """
        # 加载外部数据
        if self.load_external and os.path.exists(self.data_path + self.ext_file + '.ext'):  # 外部数据集
            ext_data = self._load_ext()
        else:
            ext_data = None
        ext_x = []
        for ts in ts_x:
            ext_x.append(self._get_external_array(ts, ext_data))
        ext_x = np.asarray(ext_x)
        # ext_x: (num_samples_plus, T_c+T_p+T_t, ext_dim)
        ext_y = self._get_external_array(ts_y, ext_data, previous_ext=True)
        # ext_y: (num_samples_plus, ext_dim)
        return ext_x, ext_y

    def _generate_data(self):
        """
        加载数据文件(.dyna/.grid/.od/.gridod)和外部数据(.ext)
        :return:
        x: (num_samples, T_c+T_p+T_t, ..., feature_dim)
        y: (num_samples, 1, ..., feature_dim)
        ext_x: (num_samples, T_c+T_p+T_t, ext_dim)
        ext_y: (num_samples, ext_dim)
        """
        x, y, ts_x, ts_y = self._load_data()
        ext_x, ext_y = self._load_ext_data(ts_x, ts_y)
        self._logger.info("Dataset created")
        self._logger.info("x shape: " + str(x.shape) + ", y shape: " + str(y.shape))
        self._logger.info("ext_x shape: " + str(ext_x.shape) + ", ext_y shape: " + str(ext_y.shape))
        return x, y, ext_x, ext_y

    def _split_train_val_test(self, x, y, ext_x, ext_y):
        """
        划分训练集、测试集、验证集，并缓存数据集
        :param x: (num_samples, T_c+T_p+T_t, ..., feature_dim)
        :param y: (num_samples, 1, ..., feature_dim)
        :param ext_x: (num_samples, T_c+T_p+T_t, ext_dim)
        :param ext_y: (num_samples, ext_dim)
        :return: x_train, y_train, x_val, y_val, x_test, y_test:
                    (num_samples, input_length, ..., feature_dim)
        """
        test_rate = 1 - self.train_rate - self.eval_rate
        num_samples = x.shape[0]
        num_test = round(num_samples * test_rate)
        num_train = round(num_samples * self.train_rate)
        num_val = num_samples - num_test - num_train

        x_train, x_val, x_test = x[:num_train], x[num_train: num_train + num_val], x[-num_test:]
        y_train, y_val, y_test = y[:num_train], y[num_train: num_train + num_val], y[-num_test:]
        ext_x_train, ext_x_val, ext_x_test = ext_x[:num_train], ext_x[num_train: num_train + num_val], ext_x[-num_test:]
        ext_y_train, ext_y_val, ext_y_test = ext_y[:num_train], ext_y[num_train: num_train + num_val], ext_y[-num_test:]
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ",y: " + str(y_train.shape)
                          + ",x_ext: " + str(ext_x_train.shape) + ",y_ext: " + str(ext_y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ",y: " + str(y_val.shape)
                          + ",x_ext: " + str(ext_x_val.shape) + ",y_ext: " + str(ext_y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ",y: " + str(y_test.shape)
                          + ",x_ext: " + str(ext_x_test.shape) + ",y_ext: " + str(ext_y_test.shape))

        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                self.cache_file_name,
                x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test,
                x_val=x_val, y_val=y_val,
                ext_x_train=ext_x_train, ext_y_train=ext_y_train,
                ext_x_test=ext_x_test, ext_y_test=ext_y_test,
                ext_x_val=ext_x_val, ext_y_val=ext_y_val,
            )
            self._logger.info('Saved at ' + self.cache_file_name)
        return x_train, y_train, x_val, y_val, x_test, y_test, \
            ext_x_train, ext_y_train, ext_x_test, ext_y_test, ext_x_val, ext_y_val

    def _generate_train_val_test(self):
        """
        加载数据集，并划分训练集、测试集、验证集，并缓存数据集
        :return: x_train, y_train, x_val, y_val, x_test, y_test:
                    (num_samples, input_length, ..., feature_dim)
        """
        x, y, ext_x, ext_y = self._generate_data()
        return self._split_train_val_test(x, y, ext_x, ext_y)

    def _load_cache_train_val_test(self):
        """
        加载之前缓存好的训练集、测试集、验证集
        :return: x_train, y_train, x_val, y_val, x_test, y_test: (num_samples, input_length, ..., feature_dim)
        """
        self._logger.info('Loading ' + self.cache_file_name)
        cat_data = np.load(self.cache_file_name)
        x_train = cat_data['x_train']
        y_train = cat_data['y_train']
        x_test = cat_data['x_test']
        y_test = cat_data['y_test']
        x_val = cat_data['x_val']
        y_val = cat_data['y_val']
        ext_x_train = cat_data['ext_x_train']
        ext_y_train = cat_data['ext_y_train']
        ext_x_test = cat_data['ext_x_test']
        ext_y_test = cat_data['ext_y_test']
        ext_x_val = cat_data['ext_x_val']
        ext_y_val = cat_data['ext_y_val']
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ",y: " + str(y_train.shape)
                          + ",x_ext: " + str(ext_x_train.shape) + ",y_ext: " + str(ext_y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ",y: " + str(y_val.shape)
                          + ",x_ext: " + str(ext_x_val.shape) + ",y_ext: " + str(ext_y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ",y: " + str(y_test.shape)
                          + ",x_ext: " + str(ext_x_test.shape) + ",y_ext: " + str(ext_y_test.shape))
        return x_train, y_train, x_val, y_val, x_test, y_test, \
            ext_x_train, ext_y_train, ext_x_test, ext_y_test, ext_x_val, ext_y_val

    def get_data(self):
        """
        获取数据，数据归一化，之后返回训练集、测试集、验证集对应的DataLoader
        :return:
            train_dataloader (pytorch.DataLoader)
            eval_dataloader (pytorch.DataLoader)
            test_dataloader (pytorch.DataLoader)
            all the dataloaders are composed of Batch (class)
        """
        # 加载数据集
        x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
        ext_x_train, ext_y_train, ext_x_test, ext_y_test, ext_x_val, ext_y_val = [], [], [], [], [], []
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train, y_train, x_val, y_val, x_test, y_test,  \
                    ext_x_train, ext_y_train, ext_x_test, ext_y_test, ext_x_val, ext_y_val \
                    = self._load_cache_train_val_test()
            else:
                x_train, y_train, x_val, y_val, x_test, y_test, \
                    ext_x_train, ext_y_train, ext_x_test, ext_y_test, ext_x_val, ext_y_val \
                    = self._generate_train_val_test()
        # 数据归一化
        self.feature_dim = x_train.shape[-1]
        self.scaler = self._get_scalar(x_train, y_train)
        x_train = self.scaler.transform(x_train)
        y_train = self.scaler.transform(y_train)
        x_val = self.scaler.transform(x_val)
        y_val = self.scaler.transform(y_val)
        x_test = self.scaler.transform(x_test)
        y_test = self.scaler.transform(y_test)
        if self.normal_external:
            ext_x_train = self.scaler.transform(ext_x_train)
            ext_y_train = self.scaler.transform(ext_y_train)
            ext_x_val = self.scaler.transform(ext_x_val)
            ext_y_val = self.scaler.transform(ext_y_val)
            ext_x_test = self.scaler.transform(ext_x_test)
            ext_y_test = self.scaler.transform(ext_y_test)
        # 把训练集的X和y聚合在一起成为list，测试集验证集同理
        # x_train/y_train: (num_samples, input_length, ..., feature_dim)
        # train_data(list): train_data[i]是一个元组，由x_train[i]和y_train[i]组成
        train_data = list(zip(x_train, y_train, ext_x_train, ext_y_train))
        eval_data = list(zip(x_val, y_val, ext_x_val, ext_y_val))
        test_data = list(zip(x_test, y_test, ext_x_test, ext_y_test))
        # 转Dataloader
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        """
        由于此类的数据输入包含`len_closeness`/`len_period`/`len_trend`的数据，但都融合到`X`中，
        因此，继承此类的子类此函数应该返回这三段数据的具体长度（不一定等于上述的三个参数的值）
        :return: data_feature (dict)
        """
        raise NotImplementedError('Please implement the function `get_data_feature()`.')

    def _add_external_information(self, df, ext_data=None):
        """
        将外部数据和原始交通状态数据结合到高维数组中，子类必须实现这个方法来指定如何融合外部数据和交通状态数据
        **由于基于`len_closeness`/`len_period`/`len_trend`的方法一般将外部数据单独处理，所以不需要实现此方法。**
        :param df: 交通状态数据多维数组
        :param ext_data: 外部数据
        :return: 融合后的外部数据和交通状态数据
        """
        return df
