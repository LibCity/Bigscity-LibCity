import os
import numpy as np

from libcity.data.utils import generate_dataloader
from libcity.utils import ensure_dir
from libcity.data.dataset import TrafficStateDataset


class TrafficImputeDataset(TrafficStateDataset):

    def __init__(self, config):
        self.missing_pattern = config.get("missing_pattern", "point")
        self.missing_ratio = config.get("missing_ratio", None)
        super().__init__(config)
        self.feature_name = {'X': 'float', 'y': 'float', 'mask': 'int'}

    def _load_dyna(self, filename):
        """
        加载.dyna文件，格式[dyna_id, type, time, entity_id, properties(若干列)]
        其中全局参数`data_col`用于指定需要加载的数据的列，不设置则默认全部加载

        Args:
            filename(str): 数据文件名，不包含后缀

        Returns:
            np.ndarray: 数据数组, 3d-array (len_time, num_nodes, feature_dim)
        """
        return super()._load_dyna_3d(filename)

    def _add_external_information(self, df, ext_data=None):
        """
        增加外部信息（一周中的星期几/day of week，一天中的某个时刻/time of day，外部数据）

        Args:
            df(np.ndarray): 交通状态数据多维数组, (len_time, num_nodes, feature_dim)
            ext_data(np.ndarray): 外部数据

        Returns:
            np.ndarray: 融合后的外部数据和交通状态数据, (len_time, num_nodes, feature_dim_plus)
        """
        return super()._add_external_information_3d(df, ext_data)

    def sample_mask(self, shape, p=0.0015, p_noise=0.05, max_seq=1, min_seq=1, rng=None):
        if rng is None:
            rand = np.random.random
            randint = np.random.randint
        else:
            rand = rng.random
            randint = rng.integers
        mask = rand(shape) < p
        # [samples, len_time, num_nodes, dim]
        if self.missing_pattern != "point":
            for sample in range(mask.shape[0]):
                for col in range(mask.shape[2]):
                    # 不为0的mask索引
                    idxs = np.flatnonzero(mask[sample, :, col, :])
                    if not len(idxs):
                        continue
                    fault_len = min_seq
                    if max_seq > min_seq:
                        fault_len = fault_len + int(randint(max_seq - min_seq))
                    # len(idxs) * fault_len
                    idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
                    # 去除其中重复的元素 ，并按元素 由小到大 返回一个新的无元素重复的元组或者列表。
                    idxs = np.unique(idxs_ext)
                    # 截取数组中小于或者大于某值的部分，并使得被截取部分等于固定值。
                    idxs = np.clip(idxs, 0, shape[1] - 1)
                    mask[sample, idxs, col, :] = True
        mask = mask | (rand(mask.shape) < p_noise)
        return mask.astype('uint8')

    def _generate_input_data(self, df):
        """
        根据全局参数`input_window`和`output_window`切分输入，产生模型需要的张量输入，
        即使用过去`input_window`长度的时间序列去预测未来`output_window`长度的时间序列

        Args:
            df(np.ndarray): 数据数组，shape: (len_time, ..., feature_dim)

        Returns:
            tuple: tuple contains:
                x(np.ndarray): 模型输入数据，(epoch_size, input_length, ..., feature_dim) \n
                y(np.ndarray): 模型输出数据，(epoch_size, output_length, ..., feature_dim)
        """
        num_samples = df.shape[0]
        # 预测用的过去时间窗口长度 取决于self.input_window
        x_offsets = np.sort(np.concatenate((np.arange(-self.input_window + 1, 1, 1),)))

        x, y = [], []
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples)
        for t in range(min_t, max_t):
            x_t = df[t + x_offsets, ...]
            x.append(x_t)
        x = np.stack(x, axis=0)
        y = np.copy(x)

        return x, y

    def _load_cache_train_val_test(self):
        """
        加载之前缓存好的训练集、测试集、验证集

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        self._logger.info('Loading ' + self.cache_file_name)
        cat_data = np.load(self.cache_file_name)
        x_train = cat_data['x_train']
        y_train = cat_data['y_train']
        mask_train = cat_data['mask_train']
        x_test = cat_data['x_test']
        y_test = cat_data['y_test']
        mask_test = cat_data['mask_test']
        x_val = cat_data['x_val']
        y_val = cat_data['y_val']
        mask_val = cat_data['mask_val']
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))
        return x_train, y_train, mask_train, x_val, y_val, mask_val, x_test, y_test, mask_test

    def _split_train_val_test(self, x, y):
        """
        划分训练集、测试集、验证集，并缓存数据集

        Args:
            x(np.ndarray): 输入数据 (num_samples, input_length, ..., feature_dim)
            y(np.ndarray): 输出数据 (num_samples, input_length, ..., feature_dim)

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        test_rate = 1 - self.train_rate - self.eval_rate
        num_samples = x.shape[0]
        num_test = round(num_samples * test_rate)
        num_train = round(num_samples * self.train_rate)
        num_val = num_samples - num_test - num_train

        SEED = 9101112
        rng = np.random.default_rng(SEED)

        if self.missing_pattern == 'block':
            eval_mask = self.sample_mask(shape=x.shape, p=0.0015, p_noise=0.05, min_seq=self.input_window // 2,
                                         max_seq=self.input_window * 4,
                                         rng=rng)
            if self.missing_ratio is not None:
                eval_mask = self.sample_mask(shape=x.shape, p=self.missing_ratio,
                                             p_noise=0.05, min_seq=self.input_window // 2,
                                             max_seq=self.input_window * 4,
                                             rng=rng)
        elif self.missing_pattern == 'point':
            eval_mask = self.sample_mask(shape=x.shape, p=0., p_noise=0.25, max_seq=self.input_window // 2,
                                         min_seq=self.input_window * 4,
                                         rng=rng)
        # train
        x_train, y_train, mask_train = x[:num_train], y[:num_train], eval_mask[:num_train]
        # val
        x_val, y_val, mask_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val], \
            eval_mask[num_train: num_train + num_val]
        # test
        x_test, y_test, mask_test = x[-num_test:], y[-num_test:], eval_mask[-num_test:]
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))

        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                self.cache_file_name,
                x_train=x_train,
                y_train=y_train,
                mask_train=mask_train,
                x_test=x_test,
                y_test=y_test,
                mask_test=mask_test,
                x_val=x_val,
                y_val=y_val,
                mask_val=mask_val,
            )
            self._logger.info('Saved at ' + self.cache_file_name)
        return x_train, y_train, mask_train, x_val, y_val, mask_val, x_test, y_test, mask_test

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        # 加载数据集
        x_train, y_train, mask_train, x_val, y_val, mask_val, x_test, y_test, mask_test = [], [], [], [], [], [], [], [], []
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train, y_train, mask_train, x_val, y_val, mask_val, x_test, y_test, mask_test = self._load_cache_train_val_test()
            else:
                x_train, y_train, mask_train, x_val, y_val, mask_val, x_test, y_test, mask_test = self._generate_train_val_test()

        # 数据归一化
        self.feature_dim = x_train.shape[-1]
        self.ext_dim = self.feature_dim - self.output_dim
        self.scaler = self._get_scalar(self.scaler_type,
                                       x_train[..., :self.output_dim], y_train[..., :self.output_dim])
        self.ext_scaler = self._get_scalar(self.ext_scaler_type,
                                           x_train[..., self.output_dim:], y_train[..., self.output_dim:])
        x_train[..., :self.output_dim] = x_train[..., :self.output_dim] * (1 - mask_train)[..., :self.output_dim]
        x_val[..., :self.output_dim] = x_val[..., :self.output_dim] * (1 - mask_val)[..., :self.output_dim]
        x_test[..., :self.output_dim] = x_test[..., :self.output_dim] * (1 - mask_test)[..., :self.output_dim]
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
        # 把训练集的X和y聚合在一起成为list，测试集验证集同理
        # x_train/y_train: (num_samples, input_length, ..., feature_dim)
        # train_data(list): train_data[i]是一个元组，由x_train[i]和y_train[i]组成
        train_data = list(zip(x_train, y_train, mask_train))
        eval_data = list(zip(x_val, y_val, mask_val))
        test_data = list(zip(x_test, y_test, mask_test))
        # 转Dataloader
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample)
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "ext_dim": self.ext_dim,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches}


if __name__ == '__main__':
    SEED = 9101112
    rng = np.random.default_rng(SEED)
    # sample_mask(shape=(14324, 12, 200, 1), p=0., p_noise=0.25, max_seq=12 // 2,
    #             min_seq=12 * 4,
    #             rng=rng)
