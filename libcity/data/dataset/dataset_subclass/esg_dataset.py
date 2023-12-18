import os
import numpy as np

from libcity.data.dataset import TrafficStateGridDataset


class ESGDataset(TrafficStateGridDataset):
    def __init__(self, config):
        super().__init__(config)
        self.static_feat = None

    def _generate_data(self):
        """
        加载数据文件(.dyna/.grid/.od/.gridod)和外部数据(.ext)，且将二者融合，以X，y的形式返回

        Returns:
            tuple: tuple contains:
                x(np.ndarray): 模型输入数据，(num_samples, input_length, ..., feature_dim) \n
                y(np.ndarray): 模型输出数据，(num_samples, output_length, ..., feature_dim)
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
        x_list, y_list = [], []
        for filename in data_files:
            df = self._load_dyna(filename)  # (len_time, ..., feature_dim)
            if self.load_external:
                df = self._add_external_information(df, ext_data)

            x, y = self._generate_input_data(df)
            num_samples = x.shape[0]
            num_train = round(num_samples * self.train_rate)
            self.static_feat = df[:num_train]

            scaler = self._get_scalar(self.scaler_type, self.static_feat, self.static_feat)
            self.static_feat = scaler.transform(self.static_feat)
            self.static_feat = self.static_feat.swapaxes(2, 1)
            self.static_feat = self.static_feat.reshape([-1, self.static_feat.shape[2]])
            # x: (num_samples, input_length, ..., input_dim)
            # y: (num_samples, output_length, ..., output_dim)
            x_list.append(x)
            y_list.append(y)
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        self._logger.info("Dataset created")
        self._logger.info("x shape: " + str(x.shape) + ", y shape: " + str(y.shape))
        return x, y

    def get_data_feature(self):
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "ext_dim": self.ext_dim,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches,
                "static_feat": self.static_feat}
