import os

from libcity.data.dataset import TrafficStateDataset


class TrafficStatePointDataset(TrafficStateDataset):

    def __init__(self, config):
        super().__init__(config)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'point_based_{}.npz'.format(self.parameters_str))

    def _load_geo(self):
        """
        加载.geo文件，格式[geo_id, type, coordinates, properties(若干列)]
        """
        super()._load_geo()

    def _load_rel(self):
        """
        加载.rel文件，格式[rel_id, type, origin_id, destination_id, properties(若干列)]

        Returns:
            np.ndarray: self.adj_mx, N*N的邻接矩阵
        """
        super()._load_rel()

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
