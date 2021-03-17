import os

from trafficdl.data.dataset import TrafficStateDataset


class TrafficStatePointDataset(TrafficStateDataset):

    def __init__(self, config):
        super().__init__(config)
        self.cache_file_name = os.path.join('./trafficdl/cache/dataset_cache/',
                                            'point_based_{}.npz'.format(self.parameters_str))

    def _load_geo(self):
        """
        加载.geo文件，格式[geo_id, type, coordinates, properties(若干列)]
        :return:
        """
        super()._load_geo()

    def _load_rel(self):
        """
        加载.rel文件，格式[rel_id, type, origin_id, destination_id, properties(若干列)]
        生成N*N的矩阵，其中权重所在的列名用全局参数`weight_col`来指定
        .rel文件中缺少的位置的权重填充为np.inf
        全局参数`calculate_weight`表示是否需要对加载的.rel的默认权重进行进一步计算，
        如果需要，则调用函数_calculate_adjacency_matrix()进行计算
        :return: N*N的邻接矩阵 self.adj_mx
        """
        super()._load_rel()

    def _load_dyna(self, filename):
        """
        加载.dyna文件，格式[dyna_id, type, time, entity_id, properties(若干列)]
        .geo文件中的id顺序应该跟.dyna中一致
        其中全局参数`data_col`用于指定需要加载的数据的列，不设置则默认全部加载
        :return: 3d-array (len_time, num_nodes, feature_dim)
        """
        return super()._load_dyna_3d(filename)

    def _add_external_information(self, df, ext_data=None):
        """
        增加外部信息（一周中的星期几/day of week，一天中的某个时刻/time of day，外部数据）
        :param df: ndarray (len_time, num_nodes, feature_dim)
        :return: data: ndarray (len_time, num_nodes, feature_dim_plus)
        """
        return super()._add_external_information_3d(df, ext_data)

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
                     feature_dim是输入数据的维度，output_dim是模型输出的维度
        :return: data_feature (dict)
        """
        return {"scaler": self.scaler, "adj_mx": self.adj_mx,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim}
