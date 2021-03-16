import os

from trafficdl.data.dataset import TrafficStateDataset


class TrafficStateGridOdDataset(TrafficStateDataset):

    def __init__(self, config):
        super().__init__(config)
        self.use_row_column = self.config.get('use_row_column', True)
        self.parameters_str = self.parameters_str + '_' + str(self.use_row_column)
        self.cache_file_name = os.path.join('./trafficdl/cache/dataset_cache/',
                                            'grid_od_based_{}.npz'.format(self.parameters_str))
        self._load_rel()  # don't care whether there is a .rel file

    def _load_geo(self):
        """
        加载.geo文件，格式[geo_id, type, coordinates, row_id, column_id, properties(若干列)]
        :return:
        """
        super()._load_grid_geo()

    def _load_rel(self):
        """
        根据网格结构构建邻接矩阵，一个格子跟他周围的8个格子邻接
        :return: N*N的邻接矩阵 self.adj_mx
        """
        super()._load_grid_rel()

    def _load_dyna(self, filename):
        """
        加载.gridod文件，格式[dyna_id, type, time, origin_row_id, origin_column_id,
                            destination_row_id, destination_column_id, properties(若干列)]
        .geo文件中的id顺序应该跟.dyna中一致
        其中全局参数`data_col`用于指定需要加载的数据的列，不设置则默认全部加载
        根据参数`use_row_column`确定转成4d还是6d的数组，True为6d
        :return: 4d-array or 6d-array (len_time, num_grids, num_grids, feature_dim)
                    / (len_time, len_row, len_column, len_row, len_column, feature_dim)
        """
        if self.use_row_column:
            return super()._load_grid_od_6d(filename)
        else:
            return super()._load_grid_od_4d(filename)

    def _add_external_information(self, df, ext_data=None):
        """
        增加外部信息（一周中的星期几/day of week，一天中的某个时刻/time of day，外部数据）
        根据参数`use_row_column`确定是4d还是6d的数组，True为6d
        :param df: ndarray (len_time, ..., feature_dim)
        :return: data: ndarray (len_time, ..., feature_dim_plus)
        """
        if self.use_row_column:
            return super()._add_external_information_6d(df, ext_data)
        else:
            return super()._add_external_information_4d(df, ext_data)

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是网格的个数，
                      len_row是网格的行数，len_column是网格的列数，
                     feature_dim是输入数据的维度，output_dim是模型输出的维度
        :return: data_feature (dict)
        """
        return {"scaler": self.scaler, "adj_mx": self.adj_mx,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "len_row": self.len_row, "len_column": self.len_column}
