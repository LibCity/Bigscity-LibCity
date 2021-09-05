import numpy as np
from libcity.data.dataset import TrafficStatePointDataset
# from libcity.data.dataset import TrafficStateGridDataset


"""
主要功能是返回训练数据数组(num_samples, num_nodes)和总的batch数(num_batches)
GTSDataset既可以继承TrafficStatePointDataset，也可以继承TrafficStateGridDataset以处理网格数据
修改成TrafficStateGridDataset时，只需要修改：
1.TrafficStatePointDataset-->TrafficStateGridDataset
2.self.use_row_column = False, 可以加到self.parameters_str中
"""


class GTSDataset(TrafficStatePointDataset):

    def __init__(self, config):
        super().__init__(config)
        self.use_row_column = False

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        datas = self.train_dataloader.dataset  # list of tuple
        # tuple[0] : shape = (input_window, num_nodes(...), feature_dim)
        # tuple[1] : shape = (output_window, num_nodes(...), feature_dim)
        train_datas = [data_tuple[0] for data_tuple in datas]
        # (num_samples, input_window, num_nodes(...), feature_dim)
        train_data = np.array(train_datas)[:, 0, ..., 0]  # (num_samples, num_nodes)
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "ext_dim": self.ext_dim,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "train_data": train_data,
                "num_batches": self.num_batches}
