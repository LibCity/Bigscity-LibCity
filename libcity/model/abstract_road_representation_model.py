from libcity.model.abstract_model import AbstractModel


class AbstractRoadRepresentationModel(AbstractModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.config = config
        self.data_feature = data_feature

    def forward(self, batch):
        """
        Args:
            batch: dict, need key 'node_features' contains tensor shape=(N, feature_dim)

        Returns:
            embedding (torch.tensor): shape=(N, output_dim), embedding of input
                                      grad in embedding will be used in training
        """

    def get_data_feature(self):
        """
        此接口返回构造函数中得到的`data_feature`，供Executor类使用，一般不需要继承，不需要修改

        Returns:
            dict: 包含数据集的相关特征的字典, self.data_feature
        """
        return self.data_feature
