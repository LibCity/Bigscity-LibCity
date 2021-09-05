from libcity.model.abstract_model import AbstractModel


class AbstractTrafficStateModel(AbstractModel):

    def __init__(self, config, data_feature):
        self.data_feature = data_feature
        super().__init__(config, data_feature)

    def predict(self, batch):
        """
        输入一个batch的数据，返回对应的预测值，一般应该是**多步预测**的结果，一般会调用nn.Moudle的forward()方法

        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: predict result of this batch
        """

    def calculate_loss(self, batch):
        """
        输入一个batch的数据，返回训练过程的loss，也就是需要定义一个loss函数

        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: return training loss
        """

    def get_data_feature(self):
        """
        此接口返回构造函数中得到的`data_feature`，供Executor类使用，一般不需要继承，不需要修改

        Returns:
            dict: 包含数据集的相关特征的字典, self.data_feature
        """
        return self.data_feature
