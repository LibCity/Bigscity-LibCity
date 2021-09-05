from libcity.model.abstract_model import AbstractModel


class TemplateTLP(AbstractModel):
    """
    请参考开源模型代码，完成本文件的编写。请务必重写 __init__, predict, calculate_loss 三个方法。
    """

    def __init__(self, config, data_feature):
        """
        参数说明：
            config (dict): 配置模块根据模型对应的 config.json 文件与命令行传递的参数
                根据 config 初始化模型参数
            data_feature (dict): 在数据预处理步骤提取到的数据集所属的特征参数，如 loc_size，uid_size 等。
        """

    def predict(self, batch):
        """
        参数说明:
            batch (libcity.data.batch): 类 dict 文件，其中包含的键值参见任务说明文件。
        返回值:
            score (pytorch.tensor): 对应张量 shape 应为 batch_size *
                loc_size。这里返回的是模型对于输入当前轨迹的下一跳位置的预测值。
        """

    def calculate_loss(self, batch):
        """
        参数说明:
            batch (libcity.data.batch): 类 dict 文件，其中包含的键值参见任务说明文件。
        返回值:
            loss (pytorch.tensor): 可以调用 pytorch 实现的 loss 函数与 batch['target']
                目标值进行 loss 计算，并将计算结果返回。如模型有自己独特的 loss 计算方式则自行参考实现。
        """
