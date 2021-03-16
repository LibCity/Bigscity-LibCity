
class AbstractDataset(object):

    def __init__(self, config):
        raise NotImplementedError("Dataset not implemented")

    def get_data(self):
        """
        return:
            train_dataloader (pytorch.DataLoader)
            eval_dataloader (pytorch.DataLoader)
            test_dataloader (pytorch.DataLoader)
            all the dataloaders are composed of Batch (class)
        """
        raise NotImplementedError("get_data not implemented")

    def get_data_feature(self):
        """
        如果模型使用了 embedding 层，一般是需要根据数据集的 loc_size、tim_size、uid_size 等特征来确定
        embedding 层的大小的
        故该方法返回一个 dict，包含表示层能够提供的数据集特征
        return:
            data_feature (dict)
        """
        raise NotImplementedError("get_data_feature not implemented")
