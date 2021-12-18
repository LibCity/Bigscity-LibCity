import torch.nn as nn


class AbstractMachineLearningModel(nn.Module):
    """
    The Abstract class of machine learning methods.
    Although the machine learning method does not rely on torch,
    some matrix calculations can still borrow torch to achieve fast calculations on the GPU
    """

    def __init__(self, config, data_feature):
        """

        Args:
            config (ConfigParser): the dict of config
            data_feature (dict): the dict of data feature passed from Dataset.
        """
        nn.Module.__init__(self)

    def predict(self, batch):
        """
        Args:
            batch (Batch): a batch of input

        Returns:
            result (object) : predict result of this batch
        """

    def fit(self, batch):
        """
        use train data to fit the machine learning model

        Args:
            batch (Batch): a batch of input. Generally speaking, the train data of machine learning method
            does not need to be divided into batches, just use Batch to pass in all the data directly.

        Returns:
            return None
        """
