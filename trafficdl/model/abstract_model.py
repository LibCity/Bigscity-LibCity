import torch.nn as nn


class AbstractModel(nn.Module):

    def __init__(self, config, data_feature):
        nn.Module.__init__(self)

    def predict(self, batch):
        """
        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: predict result of this batch
        """

    def calculate_loss(self, batch):
        """
        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: return training loss
        """
