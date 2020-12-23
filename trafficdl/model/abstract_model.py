import torch.nn as nn

class AbstractModel(nn.Module):

    def __init__(self, config, data_feature):
        nn.Module.__init__(self)
        # raise NotImplementedError('Model not implement')

    def forward(self, batch):
        '''
        Args:
            batch (Batch): a batch of input
        Returns:
            scores (tensor, shape = N*C): the scores of the batch, and will use the score to calculate loss or evaluate.
            N is the number of target
        '''

