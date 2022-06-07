import numpy as np

from libcity.data.dataset import TrafficStateOdDataset


class ODCRNDataset(TrafficStateOdDataset):
    def __init__(self, config):
        super().__init__(config)

    def _load_dyna(self, filename):
        data = super(ODCRNDataset, self)._load_dyna(filename)
        data = np.log(data + 1)
        return data
