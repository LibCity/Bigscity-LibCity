from libcity.data.dataset import TrafficStatePointDataset


class STNORMDataset(TrafficStatePointDataset):
    def __init__(self, config):
        super().__init__(config)