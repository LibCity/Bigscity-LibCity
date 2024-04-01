import os

from libcity.data.dataset import TrafficStatePointDataset
from libcity.data.utils import generate_dataloader


class RGSLDataset(TrafficStatePointDataset):
    def __init__(self, config):
        super().__init__(config)

    def get_data_feature(self):
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "num_nodes": self.num_nodes,
                "output_window": self.output_window, "output_dim": self.output_dim, "num_batches": self.num_batches}
