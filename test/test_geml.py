import numpy as np
import torch

from libcity.model.traffic_od_prediction.GEML import GEML

config = {
    "input_window": 12,
    "output_window": 12,
    "embed_dim": 10,
    "batch_size": 2,
}

data_feature = {
    "num_nodes": 4,
    "adj_mx": np.random.rand(4, 4)
}

batch = {
    'X': torch.randn((2, 12, 4, 4, 1)),
    'y': torch.randn((2, 12, 4, 4, 1))
}

model = GEML(config, data_feature)

model.calculate_loss(batch)
