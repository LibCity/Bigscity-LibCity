import os
import torch
from trafficdl.config import ConfigParser
from trafficdl.data import get_dataset
from trafficdl.utils import get_executor, get_model

config = ConfigParser('traj_loc_pred', 'LSTPM', 'foursquare_tky', None, {"history_type": 'cut_off'})
dataset = get_dataset(config)
train_data, valid_data, test_data = dataset.get_data()
data_feature = dataset.get_data_feature()
batch = valid_data.__iter__().__next__()
model = get_model(config, data_feature)
self = model.to(config['device'])
batch.to_tensor(config['device'])
logp_seq = self.forward(batch, False)
executor = get_executor(config, model)
