import os
import torch
from trafficdl.config import ConfigParser
from trafficdl.data import get_dataset
from trafficdl.utils import get_executor, get_model
from trafficdl.utils.dataset import parseCoordinate
from geopy import distance
import numpy as np
import torch.nn.functional as F

config = ConfigParser('traj_loc_pred', 'SERM', 'foursquare_tky', None, {"dataset_class": 'SermTrajectoryDataset'})
dataset = get_dataset(config)
train_data, valid_data, test_data = dataset.get_data()
batch = valid_data.__iter__().__next__()
batch.to_tensor(config['device'])
