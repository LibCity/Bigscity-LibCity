import os

from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_executor, get_model

task = 'traj_loc_pred'
model = 'DeepMove'
dataset = 'foursquare_tky'

other_args = {
    'batch_size': 1
}
config = ConfigParser(task, model, dataset, config_file=None, other_args=other_args)
dataset = get_dataset(config)
train_data, valid_data, test_data = dataset.get_data()
