# import os
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_model
# import pandas as pd


config = ConfigParser('traj_loc_pred', 'STAN', 'foursquare_nyc', other_args={'min_checkins': 3, 'min_sessions': 3,
                                                                             'min_session_len': 4,
                                                                             'cut_method': 'same_date',
                                                                             'max_session_len': 50})
dataset = get_dataset(config)
train_data, valid_data, test_data = dataset.get_data()
data_feature = dataset.get_data_feature()
batch = valid_data.__iter__().__next__()
batch.to_tensor(config['device'])
model = get_model(config, data_feature)
model = model.to(config['device'])
