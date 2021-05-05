from trafficdl.config import ConfigParser
from trafficdl.data import get_dataset

# load config
config = ConfigParser('traj_loc_pred', 'ATSTLSTM', 'foursquare_tky', None, None)

dataset = get_dataset(config)
train_data, valid_data, test_data = dataset.get_data()
