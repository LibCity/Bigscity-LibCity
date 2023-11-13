import os
import pickle
import numpy as np

from libcity.config import ConfigParser
from libcity.data.dataset import TrafficStatePointDataset
from libcity.data.utils import generate_dataloader


class STGODEDataset(TrafficStatePointDataset):
    def __init__(self, config):
        super().__init__(config)
        self.load_from_local = self.config.get('load_from_local', True)
        cache_path = './libcity/cache/dataset_cache/dtw_distance_index_' + self.dataset + '.npz'
        # if self.load_from_local and os.path.exists(cache_path):
        #     with open(cache_path, 'rb') as f:
        #         self.dist_matrix = pickle.load(f)
        # else:
        #     self.dist_matrix = self.get_dtw_matrix()
        #     with open(cache_path, 'wb') as f:
        #         pickle.dump(self.dist_matrix, f)

    def get_dtw_matrix(self):
        i = 0
        for filename in self.data_files:
            if i == 0:
                df = self._load_dyna(filename)  # (len_time, node_num, feature_dim)
            else:
                df = np.concatenate((df, self._load_dyna(filename)), axis=0)
            i += 1
        print(df)



if __name__ == '__main__':
    config = ConfigParser(task='traffic_state_pred', model='RNN',
                          dataset='PEMSD7(M)', other_args={'batch_size': 2})
    s = STGODEDataset(config)
    s.get_dtw_matrix()
