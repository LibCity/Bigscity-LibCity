import os
import json
import pandas as pd
import math
import scipy.sparse as sp
from tqdm import tqdm
import random
from logging import getLogger
from libcity.data.dataset.abstract_dataset import AbstractDataset
from libcity.data.list_dataset import ListDataset
from torch.utils.data import DataLoader

parameter_list = ['dataset', 'train_rate']

def random_dict(dict):
    dict_key = list(dict.keys())
    random.shuffle(dict_key)
    new_dict = {}
    for key in dict_key:
        new_dict[key] = dict.get(key)
    return new_dict

class MPRDataset(AbstractDataset):

    def __init__(self, config) -> object:
        self.config = config
        self.cache_filename = './libcity/cache/dataset_cache/MPRData'
        for param in parameter_list:
            if param == 'train_rate':
                self.cache_filename += '_' + str(int(self.config[param] * 100))
            else:
                self.cache_filename += '_' + str(self.config[param])
        self.cache_filename += '.json'
        self.data_path = './raw_data/{}/'.format(self.config['dataset'])
        self.dyna_filename = '{}.dyna'.format(self.config['dataset'])
        self.data = None
        self.adj_mx = None
        self.adjacent_list = None
        self.road_gps = None
        self.road_num = None
        self.traj_num = None
        self._logger = getLogger()

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class)
                test_dataloader: Dataloader composed of Batch (class)
        """
        if self.data is None:
            if os.path.exists(self.cache_filename):
                # load cache
                self._logger.info('use cache data')
                with open(self.cache_filename, 'r') as f:
                    cache_data = json.load(f)
                    self.data = cache_data['data']
                    self.traj_num = cache_data['traj_num']
                    self.load_roadmap()
                    train_data = self.data['train_data']
                    test_data = self.data['test_data']
            else:
                # first load road map infomation
                self.load_roadmap()
                # second read trajectory data
                train_data, test_data = self.load_dyna()
                # cache data
                self.data = {
                    'train_data': train_data,
                    'test_data': test_data,
                }
                # adj_mx cannot save in json
                with open(self.cache_filename, 'w') as f:
                    json.dump({
                        'data': self.data,
                        'traj_num': self.traj_num,
                    }, f)
        self._logger.info('Train data {}, Test data {}'.format(len(train_data), len(test_data)))
        # generate dataloader
        train_dataset = ListDataset(train_data)
        test_dataset = ListDataset(test_data)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.config['batch_size'],
                                      num_workers=self.config['num_workers'], shuffle=False)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.config['batch_size'],
                                      num_workers=self.config['num_workers'], shuffle=False)
        return train_dataloader, test_dataloader

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {
            'loc_num': self.road_num,
            'road_gps': self.road_gps,
            'traj_num': self.traj_num,
            'adj_mx': self.adj_mx,
            'adjacent_list': self.adjacent_list
        }

    def load_roadmap(self):
        # read geo
        road_info = pd.read_csv(os.path.join(self.data_path, '{}.geo'.format(self.config['dataset'])))
        # get road gps
        self.road_gps = {}
        self.road_num = road_info.shape[0]
        for index, row in tqdm(road_info.iterrows(), total=self.road_num, desc='cal road gps dict'):
            rid = row['geo_id']
            coordinate = row['coordinates'].replace('[', '')
            coordinate = coordinate.replace(']', '').split(',')
            lon1 = float(coordinate[0])
            lat1 = float(coordinate[1])
            lon2 = float(coordinate[-2])
            lat2 = float(coordinate[-1])
            center_gps = ((lon1 + lon2) / 2, (lat1 + lat2) / 2)
            self.road_gps[rid] = center_gps

        # read rel to get adjacent matrix
        road_rel = pd.read_csv(os.path.join(self.data_path, '{}.rel'.format(self.config['dataset'])))
        adj_row = []
        adj_col = []
        adj_data = []
        adj_set = set()
        self.adjacent_list = {}
        for index, row in tqdm(road_rel.iterrows(), total=road_rel.shape[0], desc='cal adj mx'):
            f_id = row['origin_id']
            t_id = row['destination_id']
            if (f_id, t_id) not in adj_set:
                adj_set.add((f_id, t_id))
                adj_row.append(f_id)
                adj_col.append(t_id)
                rel_id = row['rel_id']
                adj_data.append(rel_id)
                if rel_id not in self.adjacent_list:
                    self.adjacent_list[rel_id] = []
                else:
                    self.adjacent_list[rel_id].pop()
                self.adjacent_list[rel_id].append(f_id)
                self.adjacent_list[rel_id].append(t_id)
        self.adj_mx = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=(self.road_num, self.road_num), dtype=int)

    def load_dyna(self):
        """
        load dyna file. generate train data and test data
        Returns:
            train_data, test_data (list)
        """
        # first read dyna, and group the trajectory by entity_id and traj_id
        traj = {}
        with open(os.path.join(self.data_path, self.dyna_filename), 'r') as f:
            # read head
            f.readline()
            for line in tqdm(f, desc='group dyna file by entity_id and traj_id'):
                # dyna_id, type, entity_id, traj_id, location id
                items = line.split(',')
                entity_id = items[3]
                traj_id = items[4]
                location_id = int(items[5])
                if (entity_id, traj_id) not in list(traj.keys()):
                    traj[(entity_id, traj_id)] = []
                    traj[(entity_id, traj_id)].append(location_id)
                else:
                    traj[(entity_id, traj_id)].append(location_id)
        # encode trajectory
        train_data = []
        test_data = []
        self.traj_num = len(traj)
        # encode traj
        traj = random_dict(traj)
        new_traj = 0
        for t in tqdm(traj, total=len(traj), desc='encode traj'):
            sub_train_data, sub_test_data = self.encode_traj(new_traj, traj[t])
            train_data.extend(sub_train_data)
            test_data.extend(sub_test_data)
            new_traj += 1
        return train_data, test_data

    def encode_traj(self, new_traj, trajectory):
        """
        Args:
            new_traj (int): the index of trajectory
            trajectory (list): the dict of a trajectory.

        Returns:
            train_data, test_data
        """
        train_data = []
        test_data = []
        train_index = math.floor(self.traj_num * self.config['train_rate'])
        if new_traj < train_index:
            train_data.append(trajectory)
        else:
            # now generate test data.
            test_data.append(trajectory)
        return train_data, test_data
