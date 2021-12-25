import copy
import os
import json
import pandas as pd
import math
import scipy.sparse as sp
from tqdm import tqdm
from datetime import datetime
from geopy import distance
from logging import getLogger
from libcity.data.dataset.abstract_dataset import AbstractDataset
from libcity.data.list_dataset import ListDataset
from libcity.data.batch import Batch
from torch.utils.data import DataLoader

parameter_list = ['dataset', 'train_rate', 'eval_rate', 'max_history_len']


def distance_to_bin(distance_x):
    """
    discrete distance between road
    The bin size is 500 m.
    For distances over 10 km, all are mapped into a same bucket.

    Args:
        distance_x: the unit is meter.

    Returns:
        distance_bin (int)
    """
    if distance_x >= 10000:
        return 20
    else:
        return int(distance_x // 500)


def encode_time(timestamp):
    """
    encode a timestamp to weekday index and hour index
    Args:
        timestamp: the string of time

    Returns:
        weekday_index, hour_index
    """
    date = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ')
    weekday_index = date.weekday()
    hour_index = date.hour
    return weekday_index, hour_index


class NASRDataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        self.cache_filename = './libcity/cache/dataset_cache/nasr_data'
        for param in parameter_list:
            if param == 'train_rate' or param == 'eval_rate':
                self.cache_filename += '_' + str(int(self.config[param] * 100))
            else:
                self.cache_filename += '_' + str(self.config[param])
        self.cache_filename += '.json'
        self.data_path = './raw_data/{}/'.format(self.config['dataset'])
        self.dyna_filename = '{}.dyna'.format(self.config['dataset'])
        self.data = None
        self.adj_mx = None
        self.adjacent_list = None
        self.node_features = None
        self.road_gps = None
        self.max_history_len = self.config['max_history_len']
        self.uid_num = None
        self.road_num = None
        self.train_feature_name = {'uid': 'int', 'current_trace': 'no_tensor', 'history_trace': 'no_tensor',
                                   'candidate_set': 'no_tensor', 'target': 'no_tensor', 'distance': 'no_tensor'}
        self.test_feature_name = {'query': 'no_tensor', 'true_trace': 'no_tensor'}
        self._logger = getLogger()

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class)
                eval_dataloader: Dataloader composed of Batch (class)
                test_dataloader: Dataloader composed of Batch (class)
        """
        if self.data is None:
            if self.config['cache_dataset'] and os.path.exists(self.cache_filename):
                # load cache
                self._logger.info('use cache data')
                with open(self.cache_filename, 'r') as f:
                    cache_data = json.load(f)
                    self.data = cache_data['data']
                    self.uid_num = cache_data['uid_num']
                    self.load_roadmap()
                    train_data = self.data['train_data']
                    eval_data = self.data['eval_data']
                    test_data = self.data['test_data']
            else:
                # first load road map infomation
                self.load_roadmap()
                # second read trajectory data
                train_data, eval_data, test_data = self.load_dyna()
                # cache data
                self.data = {
                    'train_data': train_data,
                    'eval_data': eval_data,
                    'test_data': test_data,
                }
                # node_features and adj_mx cannot save in json
                with open(self.cache_filename, 'w') as f:
                    json.dump({
                        'data': self.data,
                        'uid_num': self.uid_num,
                    }, f)
        self._logger.info('Train data {}, Eval data {}, Test data {}'.format(len(train_data), len(eval_data),
                                                                             len(test_data)))
        # generate dataloader
        train_dataset = ListDataset(train_data)
        eval_dataset = ListDataset(eval_data)
        test_dataset = ListDataset(test_data)

        def collator_train(indices):
            batch = Batch(self.train_feature_name, {}, {})
            for item in indices:
                batch.append(copy.deepcopy(item))
            return batch

        def collator_test(indices):
            batch = Batch(self.test_feature_name, {}, {})
            for item in indices:
                batch.append(copy.deepcopy(item))
            return batch

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.config['batch_size'],
                                      num_workers=self.config['num_workers'], collate_fn=collator_train,
                                      shuffle=False)
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.config['batch_size'],
                                     num_workers=self.config['num_workers'], collate_fn=collator_train,
                                     shuffle=False)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.config['batch_size'],
                                     num_workers=self.config['num_workers'], collate_fn=collator_test,
                                     shuffle=False)
        return train_dataloader, eval_dataloader, test_dataloader

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {
            'uid_num': self.uid_num,
            'loc_num': self.road_num,
            'node_features': self.node_features,
            'adj_mx': self.adj_mx,
            'distance_bins': 21,
            'road_gps': self.road_gps,
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
            lon2 = float(coordinate[2])
            lat2 = float(coordinate[3])
            center_gps = ((lon1 + lon2) / 2, (lat1 + lat2) / 2)
            self.road_gps[rid] = center_gps
        # get node features matrix of roadmap
        na_value = {'lanes': 'unknown', 'bridge': 'no', 'access': 'unknown', 'maxspeed': 120, 'tunnel': 'no',
                    'junction': 'no', 'width': 100}
        encode_feature = ['highway', 'oneway', 'length'] + list(na_value.keys())
        node_features = road_info[encode_feature]
        # fill na
        node_features = node_features.fillna(na_value)
        # apply standardization to continuous feature
        norm_dict = {
            'length': 2,
            'maxspeed': 6,
            'width': 9
        }
        for k, v in norm_dict.items():
            d = node_features[k]
            min_ = d.min()
            max_ = d.max()
            dnew = (d - min_) / (max_ - min_)
            node_features = node_features.drop(labels=k, axis=1)
            node_features.insert(v, k, dnew)
        # apply one hot to discrete feature
        onehot_list = ['highway', 'oneway', 'lanes', 'bridge', 'access', 'tunnel', 'junction']
        for col in onehot_list:
            dum_col = pd.get_dummies(node_features[col], col)
            node_features = node_features.drop(col, axis=1)
            node_features = pd.concat([node_features, dum_col], axis=1)
        self.node_features = node_features.values
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
            f_gps = self.road_gps[f_id]
            if (f_id, t_id) not in adj_set:
                adj_set.add((f_id, t_id))
                adj_row.append(f_id)
                adj_col.append(t_id)
                t_gps = self.road_gps[t_id]
                dis_f_t = distance.distance((t_gps[1], t_gps[0]), (f_gps[1], f_gps[0])).kilometers * 1000
                adj_data.append(distance_to_bin(dis_f_t))
                if f_id not in self.adjacent_list:
                    self.adjacent_list[f_id] = [t_id]
                else:
                    self.adjacent_list[f_id].append(t_id)
        self.adj_mx = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=(self.road_num, self.road_num), dtype=int)

    def load_dyna(self):
        """
        load dyna file. generate train data, eval data and test data
        Returns:
            train_data, eval_data, test_data (list)
        """
        # first read dyna, and group the trajectory by entity_id and traj_id
        user_traj = {}
        with open(os.path.join(self.data_path, self.dyna_filename), 'r') as f:
            # read head
            f.readline()
            for line in tqdm(f, desc='group dyna file by entity_id and traj_id'):
                # dyna_id, type, timestamp, entity_id, traj_id, location id
                items = line.split(',')
                timestamp = items[2]
                entity_id = items[3]
                traj_id = items[4]
                location_id = int(items[5])
                weekday_index, hour_index = encode_time(timestamp)
                if entity_id not in user_traj:
                    user_traj[entity_id] = {traj_id: [[location_id, weekday_index, hour_index]]}
                elif traj_id not in user_traj[entity_id]:
                    user_traj[entity_id][traj_id] = [[location_id, weekday_index, hour_index]]
                else:
                    user_traj[entity_id][traj_id].append([location_id, weekday_index, hour_index])
        # encode trajectory for each user
        train_data = []
        eval_data = []
        test_data = []
        self.uid_num = len(user_traj)
        # encode uid
        new_uid = 0
        for uid in tqdm(user_traj, total=len(user_traj), desc='encode traj for each user'):
            sub_train_data, sub_eval_data, sub_test_data = self.encode_traj(new_uid, user_traj[uid])
            train_data.extend(sub_train_data)
            eval_data.extend(sub_eval_data)
            test_data.extend(sub_test_data)
            new_uid += 1
        return train_data, eval_data, test_data

    def encode_traj(self, uid, trajectory):
        """
        Because we divide data by user, we can divide data when we encode data.
        Args:
            uid (int): the uid of user
            trajectory (dict): the dict of a user's trajectory. {traj_id: list of a trajectory}.

        Returns:
            train_data, eval_data, test_data
        """
        train_data = []
        eval_data = []
        test_data = []
        total_cnt = len(trajectory)
        train_index = math.floor(total_cnt * self.config['train_rate'])
        eval_index = train_index + math.floor(total_cnt * self.config['eval_rate'])
        history_trace = []
        for index, traj_id in enumerate(trajectory):
            current_trace = trajectory[traj_id]
            if index == 0:
                # NASR need history trace
                # so skip the first trace
                history_trace.append(current_trace)
                continue
            elif index < train_index:
                # now generate train data
                # train data will need: uid, current_trace, history_trace, candidate_set, target, distance
                des_road = current_trace[-1][0]
                des_road_gps = self.road_gps[des_road]
                input_data = [uid]
                input_trace = []
                input_candidate_set = []
                input_target = []
                input_distance = []
                for step in range(1, len(current_trace)):
                    # [0, step - 1]
                    current_step_trace = current_trace[:step]
                    current_road = current_step_trace[-1][0]
                    candidate_set = self.adjacent_list[current_road]
                    target_road = current_trace[step][0]
                    candidate_distance = []
                    target = 0
                    for candidate_index, c in enumerate(candidate_set):
                        if c == target_road:
                            target = candidate_index
                        candidate_gps = self.road_gps[c]
                        d = distance.distance((des_road_gps[1], des_road_gps[0]),
                                              (candidate_gps[1], candidate_gps[0])).kilometers * 1000
                        candidate_distance.append(distance_to_bin(d))
                    input_trace.append(current_step_trace)
                    input_candidate_set.append(candidate_set)
                    input_target.append(target)
                    input_distance.append(candidate_distance)
                input_data.append(input_trace)
                input_data.append(copy.deepcopy(history_trace))
                input_data.append(input_candidate_set)
                input_data.append(input_target)
                input_data.append(input_distance)
                train_data.append(input_data)
            elif index < eval_index:
                # now generate eval data.
                # eval data will need: uid, current_trace, history_trace, candidate_set, target, distance
                des_road = current_trace[-1][0]
                des_road_gps = self.road_gps[des_road]
                input_data = [uid]
                input_trace = []
                input_candidate_set = []
                input_target = []
                input_distance = []
                for step in range(1, len(current_trace)):
                    # [0, step - 1]
                    current_step_trace = current_trace[:step]
                    current_road = current_step_trace[-1][0]
                    candidate_set = self.adjacent_list[current_road]
                    target_road = current_trace[step][0]
                    candidate_distance = []
                    target = 0
                    for candidate_index, c in enumerate(candidate_set):
                        if c == target_road:
                            target = candidate_index
                        candidate_gps = self.road_gps[c]
                        d = distance.distance((des_road_gps[1], des_road_gps[0]),
                                              (candidate_gps[1], candidate_gps[0])).kilometers * 1000
                        candidate_distance.append(distance_to_bin(d))
                    input_trace.append(current_step_trace)
                    input_candidate_set.append(candidate_set)
                    input_target.append(target)
                    input_distance.append(candidate_distance)
                input_data.append(input_trace)
                input_data.append(copy.deepcopy(history_trace))
                input_data.append(input_candidate_set)
                input_data.append(input_target)
                input_data.append(input_distance)
                eval_data.append(input_data)
            else:
                # now generate test data.
                # test data will need: query, truth
                # query is composed of (l_s, weekday_s, hour_s, l_d, uid, history_trace)
                l_s = current_trace[0][0]
                weekday_s = current_trace[0][1]
                hour_s = current_trace[0][2]
                l_d = current_trace[-1][0]
                query = (l_s, weekday_s, hour_s, l_d, uid, copy.deepcopy(history_trace))
                truth = [x[0] for x in current_trace]
                test_data.append([query, truth])
            # append history
            history_trace.append(current_trace)
            if len(history_trace) > self.max_history_len:
                # use the latest max_history_len trajectory
                history_trace = history_trace[-self.max_history_len:]
        return train_data, eval_data, test_data
