import os
import json
import pandas as pd
import math
import importlib
import numpy as np
from logging import getLogger
from collections import Counter
from tqdm import tqdm
from libcity.data.dataset import AbstractDataset
from libcity.utils import parse_time, cal_basetime, cal_timeoff
from libcity.data.utils import generate_dataloader_pad


class PBSTrajectoryDataset(AbstractDataset):
    """popularity based negative sampling
    weighted random sampling based on np.random.choice
    """

    def __init__(self, config):
        self.config = config
        self.cache_file_folder = './libcity/cache/dataset_cache/'
        self.dataset = self.config.get('dataset', '')
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        self.data_path = './raw_data/{}/'.format(self.dataset)
        self.data = None
        # 加载 encoder
        self.encoder = self.get_encoder()
        self.neg_samples = self.config['neg_samples']
        self.pad_item = None  # 因为若是使用缓存, pad_item 是记录在缓存文件中的而不是 encoder
        self.logger = getLogger()
        self.counter = Counter()

    def get_data(self):
        """
        轨迹比较特殊，原子文件中存储的并不是轨迹而是一个一个点，因此需要先对轨迹进行切割
        """
        if self.data is None:
            if self.config['cache_dataset'] and os.path.exists(self.encoder.cache_file_name):
                # load cache
                f = open(self.encoder.cache_file_name, 'r')
                self.data = json.load(f)
                self.pad_item = self.data['pad_item']
                f.close()
            else:
                cut_data = self.cutter_filter()
                encoded_data = self.encode_traj(cut_data)
                self.data = encoded_data
                self.pad_item = self.encoder.pad_item
                if self.config['cache_dataset']:
                    if not os.path.exists(self.cache_file_folder):
                        os.makedirs(self.cache_file_folder)
                    with open(self.encoder.cache_file_name, 'w') as f:
                        json.dump(encoded_data, f)
        # user 来划，以及按轨迹数来划。
        # TODO: 这里可以设一个参数，现在先按照轨迹数来划吧
        train_data, eval_data, test_data = self.divide_data()
        return generate_dataloader_pad(train_data, eval_data, test_data,
                                       self.encoder.feature_dict,
                                       self.config['batch_size'],
                                       self.config['num_workers'], self.pad_item,
                                       self.encoder.feature_max_len)

    def get_data_feature(self):
        res = self.data['data_feature']
        res['distance_upper'] = self.config['distance_upper']
        return res

    def cutter_filter(self):
        """
        切割后的轨迹存储格式: (dict)
            {
                uid: [
                    [
                        checkin_record,
                        checkin_record,
                        ...
                    ],
                    [
                        checkin_record,
                        checkin_record,
                        ...
                    ],
                    ...
                ],
                ...
            }
        """
        # load data according to config
        traj = pd.read_csv(os.path.join(
            self.data_path, '{}.dyna'.format(self.dyna_file)))
        # filter inactive poi
        group_location = traj.groupby('location').count()
        filter_location = group_location[group_location['time'] > self.config['min_checkins']]
        location_index = filter_location.index.tolist()
        traj = traj[traj['location'].isin(location_index)]

        user_set = pd.unique(traj['entity_id'])
        res = {}
        min_session_len = self.config['min_session_len']
        min_sessions = self.config['min_sessions']
        window_size = self.config['window_size']
        window_type = self.config['window_type']
        if window_type == 'time_window':
            # 按照时间窗口进行切割
            base_zero = window_size > 12
            for uid in tqdm(user_set, desc="cut and filter trajectory"):
                usr_traj = traj[traj['entity_id'] == uid]
                sessions = []  # 存放该用户所有的 session
                session = []  # 单条轨迹
                # 这里还是使用当地时间吧
                start_time = parse_time(usr_traj.iloc[0]['time'])
                base_time = cal_basetime(start_time, base_zero)
                for index, row in usr_traj.iterrows():
                    if index == 0:
                        assert start_time.hour - base_time.hour < window_size
                        session.append(row.tolist())
                    else:
                        now_time = parse_time(row['time'])
                        time_off = cal_timeoff(now_time, base_time)
                        if time_off < window_size and time_off >= 0:
                            session.append(row.tolist())
                        else:
                            if len(session) >= min_session_len:
                                sessions.append(session)
                            session = []
                            start_time = now_time
                            base_time = cal_basetime(start_time, base_zero)
                            session.append(row.tolist())
                if len(session) >= min_session_len:
                    sessions.append(session)
                if len(sessions) >= min_sessions:
                    # update counter
                    for s in sessions:
                        for row in s:
                            self.counter.update({row[4]: 1})
                    res[str(uid)] = sessions
        else:
            # 按照轨迹长度进行划分
            for uid in tqdm(user_set, desc="cut and filter trajectory"):
                usr_traj = traj[traj['entity_id'] == uid]
                sessions = []  # 存放该用户所有的 session
                session = []  # 单条轨迹
                for index, row in usr_traj.iterrows():
                    if len(session) < window_size:
                        session.append(row.tolist())
                    else:
                        sessions.append(session)
                        session = []
                        session.append(row.tolist())
                if len(session) >= min_session_len:
                    sessions.append(session)
                if len(sessions) >= min_sessions:
                    # update counter
                    for s in sessions:
                        for row in s:
                            self.counter.update({row[4]: 1})
                    res[str(uid)] = sessions
        return res

    def encode_traj(self, data):
        """encode the cut trajectory

        Args:
            data (dict): the key is uid, the value is the uid's trajectories. For example:
                {
                    uid: [
                        trajectory1,
                        trajectory2
                    ]
                }
                trajectory1 = [
                    dyna_id,
                    dyna_id,
                    .....
                ]

        Return:
            dict: For example:
                {
                    data_feature: {...},
                    pad_item: {...},
                    encoded_data: {uid: encoded_trajectories}
                }
        """
        # build sample weight
        popularity = []
        weight = []
        for key, cnt in tqdm(self.counter.most_common(), desc='calculating popularity weight'):
            popularity.append(key)
            weight.append(cnt)
        popularity = np.array(popularity)
        weight = np.array(weight) / np.sum(weight)

        encoded_data = {}
        for uid in tqdm(data, desc='encoding trajectory'):
            # generate negative samples
            total_negative_samples = []
            for i in range(len(data[uid])):
                neg = np.random.choice(popularity, size=self.neg_samples, replace=False, p=weight)
                total_negative_samples.append(neg)
            encoded_data[uid] = self.encoder.encode(int(uid), data[uid], total_negative_samples)
        self.encoder.gen_data_feature()
        return {
            'data_feature': self.encoder.data_feature,
            'pad_item': self.encoder.pad_item,
            'encoded_data': encoded_data
        }

    def divide_data(self):
        """
        return:
            train_data (list)
            eval_data (list)
            test_data (list)
        """
        train_data = []
        eval_data = []
        test_data = []
        train_rate = self.config['train_rate']
        eval_rate = self.config['eval_rate']
        user_set = self.data['encoded_data'].keys()
        for uid in tqdm(user_set, desc="dividing data"):
            encoded_trajectories = self.data['encoded_data'][uid]
            traj_len = len(encoded_trajectories)
            # 根据 traj_len 来划分 train eval test
            train_num = math.ceil(traj_len * train_rate)
            eval_num = math.ceil(
                traj_len * (train_rate + eval_rate))
            train_data += encoded_trajectories[:train_num]
            eval_data += encoded_trajectories[train_num:eval_num]
            test_data += encoded_trajectories[eval_num:]
        return train_data, eval_data, test_data

    def get_encoder(self):
        try:
            return getattr(importlib.import_module('libcity.data.dataset.trajectory_encoder'),
                           self.config['traj_encoder'])(self.config)
        except AttributeError:
            raise AttributeError('trajectory encoder is not found')
