import os
import json
import pandas as pd
import math
from tqdm import tqdm
import importlib
from logging import getLogger

from libcity.data.dataset import AbstractDataset
from libcity.utils import parse_time, cal_timeoff
from libcity.data.utils import generate_dataloader_pad

parameter_list = ['dataset', 'min_session_len', 'min_sessions', "max_session_len",
                  'cut_method', 'window_size', 'min_checkins']


class TrajectoryDataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        self.cache_file_folder = './libcity/cache/dataset_cache/'
        self.cut_data_cache = './libcity/cache/dataset_cache/cut_traj'
        for param in parameter_list:
            self.cut_data_cache += '_' + str(self.config[param])
        self.cut_data_cache += '.json'
        self.dataset = self.config.get('dataset', '')
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        self.data_path = './raw_data/{}/'.format(self.dataset)
        self.data = None
        # 加载 encoder
        self.encoder = self.get_encoder()
        self.pad_item = None  # 因为若是使用缓存, pad_item 是记录在缓存文件中的而不是 encoder
        self.logger = getLogger()

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
                if os.path.exists(self.cut_data_cache):
                    f = open(self.cut_data_cache, 'r')
                    cut_data = json.load(f)
                    f.close()
                else:
                    cut_data = self.cutter_filter()
                    if not os.path.exists(self.cache_file_folder):
                        os.makedirs(self.cache_file_folder)
                    with open(self.cut_data_cache, 'w') as f:
                        json.dump(cut_data, f)
                self.logger.info('finish cut data')
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
        filter_location = group_location[group_location['time'] >= self.config['min_checkins']]
        location_index = filter_location.index.tolist()
        traj = traj[traj['location'].isin(location_index)]
        user_set = pd.unique(traj['entity_id'])
        res = {}
        min_session_len = self.config['min_session_len']
        max_session_len = self.config['max_session_len']
        min_sessions = self.config['min_sessions']
        window_size = self.config['window_size']
        cut_method = self.config['cut_method']
        if cut_method == 'time_interval':
            # 按照时间窗口进行切割
            for uid in tqdm(user_set, desc="cut and filter trajectory"):
                usr_traj = traj[traj['entity_id'] == uid].to_numpy()
                sessions = []  # 存放该用户所有的 session
                session = []  # 单条轨迹
                for index, row in enumerate(usr_traj):
                    now_time = parse_time(row[2])
                    if index == 0:
                        session.append(row.tolist())
                        prev_time = now_time
                    else:
                        time_off = cal_timeoff(now_time, prev_time)
                        if time_off < window_size and time_off >= 0 and len(session) < max_session_len:
                            session.append(row.tolist())
                        else:
                            if len(session) >= min_session_len:
                                sessions.append(session)
                            session = []
                            session.append(row.tolist())
                    prev_time = now_time
                if len(session) >= min_session_len:
                    sessions.append(session)
                if len(sessions) >= min_sessions:
                    res[str(uid)] = sessions
        elif cut_method == 'same_date':
            # 将同一天的 check-in 划为一条轨迹
            for uid in tqdm(user_set, desc="cut and filter trajectory"):
                usr_traj = traj[traj['entity_id'] == uid].to_numpy()
                sessions = []  # 存放该用户所有的 session
                session = []  # 单条轨迹
                prev_date = None
                for index, row in enumerate(usr_traj):
                    now_time = parse_time(row[2])
                    now_date = now_time.day
                    if index == 0:
                        session.append(row.tolist())
                    else:
                        if prev_date == now_date and len(session) < max_session_len:
                            # 还是同一天
                            session.append(row.tolist())
                        else:
                            if len(session) >= min_session_len:
                                sessions.append(session)
                            session = []
                            session.append(row.tolist())
                    prev_date = now_date
                if len(session) >= min_session_len:
                    sessions.append(session)
                if len(sessions) >= min_sessions:
                    res[str(uid)] = sessions
        else:
            # cut by fix window_len used by STAN
            if max_session_len != window_size:
                raise ValueError('the fixed length window is not equal to max_session_len')
            for uid in tqdm(user_set, desc="cut and filter trajectory"):
                usr_traj = traj[traj['entity_id'] == uid].to_numpy()
                sessions = []  # 存放该用户所有的 session
                session = []  # 单条轨迹
                for index, row in enumerate(usr_traj):
                    if len(session) < window_size:
                        session.append(row.tolist())
                    else:
                        sessions.append(session)
                        session = []
                        session.append(row.tolist())
                if len(session) >= min_session_len:
                    sessions.append(session)
                if len(sessions) >= min_sessions:
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
                    checkin_record,
                    checkin_record,
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
        encoded_data = {}
        for uid in tqdm(data, desc="encoding trajectory"):
            encoded_data[uid] = self.encoder.encode(int(uid), data[uid])
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
