import os
import pandas as pd
from datetime import datetime
from geopy import distance

from libcity.data.dataset.trajectory_encoder.abstract_trajectory_encoder import AbstractTrajectoryEncoder
from libcity.utils import parse_time, parse_coordinate

parameter_list = ['dataset', 'min_session_len', 'min_sessions', 'traj_encoder', 'cut_method',
                  'window_size', 'min_checkins', 'neg_samples']


class AtstlstmEncoder(AbstractTrajectoryEncoder):
    # 这里有问题，需要重新修改

    def __init__(self, config):
        super().__init__(config)
        self.uid = 0
        self.location2id = {}  # 因为原始数据集中的部分 loc id 不会被使用到因此这里需要重新编码一下
        self.loc_id = 0
        self.tim_max = 0  # 记录最大的时间编码
        if self.config['cut_method'] == 'time_interval':
            # 对于以时间窗口切割的轨迹，最大时间编码是已知的
            self.tim_max = self.config['window_size'] - 1
        self.feature_dict = {'current_loc': 'int', 'loc_neg': 'int',
                             'current_dis': 'float', 'dis_neg': 'float',
                             'current_tim': 'float', 'tim_neg': 'float', 'uid': 'int',
                             'target_loc': 'int', 'target_dis': 'float', 'target_tim': 'float'
                             }
        parameters_str = ''
        for key in parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'trajectory_{}.json'.format(parameters_str))
        self.dataset = self.config.get('dataset', '')
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.data_path = './raw_data/{}/'.format(self.dataset)
        self.geo = pd.read_csv(os.path.join(self.data_path, '{}.geo'.format(self.geo_file)))

    def encode(self, uid, trajectories, negative_sample):
        """Encoded Method refered to the open source code
            https://github.com/drhuangliwei/An-Attention-based-Spatiotemporal-LSTM-Network-for-Next-POI-Recommendation
            row index is:
                   0      1    2     3          4
                dyna_id,type,time,entity_id,location
        """
        # 直接对 uid 进行重编码
        uid = self.uid
        self.uid += 1
        encoded_trajectories = []
        for i, traj in enumerate(trajectories):
            current_loc = []  # the checkin poi list
            loc_distance = []  # the distance between two checkin
            tim_interval = []  # the time interval between two checkin
            pre_time = None
            pre_lat = None
            pre_lon = None
            for index, row in enumerate(traj):
                loc = row[4]
                now_time = parse_time(row[2])
                lon, lat = parse_coordinate(self.geo.loc[self.geo['geo_id'] == loc].iloc[0]['coordinates'])
                if index == 0:
                    # for the first checkin, distance and time_interval set to a fixed value
                    if loc not in self.location2id:
                        self.location2id[loc] = self.loc_id
                        self.loc_id += 1
                    current_loc.append(self.location2id[loc])
                    tim_interval.append(100)  # choose the same fixed value as the reference code
                    loc_distance.append(1)
                else:
                    if loc not in self.location2id:
                        self.location2id[loc] = self.loc_id
                        self.loc_id += 1
                    current_loc.append(self.location2id[loc])
                    # the unit of time is second
                    tim_interval.append(datetime.timestamp(now_time) - datetime.timestamp(pre_time))
                    loc_distance.append(distance.distance((pre_lat, pre_lon), (lat, lon)).kilometers)
                pre_time = now_time
                pre_lat = lat
                pre_lon = lon
            # generate negative samples' current_loc loc_distance and tim_interval
            neg_loc = []
            neg_distance = []
            neg_time = []
            # the final checkin will be target (positive sample), so use the second last to cal neg
            row = traj[-2]
            loc = row[4]
            pre_lon, pre_lat = parse_coordinate(self.geo.loc[self.geo['geo_id'] == loc].iloc[0]['coordinates'])
            for neg in negative_sample[i]:
                neg_lon, neg_lat = parse_coordinate(self.geo.loc[self.geo['geo_id'] == neg].iloc[0]['coordinates'])
                if neg not in self.location2id:
                    self.location2id[neg] = self.loc_id
                    self.loc_id += 1
                neg_loc.append(self.location2id[neg])
                neg_time.append(tim_interval[-1])  # use target's time interval as the neg sample's
                neg_distance.append(distance.distance((neg_lat, neg_lon), (pre_lat, pre_lon)).kilometers)
            trace = []
            target_loc = current_loc[-1]
            target_dis = loc_distance[-1]
            target_tim = tim_interval[-1]
            trace.append(current_loc[:-1])
            trace.append(neg_loc)
            trace.append(loc_distance[:-1])
            trace.append(neg_distance)
            trace.append(tim_interval[:-1])
            trace.append(neg_time)
            trace.append(uid)
            trace.append(target_loc)
            trace.append(target_dis)
            trace.append(target_tim)
            encoded_trajectories.append(trace)
        return encoded_trajectories

    def gen_data_feature(self):
        loc_pad = self.loc_id
        self.pad_item = {
            'current_loc': loc_pad,
            'current_dis': 0.0,
            'current_tim': 0.0
        }
        self.data_feature = {
            'loc_size': self.loc_id + 1,
            'uid_size': self.uid,
            'loc_pad': loc_pad
        }
