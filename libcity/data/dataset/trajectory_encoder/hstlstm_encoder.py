import os
import math
import pandas as pd
from datetime import datetime
from geopy import distance

from libcity.data.dataset.trajectory_encoder.abstract_trajectory_encoder import AbstractTrajectoryEncoder
from libcity.utils import parse_time, parse_coordinate

parameter_list = ['dataset', 'min_session_len', 'min_sessions', 'traj_encoder', 'cut_method',
                  'window_size', 'history_type', 'min_checkins', 'max_session_len']


class HstlstmEncoder(AbstractTrajectoryEncoder):

    def __init__(self, config):
        super().__init__(config)
        self.uid = 0
        self.location2id = {}  # 因为原始数据集中的部分 loc id 不会被使用到因此这里需要重新编码一下
        self.loc_id = 0
        self.tim_max = 100.0  # 最大的时间差（单位秒）
        self.tim_interval_init = 0.0
        self.dis_max = 0.5  # 最大的距离差（单位千米）
        self.history_type = self.config['history_type']
        self.feature_dict = {'current_loc': 'int', 'tim_interval': 'float',
                             'dis': 'float', 'target': 'int', 'uid': 'int'
                             }
        if config['evaluate_method'] == 'sample':
            self.feature_dict['neg_loc'] = 'int'
            parameter_list.append('neg_samples')
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

    def encode(self, uid, trajectories, negative_sample=None):
        """standard encoder use the same method as DeepMove

        Recode poi id. Encode timestamp with its hour.

        Args:
            uid ([type]): same as AbstractTrajectoryEncoder
            trajectories ([type]): same as AbstractTrajectoryEncoder
                trajectory1 = [
                    (location ID, timestamp, timezone_offset_in_minutes),
                    (location ID, timestamp, timezone_offset_in_minutes),
                    .....
                ]
        """
        # 直接对 uid 进行重编码
        uid = self.uid
        self.uid += 1
        encoded_trajectories = []
        for index, traj in enumerate(trajectories):
            current_loc = []
            tim_interval = []
            dis = []
            pre_time = None
            pre_lon = None
            pre_lat = None
            for index, point in enumerate(traj):
                loc = point[4]
                now_time = parse_time(point[2])
                lon, lat = parse_coordinate(self.geo.loc[self.geo['geo_id'] == loc].iloc[0]['coordinates'])
                if index == 0:
                    if loc not in self.location2id:
                        self.location2id[loc] = self.loc_id
                        self.loc_id += 1
                    current_loc.append(self.location2id[loc])
                    tim_interval.append(self.tim_interval_init)  # use fixed interval for start point
                    dis.append(self.dis_max)
                else:
                    if loc not in self.location2id:
                        self.location2id[loc] = self.loc_id
                        self.loc_id += 1
                    current_loc.append(self.location2id[loc])
                    tim_diff = datetime.timestamp(now_time) - datetime.timestamp(pre_time)
                    if tim_diff > self.tim_max:
                        self.tim_max = tim_diff
                    tim_interval.append(tim_diff)
                    dis_diff = distance.distance((pre_lat, pre_lon), (lat, lon)).kilometers
                    if dis_diff > self.dis_max:
                        self.dis_max = dis_diff
                    dis.append(dis_diff)
                pre_time = now_time
                pre_lat = lat
                pre_lon = lon
            # 一条轨迹可以产生多条训练数据，根据第一个点预测第二个点，前两个点预测第三个点....
            for i in range(len(current_loc) - 1):
                trace = []
                target = current_loc[i+1]
                trace.append(current_loc[:i+1])
                trace.append(tim_interval[:i+1])
                trace.append(dis[:i+1])
                trace.append(target)
                trace.append(uid)
                if negative_sample is not None:
                    neg_loc = []
                    for neg in negative_sample[index]:
                        if neg not in self.location2id:
                            self.location2id[neg] = self.loc_id
                            self.loc_id += 1
                        neg_loc.append(self.location2id[neg])
                    trace.append(neg_loc)
                encoded_trajectories.append(trace)
        return encoded_trajectories

    def gen_data_feature(self):
        loc_pad = self.loc_id
        self.pad_item = {
            'current_loc': loc_pad,
            'tim_interval': 0.0,
            'dis': 0.0
        }
        self.data_feature = {
            'loc_size': self.loc_id + 1,
            'tim_slot_max': int(math.ceil(self.tim_max)),
            'dis_slot_max': int(math.ceil(self.dis_max)),
            'uid_size': self.uid,
            'loc_pad': loc_pad
        }
