import os
import pandas as pd
import numpy as np
import math
from libcity.data.dataset.trajectory_encoder.abstract_trajectory_encoder import AbstractTrajectoryEncoder
from libcity.utils import parse_time
from libcity.utils.dataset import parse_coordinate
from collections import defaultdict

parameter_list = ['dataset', 'min_session_len', 'min_sessions', 'traj_encoder', 'window_size', 'min_checkins',
                  'max_session_len']


def geodistance(lat1, lng1, lat2, lng2):
    lng1, lat1, lng2, lat2 = map(math.radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon = lng2-lng1
    dlat = lat2-lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    distance = 2*math.asin(math.sqrt(a))*6371*1000
    distance = round(distance/1000, 3)
    return distance


class LstpmEncoder(AbstractTrajectoryEncoder):

    def __init__(self, config):
        super().__init__(config)
        self.uid = 0
        self.location2id = {}  # 因为原始数据集中的部分 loc id 不会被使用到因此这里需要重新编码一下
        self.id2location = {}
        self.loc_id = 0
        self.tim_max = 47  # LSTPM 做的是 48 个 time slot
        self.feature_dict = {'history_loc': 'array of int', 'history_tim': 'array of int',
                             'current_loc': 'int', 'current_tim': 'int', 'dilated_rnn_input_index': 'no_pad_int',
                             'history_avg_distance': 'no_pad_float',
                             'target': 'int', 'uid': 'int'}
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
        self.poi_profile = pd.read_csv('./raw_data/{}/{}.geo'.format(self.dataset, self.geo_file))
        self.time_checkin_set = defaultdict(set)

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
        history_loc = []
        history_loc_central = []
        history_tim = []
        for index, traj in enumerate(trajectories):
            current_loc = []
            current_tim = []
            for point in traj:
                loc = point[4]
                now_time = parse_time(point[2])
                if loc not in self.location2id:
                    self.location2id[loc] = self.loc_id
                    self.id2location[self.loc_id] = loc
                    self.loc_id += 1
                current_loc.append(self.location2id[loc])
                time_code = self._time_encode(now_time)
                current_tim.append(time_code)
                if time_code not in self.time_checkin_set:
                    self.time_checkin_set[time_code] = set()
                self.time_checkin_set[time_code].add(self.location2id[loc])
            # 完成当前轨迹的编码，下面进行输入的形成
            if index == 0:
                # 因为要历史轨迹特征，所以第一条轨迹是不能构成模型输入的
                history_loc.append(current_loc)
                history_tim.append(current_tim)
                lon = []
                lat = []
                for poi in current_loc:
                    lon_cur, lat_cur = parse_coordinate(self.poi_profile.loc[self.poi_profile['geo_id']
                                                        == self.id2location[poi]].iloc[0]['coordinates'])
                    lon.append(lon_cur)
                    lat.append(lat_cur)
                history_loc_central.append((np.mean(lat), np.mean(lon)))
                continue
            # 一条轨迹可以生成多个数据点
            for i in range(len(current_loc) - 1):
                trace = []
                target = current_loc[i+1]
                dilated_rnn_input_index = self._create_dilated_rnn_input(current_loc[:i+1])
                history_avg_distance = self._gen_distance_matrix(current_loc[:i+1], history_loc_central)
                trace.append(history_loc.copy())
                trace.append(history_tim.copy())
                trace.append(current_loc[:i+1])
                trace.append(current_tim[:i+1])
                trace.append(dilated_rnn_input_index)
                trace.append(history_avg_distance)
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
            history_loc.append(current_loc)
            history_tim.append(current_tim)
            # calculate current_loc
            lon = []
            lat = []
            for poi in current_loc:
                lon_cur, lat_cur = parse_coordinate(self.poi_profile.loc[self.poi_profile['geo_id']
                                                    == self.id2location[poi]].iloc[0]['coordinates'])
                lon.append(lon_cur)
                lat.append(lat_cur)
            history_loc_central.append((np.mean(lat), np.mean(lon)))
        return encoded_trajectories

    def gen_data_feature(self):
        loc_pad = self.loc_id
        tim_pad = self.tim_max + 1
        self.pad_item = {
            'current_loc': loc_pad,
            'current_tim': tim_pad
        }
        # generate time_sim_matrix
        # the pad time will not appear here
        sim_matrix = np.zeros((self.tim_max+1, self.tim_max+1))
        for i in range(self.tim_max+1):
            sim_matrix[i][i] = 1
            for j in range(i+1, self.tim_max+1):
                set_i = self.time_checkin_set[i]
                set_j = self.time_checkin_set[j]
                if len(set_i | set_j) != 0:
                    jaccard_ij = len(set_i & set_j) / len(set_i | set_j)
                    sim_matrix[i][j] = jaccard_ij
                    sim_matrix[j][i] = jaccard_ij
        self.data_feature = {
            'loc_size': self.loc_id + 1,
            'tim_size': self.tim_max + 2,
            'uid_size': self.uid,
            'loc_pad': loc_pad,
            'tim_pad': tim_pad,
            'tim_sim_matrix': sim_matrix.tolist()
        }

    def _create_dilated_rnn_input(self, current_loc):
        current_loc.reverse()
        sequence_length = len(current_loc)
        session_dilated_rnn_input_index = [0] * sequence_length
        for i in range(sequence_length - 1):
            current_poi = current_loc[i]
            poi_before = current_loc[i + 1:]
            current_poi_profile = self.poi_profile.loc[self.poi_profile['geo_id']
                                                       == self.id2location[current_poi]].iloc[0]
            lon_cur, lat_cur = parse_coordinate(current_poi_profile['coordinates'])
            distance_row_explicit = []
            for target in poi_before:
                lon, lat = parse_coordinate(self.poi_profile.loc[self.poi_profile['geo_id']
                                            == self.id2location[target]].iloc[0]['coordinates'])
                distance_row_explicit.append(geodistance(lat_cur, lon_cur, lat, lon))
            index_closet = np.argmin(distance_row_explicit).item()
            # reverse back
            session_dilated_rnn_input_index[sequence_length - i - 1] = sequence_length - 2 - index_closet - i
        current_loc.reverse()
        return session_dilated_rnn_input_index

    def _gen_distance_matrix(self, current_loc, history_loc_central):
        # 使用 profile 计算当前位置与历史轨迹中心点之间的距离
        history_avg_distance = []  # history_session_count
        now_loc = current_loc[-1]
        lon_cur, lat_cur = parse_coordinate(self.poi_profile.loc[self.poi_profile['geo_id']
                                            == self.id2location[now_loc]].iloc[0]['coordinates'])
        for central in history_loc_central:
            dis = geodistance(central[0], central[1], lat_cur, lon_cur)
            if dis < 1:
                dis = 1
            history_avg_distance.append(dis)
        return history_avg_distance

    def _time_encode(self, time):
        if time.weekday() in [0, 1, 2, 3, 4]:
            return time.hour
        else:
            return time.hour + 24
