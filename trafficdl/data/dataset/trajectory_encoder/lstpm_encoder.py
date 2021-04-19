import os
import pandas as pd
import numpy as np
from geopy import distance
from trafficdl.data.dataset.trajectory_encoder.abstract_trajectory_encoder import AbstractTrajectoryEncoder
from trafficdl.utils import parse_time, cal_basetime, cal_timeoff
from trafficdl.utils.dataset import parse_coordinate
from collections import defaultdict

parameter_list = ['dataset', 'min_session_len', 'min_sessions', 'traj_encoder', 'window_size']


class LstpmEncoder(AbstractTrajectoryEncoder):

    def __init__(self, config):
        super().__init__(config)
        self.uid = 0
        self.location2id = {}  # 因为原始数据集中的部分 loc id 不会被使用到因此这里需要重新编码一下
        self.id2location = {}
        self.loc_id = 0
        self.tim_max = 47  # LSTPM 做的是 48 个 time slot
        self.feature_dict = {'history_loc': 'array of int', 'history_tim': 'array of int',
                             'current_loc': 'int', 'current_tim': 'int', 'dilated_rnn_input_index': 'array of int',
                             'history_avg_distance': 'no_pad_float',
                             'target': 'int', 'uid': 'int'}
        parameters_str = ''
        for key in parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './trafficdl/cache/dataset_cache/', 'trajectory_{}.json'.format(parameters_str))
        self.poi_profile = pd.read_csv('./raw_data/{}/{}.geo'.format(self.config['dataset'], self.config['dataset']))
        self.time_checkin_set = defaultdict(set)

    def encode(self, uid, trajectories):
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
        history_tim = []
        for index, traj in enumerate(trajectories):
            current_loc = []
            current_tim = []
            start_time = parse_time(traj[0][1], traj[0][2])
            # 以当天凌晨的时间作为计算 time_off 的基准
            base_time = cal_basetime(start_time, True)
            for point in traj:
                loc = point[0]
                now_time = parse_time(point[1], point[2])
                if loc not in self.location2id:
                    self.location2id[loc] = self.loc_id
                    self.id2location[self.loc_id] = loc
                    self.loc_id += 1
                current_loc.append(self.location2id[loc])
                time_code = int(cal_timeoff(now_time, base_time))
                if now_time.weekday() in [5, 6]:
                    time_code += 24
                current_tim.append(time_code)
                if time_code not in self.time_checkin_set:
                    self.time_checkin_set[time_code] = set()
                self.time_checkin_set[time_code].add(self.location2id[loc])
            # 完成当前轨迹的编码，下面进行输入的形成
            if index == 0:
                # 因为要历史轨迹特征，所以第一条轨迹是不能构成模型输入的
                history_loc.append(current_loc)
                history_tim.append(current_tim)
                continue
            trace = []
            target = current_loc[-1]
            current_loc = current_loc[:-1]
            current_tim = current_tim[:-1]
            dilated_rnn_input_index = self._create_dilated_rnn_input(current_loc)
            history_avg_distance = self._gen_distance_matrix(current_loc, history_loc)
            trace.append(history_loc)
            trace.append(history_tim)
            trace.append(current_loc)
            trace.append(current_tim)
            trace.append(dilated_rnn_input_index)
            trace.append(history_avg_distance)
            trace.append(target)
            trace.append(uid)
            encoded_trajectories.append(trace)
            history_loc.append(current_loc)
            history_tim.append(current_tim)
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
        sim_matrix = np.zeros((self.time_max+1, self.tim_max+1))
        for i in range(self.time_max+1):
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
            'tim_sim_matrix': sim_matrix
        }

    def _create_dilated_rnn_input(self, current_loc):
        current_loc.reverse()
        sequence_length = len(current_loc)
        session_dilated_rnn_input_index = [0] * sequence_length
        for i in range(sequence_length - 1):
            current_poi = current_loc[i]
            poi_before = current_loc[i + 1:]
            current_poi_profile = self.poi_profile.iloc[self.id2location[current_poi]]
            lon_cur, lat_cur = parse_coordinate(current_poi_profile['coordinates'])
            distance_row_explicit = []
            for target in poi_before:
                lon, lat = parse_coordinate(self.poi_profile.iloc[self.id2location[target]]['coordinates'])
                distance_row_explicit.append(distance.distance((lat_cur, lon_cur), (lat, lon)).kilometers)
            index_closet = np.argmin(distance_row_explicit)
            # reverse back
            session_dilated_rnn_input_index[sequence_length - i - 1] = sequence_length - 2 - index_closet - i
        current_loc.reverse()
        return session_dilated_rnn_input_index

    def _gen_distance_matrix(self, current_loc, history_loc):
        # 使用 profile 计算局部距离矩阵
        history_avg_distance = []  # history_session_count * cur_seq_len
        for sequence in history_loc:
            distance_matrix = []
            for origin in current_loc:
                lon_cur, lat_cur = parse_coordinate(self.poi_profile.iloc[self.id2location[origin]]['coordinates'])
                distance_row = []
                for target in sequence:
                    lon, lat = parse_coordinate(self.poi_profile.iloc[self.id2location[target]]['coordinates'])
                    distance_row.append(distance.distance((lat_cur, lon_cur), (lat, lon)).kilometers)
                distance_matrix.append(distance_row)
            # distance_matrix is cur_seq_len * history_len
            distance_avg = np.mean(distance_matrix, axis=1).tolist()  # cur_seq_len
            history_avg_distance.append(distance_avg)
        return history_avg_distance
