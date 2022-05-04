import os
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from libcity.data.dataset.trajectory_encoder.abstract_trajectory_encoder import AbstractTrajectoryEncoder
from libcity.utils import parse_time, cal_timeoff
from libcity.utils.dataset import parse_coordinate
from tqdm import tqdm

parameter_list = ['dataset', 'min_session_len', 'min_sessions', 'traj_encoder', 'cut_method',
                  'window_size', 'min_checkins', 'max_session_len']


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


class StanEncoder(AbstractTrajectoryEncoder):

    def __init__(self, config):
        super().__init__(config)
        self.uid = 1  # 0 for padding
        self.location2id = {}  # 因为原始数据集中的部分 loc id 不会被使用到因此这里需要重新编码一下
        self.id2location = {}
        self.ex = [0, 0, 0, 0]  # 距离最大值 最小值（0） 时间差最大值 最小值（0）
        self.loc_id = 1  # 0 for padding
        self.feature_dict = {'traj': 'int', 'traj_temporal_mat': 'float',
                             'candiate_temporal_vec': 'float', 'traj_len': 'int',
                             'target': 'int', 'uid': 'int'}
        self.max_len = self.config['max_session_len']  # 最后一个点需要留作 target
        parameters_str = ''
        for key in parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'trajectory_{}.json'.format(parameters_str))

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
            # 切割会保证 len(traj) < self.session_max_len
            current_traj = np.zeros((self.max_len + 1, 3), np.int32)
            current_tim = []
            for i, point in enumerate(traj):
                loc = point[4]
                now_time = parse_time(point[2])
                if loc not in self.location2id:
                    self.location2id[loc] = self.loc_id
                    self.id2location[self.loc_id] = loc
                    self.loc_id += 1
                time_code = self._time_encode(now_time)
                current_traj[i][0] = uid
                current_traj[i][1] = self.location2id[loc]
                current_traj[i][2] = time_code
                current_tim.append(now_time)
            # 完成当前轨迹的编码，下面进行输入的形成
            # calculate trajectory temporal relation matrix
            traj_temporal_mat = self._cal_mat1(current_tim[:-1])
            # calculate candidate temporal relation matrix
            candiate_temporal_mat = self._cal_mat2(current_tim)
            # 一条轨迹可以产生多条训练数据，根据第一个点预测第二个点，前两个点预测第三个点....
            for i in range(len(traj) - 1):
                trace = []
                target = int(current_traj[i+1][1])
                # mask current_traj and traj_temporal_mat
                mask = np.zeros((self.max_len, 3), np.int32)
                mask[:i+1, :] = 1
                mask_traj = current_traj[:-1] * mask
                mask = np.zeros((self.max_len, self.max_len))
                mask[:i+1, :i+1] = 1
                mask_traj_temporal_mat = traj_temporal_mat * mask
                trace.append(mask_traj.tolist())
                trace.append(mask_traj_temporal_mat.tolist())
                trace.append(candiate_temporal_mat[i].tolist())
                trace.append(i+1)
                trace.append(target-1)  # 因为模型预测是从 0 开始预测，而我们的 encode 是从 1 开始
                trace.append(uid)
                encoded_trajectories.append(trace)
        return encoded_trajectories

    def gen_data_feature(self):
        spatial_mat = self._cal_poi_matrix()
        self.data_feature = {
            'loc_size': self.loc_id,
            'tim_size': 169,  # padding value is zero, true time code is 1-168
            'uid_size': self.uid,
            'spatial_matrix': spatial_mat,
            'ex': self.ex
        }

    def _cal_mat1(self, current_tim):
        # calculate the temporal relation matrix
        mat = np.zeros((self.max_len, self.max_len))
        cur_len = len(current_tim)
        for i in range(cur_len):
            for j in range(cur_len):
                off = abs(cal_timeoff(current_tim[i], current_tim[j]))
                mat[i][j] = off
                if off > self.ex[3]:
                    self.ex[3] = off
        return mat

    def _cal_mat2(self, current_tim):
        # calculate the temporal relation matrix
        mat = np.zeros((self.max_len, self.max_len))
        cur_len = len(current_tim)
        for i in range(cur_len):
            if i == 0:
                continue
            for j in range(i):
                off = abs(cal_timeoff(current_tim[i], current_tim[j]))
                mat[i-1][j] = off
                if off > self.ex[3]:
                    self.ex[3] = off
        return mat

    def _time_encode(self, time):
        # 0 for padding value
        return time.hour + time.weekday() * 24 + 1

    def _cal_poi_matrix(self):
        self.dataset = self.config.get('dataset', '')
        self.geo_file = self.config.get('geo_file', self.dataset)
        poi_profile = pd.read_csv('./raw_data/{}/{}.geo'.format(self.dataset, self.geo_file))
        mat = np.zeros((self.loc_id-1, self.loc_id-1))
        for i in tqdm(range(1, self.loc_id), desc='calculate poi distance matrix'):
            lon_i, lat_i = parse_coordinate(poi_profile.iloc[self.id2location[i]]['coordinates'])
            for j in range(1, self.loc_id):
                lon_j, lat_j = parse_coordinate(poi_profile.iloc[self.id2location[j]]['coordinates'])
                dis = haversine(lon_i, lat_i, lon_j, lat_j)
                mat[i-1][j-1] = dis
                if dis > self.ex[0]:
                    self.ex[0] = dis
        return mat.tolist()
