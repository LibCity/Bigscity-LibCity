import os
import pandas as pd

from libcity.data.dataset.trajectory_encoder.abstract_trajectory_encoder import AbstractTrajectoryEncoder
from libcity.utils import parse_time

parameter_list = ['dataset', 'min_session_len', 'min_sessions', 'traj_encoder', 'cut_method',
                  'window_size']


class CARATrajectoryEncoder(AbstractTrajectoryEncoder):

    def __init__(self, config):
        super().__init__(config)
        self.uid = 0
        self.location2id = {}  # 因为原始数据集中的部分 loc id 不会被使用到因此这里需要重新编码一下
        self.loc_id = 0
        self.id2locid = {}
        self.tim_max = 47  # 时间编码方式得改变
        self.feature_dict = {'current_loc': 'int', 'current_tim': 'int',
                             'target': 'int', 'target_tim': 'int', 'uid': 'int'
                             }
        parameters_str = ''
        for key in parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'trajectory_{}.json'.format(parameters_str))
        self.dataset = self.config.get('dataset', '')
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.poi_profile = pd.read_csv('./raw_data/{}/{}.geo'.format(self.dataset, self.geo_file))

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
            current_tim = []
            for point in traj:
                loc = point[4]
                now_time = parse_time(point[2])
                if loc not in self.location2id:
                    self.location2id[loc] = self.loc_id
                    self.id2locid[str(self.loc_id)] = loc
                    self.loc_id += 1
                current_loc.append(self.location2id[loc])
                time_code = self._time_encode(now_time)
                current_tim.append(time_code)
            # 完成当前轨迹的编码，下面进行输入的形成
            trace = []
            target = current_loc[-1]
            target_tim = current_tim[-1]
            current_loc = current_loc[:-1]
            current_tim = current_tim[:-1]
            trace.append(current_loc)
            trace.append(current_tim)
            trace.append(target)
            trace.append(target_tim)
            trace.append(uid)
            encoded_trajectories.append(trace)
        return encoded_trajectories

    def gen_data_feature(self):
        loc_pad = self.loc_id
        tim_pad = self.tim_max + 1
        self.pad_item = {
            'current_loc': loc_pad,
            'current_tim': tim_pad
        }
        # 构建 poi 坐标字典
        poi_coor = {}
        for index, row in self.poi_profile.iterrows():
            geo_id = row['geo_id']
            coor = eval(row['coordinates'])
            poi_coor[str(geo_id)] = coor
        self.data_feature = {
            'loc_size': self.loc_id + 1,
            'tim_size': self.tim_max + 2,
            'uid_size': self.uid,
            'loc_pad': loc_pad,
            'tim_pad': tim_pad,
            'id2locid': self.id2locid,
            'poi_coor': poi_coor
        }

    def _time_encode(self, time):
        if time.weekday() in [0, 1, 2, 3, 4]:
            return time.hour
        else:
            return time.hour + 24
