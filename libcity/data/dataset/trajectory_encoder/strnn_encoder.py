import os
import numpy as np
from libcity.data.dataset.trajectory_encoder.abstract_trajectory_encoder import AbstractTrajectoryEncoder
from libcity.utils import parse_time, cal_basetime, cal_timeoff

parameter_list = ['dataset', 'min_session_len', 'min_sessions', 'traj_encoder', 'cut_method',
                  'window_size', 'history_type']


class StrnnEncoder(AbstractTrajectoryEncoder):

    def __init__(self, config):
        super().__init__(config)
        self.uid = 0
        self.location2id = {}  # 因为原始数据集中的部分 loc id 不会被使用到因此这里需要重新编码一下
        self.loc_id = 0
        self.tim_max = 0  # 记录最大的时间编码
        self.feature_dict = {'current_loc': 'int', 'current_tim': 'int',
                             'target': 'int', 'target_tim': 'int', 'uid': 'int', 'current_dis': 'float'
                             }
        parameters_str = ''
        for key in parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'trajectory_{}.json'.format(parameters_str))

        self.geo_coord = {}
        self.dataset = self.config.get('dataset', '')
        self.geo_file = self.config.get('geo_file', self.dataset)
        path = "./raw_data/{}/{}.geo".format(self.dataset, self.geo_file)
        f_geo = open(path)
        lines = f_geo.readlines()

        for i, line in enumerate(lines):
            if i == 0:
                continue
            tokens = line.strip().replace("\"", "").replace("[", "").replace("]", "").split(',')

            loc_id, loc_longi, loc_lati = int(tokens[0]), eval(tokens[2]), eval(tokens[3])
            self.geo_coord[loc_id] = [loc_lati, loc_longi]
        f_geo.close()

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
            current_longi = []
            current_lati = []
            current_points = []
            start_time = parse_time(traj[0][2])
            # 以当天凌晨的时间作为计算 time_off 的基准
            base_time = cal_basetime(start_time, True)
            for point in traj:
                loc = point[4]
                now_time = parse_time(point[2])
                if loc not in self.location2id:
                    self.location2id[loc] = self.loc_id
                    self.loc_id += 1
                current_points.append(loc)
                current_loc.append(self.location2id[loc])
                current_lati.append(self.geo_coord[loc][0])
                current_longi.append(self.geo_coord[loc][1])
                time_code = int(cal_timeoff(now_time, base_time))
                if time_code > self.tim_max:
                    self.tim_max = time_code
                current_tim.append(time_code)
            # 完成当前轨迹的编码，下面进行输入的形成
            trace = []
            target = current_loc[-1]
            target_tim = current_tim[-1]
            current_loc = current_loc[:-1]
            current_tim = current_tim[:-1]
            lati = self.geo_coord[current_points[-1]][0]
            lati = np.array([lati for i in range(len(current_loc))])
            longi = self.geo_coord[current_points[-1]][1]
            longi = np.array([longi for i in range(len(current_loc))])
            current_dis = euclidean_dist(lati - current_lati[:-1], longi - current_longi[:-1])
            trace.append(current_loc)
            trace.append(current_tim)
            trace.append(target)
            trace.append(target_tim)
            trace.append(uid)
            trace.append(current_dis)
            encoded_trajectories.append(trace)
        return encoded_trajectories

    def gen_data_feature(self):
        loc_pad = self.loc_id
        tim_pad = self.tim_max + 1
        dis_pad = 0.0
        self.pad_item = {
            'current_loc': loc_pad,
            'current_tim': tim_pad,
            'current_dis': dis_pad
        }
        self.data_feature = {
            'loc_size': self.loc_id + 1,
            'tim_size': self.tim_max + 2,
            'uid_size': self.uid,
            'loc_pad': loc_pad,
            'tim_pad': tim_pad,
            'dis_pad': dis_pad
        }


def euclidean_dist(x, y):
    return np.sqrt(np.power(x, 2) + np.power(y, 2)).tolist()
