import os
import numpy as np
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

from libcity.data.dataset.eta_encoder.abstract_eta_encoder import AbstractETAEncoder


parameter_list = [
    'dataset', 'eta_encoder',
]
parameter_list_cut = [
    'dataset', 'eta_encoder', 'cut_method', 'min_session_len', 'max_session_len', 'min_sessions', 'window_size',
]


def geo_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = tuple(map(lambda x: radians(x), (lon1, lat1, lon2, lat2)))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


class DeeptteEncoder(AbstractETAEncoder):

    def __init__(self, config):
        super().__init__(config)
        self.feature_dict = {
            'current_longi': 'float', 'current_lati': 'float',
            'current_tim': 'float', 'current_dis': 'float',
            'current_state': 'float',
            'uid': 'int',
            'weekid': 'int',
            'timeid': 'int',
            'dist': 'float',
            'time': 'float',
            'traj_len': 'int',
            'traj_id': 'int',
            'start_timestamp': 'int',
        }
        self.traj_len_idx = len(self.feature_dict) - 1
        parameters_str = ''
        need_cut = self.config.get("need_cut", False)
        self.parameter_list = parameter_list_cut if need_cut else parameter_list
        for key in self.parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'eta{}.json'.format(parameters_str))

        self.uid_size = 0

    def encode(self, uid, trajectories, dyna_feature_column):
        self.uid_size = max(uid + 1, self.uid_size)
        encoded_trajectories = []
        for traj in trajectories:
            current_longi = []
            current_lati = []
            current_tim = []
            current_dis = []
            current_state = []
            begin_time = datetime.strptime(traj[0][dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')
            end_time = datetime.strptime(traj[-1][dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')
            weekid = int(begin_time.weekday())
            timeid = int(begin_time.strftime('%H')) * 60 + int(begin_time.strftime('%M'))
            time = (end_time - begin_time).seconds
            traj_len = len(traj)
            traj_id = int(traj[-1][dyna_feature_column["traj_id"]])
            start_timestamp = datetime.timestamp(begin_time)
            last_dis = 0
            for point in traj:
                coordinate = eval(point[dyna_feature_column["coordinates"]])
                longi, lati = float(coordinate[0]), float(coordinate[1])
                current_longi.append(longi)
                current_lati.append(lati)

                if "current_dis" in dyna_feature_column:
                    dis = point[dyna_feature_column["current_dis"]]
                elif len(current_longi) == 1:
                    dis = 0
                else:
                    dis = geo_distance(current_longi[-2], current_lati[-2], longi, lati) + last_dis
                    last_dis = dis
                current_dis.append(dis)

                tim = datetime.strptime(point[dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')
                current_tim.append(float((tim - begin_time).seconds))

                if "current_state" in dyna_feature_column:
                    state = point[dyna_feature_column["current_state"]]
                else:
                    state = 0
                current_state.append(state)
            if "current_dis" in dyna_feature_column:
                dist = traj[-1][dyna_feature_column["current_dis"]] - traj[0][dyna_feature_column["current_dis"]]
            else:
                dist = last_dis
            encoded_trajectories.append([
                current_longi[:], current_lati[:],
                current_tim[:], current_dis[:],
                current_state[:],
                [uid],
                [weekid],
                [timeid],
                [dist],
                [time],
                [traj_len],
                [traj_id],
                [start_timestamp],
            ])
        return encoded_trajectories

    def gen_data_feature(self):
        self.pad_item = {
            'current_longi': 0,
            'current_lati': 0,
            'current_tim': 0,
            'current_dis': 0,
            'current_state': 0,
        }
        self.data_feature = {
            'traj_len_idx': self.traj_len_idx,
            'uid_size': self.uid_size,
        }

    def gen_scalar_data_feature(self, train_data):
        longi_list = []
        lati_list = []
        dist_list = []
        time_list = []
        dist_gap_list = []
        time_gap_list = []
        scalar_feature_column = {}
        for i, key in enumerate(self.feature_dict):
            scalar_feature_column[key] = i
        for data in train_data:
            traj_len = data[scalar_feature_column["traj_len"]][0]
            longi_list.extend(data[scalar_feature_column["current_longi"]])
            lati_list.extend(data[scalar_feature_column["current_lati"]])
            dist_list.extend(data[scalar_feature_column["dist"]])
            time_list.extend(data[scalar_feature_column["time"]])
            dist_gap = data[scalar_feature_column["current_dis"]][:traj_len]
            dist_gap = list(map(lambda x: x[0] - x[1], zip(dist_gap[1:], dist_gap[:-1])))
            dist_gap_list.extend(dist_gap)
            time_gap = data[scalar_feature_column["current_tim"]][:traj_len]
            time_gap = list(map(lambda x: x[0] - x[1], zip(time_gap[1:], time_gap[:-1])))
            time_gap_list.extend(time_gap)
        scalar_data_feature = {
            'longi_mean': np.mean(longi_list),
            'longi_std': np.std(longi_list),
            'lati_mean': np.mean(lati_list),
            'lati_std': np.std(lati_list),
            'dist_mean': np.mean(dist_list),
            'dist_std': np.std(dist_list),
            'time_mean': np.mean(time_list),
            'time_std': np.std(time_list),
            'dist_gap_mean': np.mean(dist_gap_list),
            'dist_gap_std': np.std(dist_gap_list),
            'time_gap_mean': np.mean(time_gap_list),
            'time_gap_std': np.std(time_gap_list),
        }
        for k, v in scalar_data_feature.items():
            self._logger.info("{}: {}".format(k, v))
        return scalar_data_feature
