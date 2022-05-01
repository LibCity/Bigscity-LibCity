import os
import numpy as np
import pandas as pd
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


class TtpnetEncoder(AbstractETAEncoder):

    def __init__(self, config):
        super().__init__(config)
        self.feature_dict = {
            'current_longi': 'float', 'current_lati': 'float', 'current_loc': 'int',
            'current_tim': 'float', 'masked_current_tim': 'float', 'current_dis': 'float',
            'speeds': 'float', 'speeds_relevant1': 'float', 'speeds_relevant2': 'float', 'speeds_long': 'float',
            'grid_len': 'float',
            'uid': 'int',
            'weekid': 'int',
            'timeid': 'int',
            'dist': 'float',
            'holiday': 'int',
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

        self.geo_embedding = []
        self.dataset = self.config.get('dataset', '')
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        path = "./raw_data/{}/{}.geo".format(self.dataset, self.geo_file)
        f_geo = pd.read_csv(path)
        self._logger.info("Loaded file " + self.geo_file + '.geo')
        for row in f_geo.itertuples():
            embedding = eval(getattr(row, 'embeddings'))
            self.geo_embedding.append(embedding[:])

        self.uid_size = 0

    def encode(self, uid, trajectories, dyna_feature_column):
        self.uid_size = max(uid + 1, self.uid_size)
        encoded_trajectories = []
        for traj in trajectories:
            current_longi = []
            current_lati = []
            current_loc = []
            current_tim = []
            masked_current_tim = []
            current_dis = []
            speeds = []
            speeds_relevant1 = []
            speeds_relevant2 = []
            speeds_long = []
            grid_len = []

            begin_time = datetime.strptime(traj[0][dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')
            end_time = datetime.strptime(traj[-1][dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')
            weekid = int(begin_time.weekday())
            timeid = int(begin_time.strftime('%H')) * 4 + int(begin_time.strftime('%M')) // 15
            time = (end_time - begin_time).seconds

            holiday = 0
            if "holiday" in dyna_feature_column:
                holiday = traj[0][dyna_feature_column["holiday"]]
            elif weekid == 0 or weekid == 6:
                holiday = 1

            traj_len = len(traj)
            traj_id = int(traj[-1][dyna_feature_column["traj_id"]])
            start_timestamp = datetime.timestamp(begin_time)
            last_dis = 0
            for point in traj:
                coordinate = eval(point[dyna_feature_column["coordinates"]])
                longi, lati = float(coordinate[0]), float(coordinate[1])
                current_longi.append(longi)
                current_lati.append(lati)

                loc = point[dyna_feature_column["location"]]
                current_loc.append(loc)

                speed = eval(point[dyna_feature_column["speeds"]])
                speeds.extend(speed)

                speed_relevant1 = eval(point[dyna_feature_column["speeds_relevant1"]])
                speeds_relevant1.extend(speed_relevant1)

                speed_relevant2 = eval(point[dyna_feature_column["speeds_relevant2"]])
                speeds_relevant2.extend(speed_relevant2)

                speed_long = eval(point[dyna_feature_column["speeds_long"]])
                speeds_long.extend(speed_long)

                grid_length = point[dyna_feature_column["grid_len"]]
                grid_len.append(grid_length)

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
                masked_current_tim.append(1)

            if "current_dis" in dyna_feature_column:
                dist = traj[-1][dyna_feature_column["current_dis"]] - traj[0][dyna_feature_column["current_dis"]]
            else:
                dist = last_dis
            encoded_trajectories.append([
                current_longi[:], current_lati[:],
                current_loc[:],
                current_tim[:], masked_current_tim[:], current_dis[:],
                speeds[:], speeds_relevant1[:], speeds_relevant2[:], speeds_long[:],
                grid_len[:],
                [uid],
                [weekid],
                [timeid],
                [dist],
                [holiday],
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
            'current_loc': 0,
            'current_tim': 1,
            'masked_current_tim': 0,
            'current_dis': 0,
            'speeds': 0,
            'speeds_relevant1': 0,
            'speeds_relevant2': 0,
            'speeds_long': 0,
            'grid_len': 0,
        }
        self.data_feature = {
            'traj_len_idx': self.traj_len_idx,
            'geo_embedding': self.geo_embedding,
            'uid_size': self.uid_size,
        }

    def gen_scalar_data_feature(self, train_data):
        longi_list = []
        lati_list = []
        dist_list = []
        time_list = []
        dist_gap_list = []
        time_gap_list = []
        speeds_list = []
        speeds_relevant1_list = []
        speeds_relevant2_list = []
        speeds_long_list = []
        grid_len_list = []
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
            speeds_list.extend(data[scalar_feature_column["speeds"]])
            speeds_relevant1_list.extend(data[scalar_feature_column["speeds_relevant1"]])
            speeds_relevant2_list.extend(data[scalar_feature_column["speeds_relevant2"]])
            speeds_long_list.extend(data[scalar_feature_column["speeds_long"]])
            grid_len_list.extend(data[scalar_feature_column["grid_len"]])
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
            'speeds_mean': np.mean(speeds_list),
            'speeds_std': np.std(speeds_list),
            'speeds_relevant1_mean': np.mean(speeds_relevant1_list),
            'speeds_relevant1_std': np.std(speeds_relevant1_list),
            'speeds_relevant2_mean': np.mean(speeds_relevant2_list),
            'speeds_relevant2_std': np.std(speeds_relevant2_list),
            'speeds_long_mean': np.mean(speeds_long_list),
            'speeds_long_std': np.std(speeds_long_list),
            'grid_len_mean': np.mean(grid_len_list),
            'grid_len_std': np.std(grid_len_list),
        }
        for k, v in scalar_data_feature.items():
            self._logger.info("{}: {}".format(k, v))
        return scalar_data_feature
