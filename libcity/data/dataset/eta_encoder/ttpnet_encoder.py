import os
import numpy as np
import pandas as pd
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

from libcity.data.dataset.eta_encoder.abstract_eta_encoder import AbstractETAEncoder


parameter_list = [
    'dataset',
    'eta_encoder'
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
            'current_longi': 'float', 'current_lati': 'float',
            'current_loc': 'int',
            'current_tim': 'float', 'masked_current_tim': 'float', 'current_dis': 'float',
            'speeds': 'float', 'speeds_relevant1': 'float',
            'speeds_relevant2': 'float', 'speeds_long': 'float',
            'grid_len': 'float',
            'uid': 'int',
            'weekid': 'int',
            'timeid': 'int',
            'dist': 'float',
            'holiday': 'int',
            'time': 'float',
        }
        parameters_str = ''
        for key in parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'eta{}.json'.format(parameters_str))

        self.geo_embedding = []
        path = "./raw_data/{}/{}.geo".format(config['dataset'], config['dataset'])
        f_geo = pd.read_csv(path)
        self._logger.info("Loaded file " + self.config['dataset'] + '.geo')
        for row in f_geo.itertuples():
            embedding = eval(getattr(row, 'embeddings'))
            self.geo_embedding.append(embedding[:])

        self.uid_size = 0
        self.longi_list = []
        self.lati_list = []
        self.dist_list = []
        self.time_list = []
        self.dist_gap_list = []
        self.time_gap_list = []
        self.speeds_list = []
        self.speeds_relevant1_list = []
        self.speeds_relevant2_list = []
        self.speeds_long_list = []
        self.grid_len_list = []

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

            dist = traj[-1][dyna_feature_column["current_dis"]] - traj[0][dyna_feature_column["current_dis"]]
            self.dist_list.append(dist)

            begin_time = datetime.strptime(traj[0][dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')
            end_time = datetime.strptime(traj[-1][dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')
            weekid = int(begin_time.weekday())
            timeid = int(begin_time.strftime('%H')) * 4 + int(begin_time.strftime('%M')) // 15
            time = (end_time - begin_time).seconds
            self.time_list.append(time)

            holiday = 0
            if "holiday" in dyna_feature_column:
                holiday = traj[0][dyna_feature_column["holiday"]]
            elif weekid == 0 or weekid == 6:
                holiday = 1

            last_dis = 0
            last_tim = begin_time
            for point in traj:
                coordinate = eval(point[dyna_feature_column["coordinates"]])
                longi, lati = float(coordinate[0]), float(coordinate[1])

                current_longi.append(longi)
                self.longi_list.append(longi)
                current_lati.append(lati)
                self.lati_list.append(lati)

                loc = point[dyna_feature_column["location"]]
                current_loc.append(loc)

                speed = eval(point[dyna_feature_column["speeds"]])
                speeds.extend(speed)
                self.speeds_list.extend(speed)

                speed_relevant1 = eval(point[dyna_feature_column["speeds_relevant1"]])
                speeds_relevant1.extend(speed_relevant1)
                self.speeds_relevant1_list.extend(speed_relevant1)

                speed_relevant2 = eval(point[dyna_feature_column["speeds_relevant2"]])
                speeds_relevant2.extend(speed_relevant2)
                self.speeds_relevant2_list.extend(speed_relevant2)

                speed_long = eval(point[dyna_feature_column["speeds_long"]])
                speeds_long.extend(speed_long)
                self.speeds_long_list.extend(speed_long)

                len = point[dyna_feature_column["grid_len"]]
                grid_len.append(len)
                self.grid_len_list.append(len)

                if "current_dis" in dyna_feature_column:
                    dis = point[dyna_feature_column["current_dis"]]
                elif len(current_longi) == 1:
                    dis = 0
                else:
                    dis = geo_distance(current_longi[-2], current_lati[-2], longi, lati) + last_dis
                current_dis.append(dis)
                self.dist_gap_list.append(dis - last_dis)
                last_dis = dis

                tim = datetime.strptime(point[dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')
                current_tim.append(float((tim - begin_time).seconds))
                masked_current_tim.append(1)
                self.time_gap_list.append(float((tim - last_tim).seconds))
                last_tim = tim

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
            'geo_embedding': self.geo_embedding,
            'uid_size': self.uid_size,
            'longi_mean': np.mean(self.longi_list),
            'longi_std': np.std(self.longi_list),
            'lati_mean': np.mean(self.lati_list),
            'lati_std': np.std(self.lati_list),
            'dist_mean': np.mean(self.dist_list),
            'dist_std': np.std(self.dist_list),
            'time_mean': np.mean(self.time_list),
            'time_std': np.std(self.time_list),
            'dist_gap_mean': np.mean(self.dist_gap_list),
            'dist_gap_std': np.std(self.dist_gap_list),
            'time_gap_mean': np.mean(self.time_gap_list),
            'time_gap_std': np.std(self.time_gap_list),
            'speeds_mean': np.mean(self.speeds_list),
            'speeds_std': np.std(self.speeds_list),
            'speeds_relevant1_mean': np.mean(self.speeds_relevant1_list),
            'speeds_relevant1_std': np.std(self.speeds_relevant1_list),
            'speeds_relevant2_mean': np.mean(self.speeds_relevant2_list),
            'speeds_relevant2_std': np.std(self.speeds_relevant2_list),
            'speeds_long_mean': np.mean(self.speeds_long_list),
            'speeds_long_std': np.std(self.speeds_long_list),
            'grid_len_mean': np.mean(self.grid_len_list),
            'grid_len_std': np.std(self.grid_len_list),
        }
        self._logger.info("longi_mean: {}".format(self.data_feature["longi_mean"]))
        self._logger.info("longi_std : {}".format(self.data_feature["longi_std"]))
        self._logger.info("lati_mean : {}".format(self.data_feature["lati_mean"]))
        self._logger.info("lati_std  : {}".format(self.data_feature["lati_std"]))
        self._logger.info("dist_mean : {}".format(self.data_feature["dist_mean"]))
        self._logger.info("dist_std  : {}".format(self.data_feature["dist_std"]))
        self._logger.info("time_mean : {}".format(self.data_feature["time_mean"]))
        self._logger.info("time_std  : {}".format(self.data_feature["time_std"]))
        self._logger.info("dist_gap_mean : {}".format(self.data_feature["dist_gap_mean"]))
        self._logger.info("dist_gap_std  : {}".format(self.data_feature["dist_gap_std"]))
        self._logger.info("time_gap_mean : {}".format(self.data_feature["time_gap_mean"]))
        self._logger.info("time_gap_std  : {}".format(self.data_feature["time_gap_std"]))
        self._logger.info("speeds_mean : {}".format(self.data_feature["speeds_mean"]))
        self._logger.info("speeds_std  : {}".format(self.data_feature["speeds_std"]))
        self._logger.info("speeds_relevant1_mean : {}".format(self.data_feature["speeds_relevant1_mean"]))
        self._logger.info("speeds_relevant1_std  : {}".format(self.data_feature["speeds_relevant1_std"]))
        self._logger.info("speeds_relevant2_mean : {}".format(self.data_feature["speeds_relevant2_mean"]))
        self._logger.info("speeds_relevant2_std  : {}".format(self.data_feature["speeds_relevant2_std"]))
        self._logger.info("speeds_long_mean : {}".format(self.data_feature["speeds_long_mean"]))
        self._logger.info("speeds_long_std  : {}".format(self.data_feature["speeds_long_std"]))
        self._logger.info("grid_len_mean : {}".format(self.data_feature["grid_len_mean"]))
        self._logger.info("grid_len_std  : {}".format(self.data_feature["grid_len_std"]))
