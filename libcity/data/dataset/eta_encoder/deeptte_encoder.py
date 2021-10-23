import os
import numpy as np
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
        }
        parameters_str = ''
        for key in parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'eta{}.json'.format(parameters_str))

        # self.geo_coord = dict()
        # path = "./raw_data/{}/{}.geo".format(config['dataset'], config['dataset'])
        # f_geo = open(path)
        # self._logger.info("Loaded file " + self.config['dataset'] + '.geo')
        # lines = f_geo.readlines()

        # for i, line in enumerate(lines):
        #     if i == 0:
        #         continue
        #     tokens = line.strip().replace("\"", "").replace("[", "").replace("]", "").split(',')

        #     loc_id, loc_longi, loc_lati = int(tokens[0]), float(tokens[2]), float(tokens[3])
        #     self.geo_coord[loc_id] = (loc_longi, loc_lati)
        # f_geo.close()

        self.uid_size = 0
        self.longi_list = []
        self.lati_list = []
        self.dist_list = []
        self.time_list = []
        self.dist_gap_list = []
        self.time_gap_list = []

    def encode(self, uid, trajectories, dyna_feature_column):
        self.uid_size = max(uid, self.uid_size)
        encoded_trajectories = []
        for traj in trajectories:
            current_longi = []
            current_lati = []
            current_tim = []
            current_dis = []
            current_state = []
            dist = traj[-1][dyna_feature_column["current_dis"]] - traj[0][dyna_feature_column["current_dis"]]
            self.dist_list.append(dist)
            begin_time = datetime.strptime(traj[0][dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')
            end_time = datetime.strptime(traj[-1][dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')
            weekid = int(begin_time.weekday())
            timeid = int(begin_time.strftime('%H'))*60 + int(begin_time.strftime('%M'))
            time = (end_time - begin_time).seconds
            self.time_list.append(time)
            last_dis = 0
            last_tim = begin_time
            for point in traj:
                # loc = point[dyna_feature_column["location"]]
                # longi, lati = self.geo_coord[loc][0], self.geo_coord[loc][1]
                coordinate = eval(point[dyna_feature_column["coordinates"]])
                longi, lati = float(coordinate[0]), float(coordinate[1])

                current_longi.append(longi)
                self.longi_list.append(longi)
                current_lati.append(lati)
                self.lati_list.append(lati)

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
                current_tim.append(float((tim - last_tim).seconds))
                self.time_gap_list.append(float((tim - last_tim).seconds))
                last_tim = tim

                if "current_state" in dyna_feature_column:
                    state = point[dyna_feature_column["current_state"]]
                else:
                    state = 0
                current_state.append(state)
            encoded_trajectories.append([
                current_longi, current_lati,
                current_tim, current_dis,
                current_state,
                [uid],
                [weekid],
                [timeid],
                [dist],
                [time],
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
            # 'uid_size': self.uid_size,
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
