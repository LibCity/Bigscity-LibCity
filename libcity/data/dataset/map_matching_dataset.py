import pickle

from libcity.data.dataset import AbstractDataset
import os
import pandas as pd
import numpy as np
from logging import getLogger
from libcity.utils.dataset import parse_time
from libcity.utils.GPS_utils import dist, angle2radian
from libcity.utils.utils import ensure_dir
import networkx as nx


class MapMatchingDataset(AbstractDataset):
    """
    路网匹配数据集的基类。
    """

    def __init__(self, config):

        # config and dataset name
        self.config = config
        self.dataset = self.config.get('dataset', '')

        # logger
        self._logger = getLogger()

        # features
        self.with_time = config.get('with_time', True)  # 输入轨迹数据是否包含时间
        self.delta_time = config.get('delta_time', True)  # True则轨迹输入时间差(s)，False则轨迹输入时间datetime.datetime
        self.with_rd_speed = ('speed' in config['rel']['geo'].keys())

        # cache
        self.cache_dataset = self.config.get('cache_dataset', True)
        self.parameters_str = \
            str(self.dataset) + '_' + str(self.delta_time)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'map_matching_{}.pkl'.format(self.parameters_str))
        self.cache_file_folder = './libcity/cache/dataset_cache/'
        ensure_dir(self.cache_file_folder)

        # ensure dataset
        self.data_path = './raw_data/' + self.dataset + '/'
        if not os.path.exists(self.data_path):
            raise ValueError("Dataset {} not exist! Please ensure the path "
                             "'./raw_data/{}/' exist!".format(self.dataset, self.dataset))

        # related file names
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        self.usr_file = self.config.get('usr_file', self.dataset)
        self.route_file = self.config.get('usr_file', self.dataset)

        # result
        self.trajectory = None
        self.rd_nwk = None
        self.route = None

        # load 5 files
        if not self.cache_dataset or not os.path.exists(self.cache_file_name):
            if os.path.exists(self.data_path + self.geo_file + '.geo'):
                self._load_geo()
            else:
                raise ValueError('Not found .geo file!')

            if os.path.exists(self.data_path + self.rel_file + '.rel'):
                self._load_rel()
            else:
                raise ValueError('Not found .rel file!')

            if os.path.exists(self.data_path + self.usr_file + '.usr'):
                self._load_usr()
            else:
                raise ValueError('Not found .rel file!')

            if os.path.exists(self.data_path + self.dyna_file + '.dyna'):
                self._load_dyna()
            else:
                raise ValueError('Not found .dyna file!')
            if os.path.exists(self.data_path + self.route_file + '.route'):
                self._load_route()

    def _load_geo(self):
        """
        加载.geo文件，格式[geo_id, type, coordinates, properties(若干列)]
        Returns:
                self.geo_data: a dictionary, key: geo_id, value: [lon, lat]
        """
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        if not ['Point'] == self.config['geo']['including_types']:
            raise ValueError('.geo file should include Point type in Map Matching task!')

        self.geo_data = dict()
        for index, row in geofile.iterrows():
            self.geo_data[row[0]] = eval(row[2])
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_nodes=' + str(len(self.geo_data)))

    def _load_rel(self):
        """
        加载.rel文件，格式[rel_id, type, origin_id, destination_id, properties(若干列)],
        .rel文件用来表示路网数据

        Returns:
            self.rd_nwk: networkx.MultiDiGraph
        """

        # load rel file
        relfile = pd.read_csv(self.data_path + self.rel_file + '.rel')

        # check type geo
        if not ['geo'] == self.config['rel']['including_types']:
            raise ValueError('.rel file should include geo type in Map Matching task!')

        # get properties
        columns = relfile.columns.tolist()[4:]

        # init road network
        self.rd_nwk = nx.DiGraph(name="road network")

        # generate MultiDigraph
        for index, row in relfile.iterrows():

            # origin and destination
            rel_id = int(row[0])
            origin_id = int(row[2])
            destination_id = int(row[3])

            # is valid origin/destination
            if origin_id not in self.geo_data.keys():
                raise ValueError('origin_id %d should be in geo_ids in Map Matching task!' % origin_id)
            if destination_id not in self.geo_data.keys():
                raise ValueError('destination_id %d should be in geo_ids in Map Matching task!' % destination_id)

            # add node
            if origin_id not in self.rd_nwk.nodes:
                self.rd_nwk.add_node(origin_id, lon=self.geo_data[origin_id][0], lat=self.geo_data[origin_id][1])
            if destination_id not in self.rd_nwk.nodes:
                self.rd_nwk.add_node(destination_id, lon=self.geo_data[destination_id][0],
                                     lat=self.geo_data[destination_id][1])

            # add edge
            self.rd_nwk.add_edge(origin_id, destination_id)
            dct = dict()
            for i, column in enumerate(columns):
                dct[column] = row[i + 4]
            if 'distance' not in dct.keys():
                dct['distance'] = dist(
                    angle2radian(self.rd_nwk.nodes[origin_id]['lat']),
                    angle2radian(self.rd_nwk.nodes[origin_id]['lon']),
                    angle2radian(self.rd_nwk.nodes[destination_id]['lat']),
                    angle2radian(self.rd_nwk.nodes[destination_id]['lon'])
                )
            dct['rel_id'] = rel_id
            self.rd_nwk.edges[origin_id, destination_id].update(dct)

        # logger
        self._logger.info("Loaded file " + self.rel_file + '.rel, num_roads=' + str(len(self.rd_nwk)))

    def _load_usr(self):
        """
        加载.usr文件， 格式 [usr_id]
        Returns:
            np.ndarray: self.usr_lst 用户id的集合
        """
        usrfile = pd.read_csv(self.data_path + self.usr_file + '.usr')
        self.usr_lst = []
        for index, row in usrfile.iterrows():
            self.usr_lst.append(row[0])

        self._logger.info("Loaded file " + self.rel_file + '.usr, num_users=' + str(len(self.usr_lst)))

    def _load_dyna(self):
        """
        加载.dyna文件，格式 [dyna_id,type,time,entity_id,location]
        self.with_time 用于表示轨迹是否包含时间信息
        Returns:
            np.ndarray: 数据数组
        """
        dynafile = pd.read_csv(self.data_path + self.dyna_file + '.dyna')
        if not ['trajectory'] == self.config['dyna']['including_types']:
            raise ValueError('.dyna file should include trajectory type in Map Matching task!')
        if not self.config['dyna']['trajectory']["entity_id"] == "usr_id":
            raise ValueError('entity_id should be usr_id in Map Matching task!')
        if not self.config['dyna']['trajectory']["location"] == "geo_id":
            raise ValueError('location should be geo_id in Map Matching task!')

        self.trajectory = {}
        self.multi_traj = 'traj_id' in dynafile.keys()
        for index, row in dynafile.iterrows():
            if row['entity_id'] not in self.usr_lst:
                raise ValueError('entity_id %d should be in usr_ids in Map Matching task!' % row['entity_id'])
            if row['location'] not in self.geo_data.keys():
                raise ValueError('location %d should be in geo_ids in Map Matching task!' % row['location'])
            traj_id = row['traj_id'] if self.multi_traj else 0
            if self.with_time:
                if row['entity_id'] in self.trajectory.keys():
                    if traj_id in self.trajectory[row['entity_id']].keys():
                        self.trajectory[row['entity_id']][traj_id].append(
                            [row['dyna_id']] + self.geo_data[row['location']] + [parse_time(row['time'])])
                    else:
                        self.trajectory[row['entity_id']][traj_id] = \
                            [[row['dyna_id']] + self.geo_data[row['location']] + [parse_time(row['time'])]]
                else:
                    self.trajectory[row['entity_id']] = \
                        {traj_id: [[row['dyna_id']] + self.geo_data[row['location']] + [parse_time(row['time'])]]}

            else:
                if row['entity_id'] in self.trajectory.keys():
                    if traj_id in self.trajectory[row['entity_id']].keys():
                        self.trajectory[row['entity_id']][traj_id].append(
                            [row['dyna_id']] + self.geo_data[row['location']])
                    else:
                        self.trajectory[row['entity_id']][traj_id] = [[row['dyna_id']] + self.geo_data[row['location']]]
                else:
                    self.trajectory[row['entity_id']] = {traj_id: [[row['dyna_id']] + self.geo_data[row['location']]]}
        if self.delta_time and self.with_time:
            for usr_id, trajectory in self.trajectory.items():
                t0 = trajectory[0][3]
                trajectory[0][3] = 0
                for i in range(1, len(trajectory)):
                    trajectory[i][3] = (trajectory[i][3] - t0).seconds

        for key, value in self.trajectory.items():
            for key_i, value_i in value.items():
                self.trajectory[key][key_i] = np.array(value_i)

        self._logger.info("Loaded file " + self.dyna_file + '.dyna, num of GPS samples=' + str(dynafile.shape[0]))

    def _load_route(self):
        """
        加载.dyna文件，格式: 每行一个 rel_id 或一组 rel_id
        Returns:

        """
        routefile = pd.read_csv(self.data_path + self.route_file + '.route')
        self.route = {}
        multi_traj = 'traj_id' in routefile.keys()
        if multi_traj != self.multi_traj:
            raise ValueError('cannot match traj_id in route file and dyna file')
        for index, row in routefile.iterrows():
            usr_id = int(row['usr_id'])
            if usr_id not in self.usr_lst:
                raise ValueError('usr_id %d should be in usr_ids in Map Matching task!' % usr_id)
            traj_id = row['traj_id'] if multi_traj else 0

            if usr_id in self.route.keys():
                if traj_id in self.route[usr_id].keys():
                    self.route[usr_id][traj_id].append([row['route_id'], row['rel_id']])
                else:
                    self.route[usr_id][traj_id] = [[row['route_id'], row['rel_id']]]
            else:
                self.route[usr_id] = {traj_id: [[row['route_id'], row['rel_id']]]}

        for key, value in self.route.items():
            for key_i, value_i in value.items():
                self.route[key][key_i] = np.array(value_i)

        self._logger.info("Loaded file " + self.route_file + '.route, route length=' + str(routefile.shape[0]))

    def get_data(self):
        """
        返回训练数据、验证数据、测试数据
        对于MapMatching，训练数据和验证数据为None。
        Returns:
            dictionary:
                {
                'trajectory': np.array (time, lon, lat) if with_time else (lon, lat)
                'rd_nwk': networkx.MultiDiGraph
                'route': ground truth, numpy array
                }
        """
        if self.cache_dataset and os.path.exists(self.cache_file_name):
            self._logger.info('Loading ' + self.cache_file_name)
            with open(self.cache_file_name, 'rb') as f:
                res = pickle.load(f)
                self.multi_traj = res['multi_traj']
            return None, None, res
        res = dict()
        res['trajectory'] = self.trajectory
        res['rd_nwk'] = self.rd_nwk
        res['route'] = self.route
        res['multi_traj'] = self.multi_traj
        with open(self.cache_file_name, 'wb') as f:
            pickle.dump(res, f)
        self._logger.info('Saved at ' + self.cache_file_name)
        return None, None, res

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        res = dict()
        res['with_time'] = self.with_time
        res['with_rd_speed'] = self.with_rd_speed
        res['delta_time'] = self.delta_time
        res['multi_traj'] = self.multi_traj
        return res
