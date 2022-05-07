import pandas as pd
import json
from libcity.utils.utils import ensure_dir
import os


class VisHelper:
    def __init__(self, _config):

        self.config = _config
        self.raw_path = './raw_data/'
        self.dataset = _config.get("dataset", "")
        self.save_path = _config.get("save_path", "./visualized_data/")

        # get type
        self.config_path = self.raw_path + self.dataset + '/config.json'
        self.data_config = json.load(open(self.config_path, 'r'))
        if 'dyna' in self.data_config and ['state'] == self.data_config['dyna']['including_types']:
            self.type = 'state'
        elif 'grid' in self.data_config and ['state'] == self.data_config['grid']['including_types']:
            self.type = 'grid'
        elif 'dyna' in self.data_config and ['trajectory'] == self.data_config['dyna']['including_types']:
            self.type = 'trajectory'
        else:
            self.type = 'geo'
        # get geo and dyna files
        all_files = os.listdir(self.raw_path + self.dataset)
        self.geo_file = []
        self.geo_path = None
        self.dyna_file = []
        self.dyna_path = None
        self.grid_file = []
        self.grid_path = None
        for file in all_files:
            if file.split('.')[1] == 'geo':
                self.geo_file.append(file)
            if file.split('.')[1] == 'dyna':
                self.dyna_file.append(file)
            if file.split('.')[1] == 'grid':
                self.grid_file.append(file)

        # reserved columns
        self.geo_reserved_lst = ['type', 'coordinates']
        self.dyna_reserved_lst = ['dyna_id', 'type', 'time', 'entity_id', 'traj_id', 'coordinates']
        self.grid_reserved_lst = ['dyna_id', 'type', 'time', 'row_id', 'column_id']

    def visualize(self):
        if self.type == 'trajectory':
            # geo
            if len(self.geo_file) > 0:
                self.geo_path = self.raw_path + self.dataset + '/' + self.geo_file[0]
                self._visualize_geo()

            # dyna
            for dyna_file in self.dyna_file:
                self.dyna_path = self.raw_path + self.dataset + '/' + dyna_file
                self._visualize_trajectory()

        elif self.type == 'state':
            self.geo_path = self.raw_path + self.dataset + '/' + self.geo_file[0]
            for dyna_file in self.dyna_file:
                self.dyna_path = self.raw_path + self.dataset + '/' + dyna_file
                self._visualize_state()
        elif self.type == 'grid':
            self.geo_path = self.raw_path + self.dataset + '/' + self.geo_file[0]
            for grid_file in self.grid_file:
                self.grid_path = self.raw_path + self.dataset + '/' + grid_file
                self._visualize_grid()
        elif self.type == 'geo':
            # geo
            self.geo_path = self.raw_path + self.dataset + '/' + self.geo_file[0]
            self._visualize_geo()

    def _visualize_state(self):
        geo_file = pd.read_csv(self.geo_path, index_col=None)
        dyna_file = pd.read_csv(self.dyna_path, index_col=None)
        geojson_obj = {'type': "FeatureCollection", 'features': []}

        # get feature_lst
        geo_feature_lst = [_ for _ in list(geo_file.columns) if _ not in self.geo_reserved_lst]
        dyna_feature_lst = [_ for _ in list(dyna_file.columns) if _ not in self.dyna_reserved_lst]

        for _, row in geo_file.iterrows():

            # get feature dictionary
            geo_id = row['geo_id']
            feature_dct = row[geo_feature_lst].to_dict()
            dyna_i = dyna_file[dyna_file['entity_id'] == geo_id]
            for f in dyna_feature_lst:
                feature_dct[f] = float(dyna_i[f].mean())

            # form a feature
            feature_i = dict()
            feature_i['type'] = 'Feature'
            feature_i['properties'] = feature_dct
            feature_i['geometry'] = {}
            feature_i['geometry']['type'] = row['type']
            feature_i['geometry']['coordinates'] = eval(row['coordinates'])
            geojson_obj['features'].append(feature_i)

        ensure_dir(self.save_path)
        save_name = "_".join(self.dyna_path.split('/')[-1].split('.')) + '.json'
        print(f"visualization file saved at {save_name}")
        json.dump(geojson_obj, open(self.save_path + '/' + save_name, 'w',
                                    encoding='utf-8'),
                  ensure_ascii=False, indent=4)

    def _visualize_grid(self):
        geo_file = pd.read_csv(self.geo_path, index_col=None)
        grid_file = pd.read_csv(self.grid_path, index_col=None)
        geojson_obj = {'type': "FeatureCollection", 'features': []}

        # get feature_lst
        geo_feature_lst = [_ for _ in list(geo_file.columns) if _ not in self.geo_reserved_lst]
        grid_feature_lst = [_ for _ in list(grid_file.columns) if _ not in self.grid_reserved_lst]

        for _, row in geo_file.iterrows():

            # get feature dictionary
            row_id, column_id = row['row_id'], row['column_id']
            feature_dct = row[geo_feature_lst].to_dict()
            dyna_i = grid_file[(grid_file['row_id'] == row_id) & (grid_file['column_id'] == column_id)]
            for f in grid_feature_lst:
                feature_dct[f] = float(dyna_i[f].mean())

            # form a feature
            feature_i = dict()
            feature_i['type'] = 'Feature'
            feature_i['properties'] = feature_dct
            feature_i['geometry'] = {}
            feature_i['geometry']['type'] = row['type']
            feature_i['geometry']['coordinates'] = eval(row['coordinates'])
            geojson_obj['features'].append(feature_i)

        ensure_dir(self.save_path)
        save_name = "_".join(self.grid_path.split('/')[-1].split('.')) + '.json'
        print(f"visualization file saved at {save_name}")
        json.dump(geojson_obj, open(self.save_path + '/' + save_name, 'w',
                                    encoding='utf-8'),
                  ensure_ascii=False, indent=4)

    def _visualize_geo(self):
        geo_file = pd.read_csv(self.geo_path, index_col=None)
        if "coordinates" not in list(geo_file.columns):
            return
        geojson_obj = {'type': "FeatureCollection", 'features': []}
        extra_feature = [_ for _ in list(geo_file.columns) if _ not in self.geo_reserved_lst]
        for _, row in geo_file.iterrows():
            feature_dct = row[extra_feature].to_dict()
            feature_i = dict()
            feature_i['type'] = 'Feature'
            feature_i['properties'] = feature_dct
            feature_i['geometry'] = {}
            feature_i['geometry']['type'] = row['type']
            feature_i['geometry']['coordinates'] = eval(row['coordinates'])
            if len(feature_i['geometry']['coordinates']) == 0:
                return
            geojson_obj['features'].append(feature_i)

        ensure_dir(self.save_path)
        save_name = "_".join(self.geo_path.split('/')[-1].split('.')) + '.json'
        print(f"visualization file saved at {save_name}")
        json.dump(geojson_obj, open(self.save_path + '/' + save_name, 'w',
                                    encoding='utf-8'),
                  ensure_ascii=False, indent=4)

    def _visualize_trajectory(self):
        dyna_file = pd.read_csv(self.dyna_path, index_col=None)
        geojson_obj = {'type': "FeatureCollection", 'features': []}
        trajectory = {}
        GPS_traj = "coordinates" in dyna_file.columns
        if not GPS_traj:
            geo_file = pd.read_csv(self.geo_path, index_col=None)

        a = dyna_file.groupby("entity_id")
        for entity_id, entity_value in a:
            if "traj_id" in dyna_file.columns:
                trajectory[entity_id] = {}
                entity_value = entity_value.groupby("traj_id")
                for traj_id, traj_value in entity_value:
                    feature_dct = {"usr_id": entity_id, "traj_id": traj_id}
                    feature_i = dict()
                    feature_i['type'] = 'Feature'
                    feature_i['properties'] = feature_dct
                    feature_i['geometry'] = {}
                    feature_i['geometry']['type'] = "LineString"
                    feature_i['geometry']['coordinates'] = []
                    if GPS_traj:
                        for _, row in traj_value.iterrows():
                            feature_i['geometry']['coordinates'].append(eval(row['coordinates']))
                    else:
                        for _, row in traj_value.iterrows():
                            coor = eval(geo_file.loc[row['location']]['coordinates'])
                            if _ == 0:
                                feature_i['geometry']['coordinates'].append(coor[0])
                            feature_i['geometry']['coordinates'].append(coor[1])
                    geojson_obj['features'].append(feature_i)

            else:
                feature_dct = {"usr_id": entity_id}
                feature_i = dict()
                feature_i['type'] = 'Feature'
                feature_i['properties'] = feature_dct
                feature_i['geometry'] = {}
                feature_i['geometry']['type'] = "LineString"
                feature_i['geometry']['coordinates'] = []
                if GPS_traj:
                    for _, row in entity_value.iterrows():
                        feature_i['geometry']['coordinates'].append(eval(row['coordinates']))
                else:
                    for _, row in entity_value.iterrows():
                        coor = eval(geo_file.loc[row['location']]['coordinates'])
                        if _ == 0:
                            feature_i['geometry']['coordinates'].append(coor[0])
                        feature_i['geometry']['coordinates'].append(coor[1])
                geojson_obj['features'].append(feature_i)

        ensure_dir(self.save_path)
        save_name = "_".join(self.dyna_path.split('/')[-1].split('.')) + '.json'
        print(f"visualization file saved at {save_name}")
        json.dump(geojson_obj, open(self.save_path + '/' + save_name, 'w',
                                    encoding='utf-8'),
                  ensure_ascii=False, indent=4)
