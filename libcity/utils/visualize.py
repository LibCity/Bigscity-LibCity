import pandas as pd
import json
from utils import ensure_dir
import os

config = {
    "dataset_name": "00000000_30s",
    "save_path": "../../visualized_data/"
}


class VisHelper:
    def __init__(self, _config):
        self.config = _config
        self.raw_path = '../../raw_data/'
        self.dataset_name = config.get("dataset_name", "")
        self.save_path = config.get("save_path", "../../visualized_data/")
        all_files = os.listdir(self.raw_path + self.dataset_name)
        self.geo_file = []
        self.geo_path = None
        self.dyna_file = []
        self.dyna_path = None
        for file in all_files:
            if file.split('.')[1] == 'geo':
                self.geo_file.append(file)
            if file.split('.')[1] == 'dyna':
                self.dyna_file.append(file)

        assert len(self.geo_file) == 1

    def visualize(self):
        # geo
        self.geo_path = self.raw_path + self.dataset_name + '/' + self.geo_file[0]
        self._visualize_geo()
        for dyna_file in self.dyna_file:
            self.dyna_path = self.raw_path + self.dataset_name + '/' + dyna_file
            self._visualize_dyna()

    def _visualize_geo(self):
        geo_file = pd.read_csv(self.geo_path, index_col=None)
        geojson_obj = {'type': "FeatureCollection", 'features': []}
        extra_feature = geo_file.columns[3:]
        for _, row in geo_file.iterrows():
            feature_dct = row[extra_feature].to_dict()
            feature_i = dict()
            feature_i['type'] = 'Feature'
            feature_i['properties'] = feature_dct
            feature_i['geometry'] = {}
            feature_i['geometry']['type'] = row['type']
            feature_i['geometry']['coordinates'] = eval(row['coordinates'])
            geojson_obj['features'].append(feature_i)

        ensure_dir(self.save_path)
        save_name = "_".join(self.geo_path.split('/')[-1].split('.')) + '.json'
        json.dump(geojson_obj, open(self.save_path + '/' + save_name, 'w',
                                    encoding='utf-8'),
                  ensure_ascii=False, indent=4)

    def _visualize_dyna(self):
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
        json.dump(geojson_obj, open(self.save_path + '/' + save_name, 'w',
                                    encoding='utf-8'),
                  ensure_ascii=False, indent=4)


if __name__ == '__main__':
    vis_helper = VisHelper(config)
    vis_helper.visualize()
