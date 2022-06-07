import csv
import datetime
import json
import os.path

import numpy as np
import pandas as pd
import scipy.sparse as ss


def gen_config_info(file_name, interval):
    info = \
        {
            "data_col": [
                "flow"
            ],
            "data_files": [
                file_name
            ],
            "geo_file": file_name,
            "rel_file": file_name,
            "output_dim": 1,
            "init_weight_inf_or_zero": "inf",
            "set_weight_link_or_dist": "dist",
            "calculate_weight_adj": True,
            "weight_adj_epsilon": 0.1,
            "time_intervals": interval
        }
    return info


def gen_config_geo():
    geo = {
        "including_types": [
            "Point"
        ],
        "Point": {
        }
    }
    return geo


def gen_config_dyna():
    dyna = {
        "including_types": [
            "state"
        ],
        "state": {
            "entity_id": "geo_id",
            "inflow": "num",
            "outflow": "num"
        }
    }
    return dyna


def gen_config_od():
    od = {
        "including_types": [
            "state"
        ],
        "state": {
            "origin_id": "geo_id",
            "destination_id": "geo_id",
            "flow": "num"
        }
    }
    return od


def gen_config(output_dir_flow, file_name, interval):
    config = {}
    data = json.loads(json.dumps(config))
    data["geo"] = gen_config_geo()
    data['od'] = gen_config_od()
    data["info"] = gen_config_info(file_name, interval)
    config = json.dumps(data)
    with open(output_dir_flow + "/config.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(config)


if not os.path.exists('odcrn'):
    os.mkdir('odcrn')

ODPATH = './od_day20180101_20210228.npz'
OD_DAYS = [date.strftime('%Y-%m-%d') for date in pd.date_range(start='2020-01-01', end='2021-02-28', freq='1D')]
prov_day_data = ss.load_npz(ODPATH)
prov_day_data_dense = np.array(prov_day_data.todense()).reshape((-1, 47, 47))
data = prov_day_data_dense[-len(OD_DAYS):, :, :, np.newaxis]
len_times, num_nodes = data.shape[0:2]
# O, D, T
dyna_file = open('./odcrn/odcrn.od', 'w')
writer = csv.writer(dyna_file)
writer.writerow(["dyna_id", "type", "time", "origin_id", "destination_id", "flow"])

dyna_id = 0
for i in range(num_nodes):
    for j in range(num_nodes):
        for t in range(len_times):
            # TODO 论文对其进行了对数处理
            writer.writerow([dyna_id, 'state', '', i, j, data[t, i, j, 0]])
            dyna_id += 1

od_file = open('./odcrn/odcrn.geo', 'w')
writer = csv.writer(od_file)
writer.writerow(['geo_id', 'type', 'coordinates'])
for i in range(num_nodes):
    writer.writerow([i, 'Point', "[]"])

adj = np.load('./adjacency_matrix.npy')
rel_file = open('./odcrn/odcrn.rel', 'w')
writer = csv.writer(rel_file)
writer.writerow(['rel_id', 'type', 'origin_id', 'destination_id', 'cost'])

rel_id = 0
for i in range(num_nodes):
    for j in range(num_nodes):
        if adj[i, j] != 0:
            writer.writerow([rel_id, 'geo', i, j])
            rel_id += 1

# TODO interval 是多少
gen_config('.', 'odcrn', 86400)
