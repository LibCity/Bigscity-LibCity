import csv
import datetime

import numpy as np


def get_range_date():
    start_time = datetime.date(2020, 1, 1)
    end_time = datetime.date(2021, 2, 28)
    day_range = list()
    for i in range((end_time - start_time).days + 1):
        day = start_time + datetime.timedelta(days=i)
        day_range.append(str(day))

    return day_range


data = np.load('./datacache.npy')

num_nodes = data.shape[1]
timeslot = get_range_date()
len_times = len(timeslot)
# O, D, T
dyna_file = open('./odcrn.od', 'w')
writer = csv.writer(dyna_file)
writer.writerow(["dyna_id", "type", "time", "origin_id", "destination_id", "flow"])

dyna_id = 0
for i in range(num_nodes):
    for j in range(num_nodes):
        for t in range(len_times):
            writer.writerow([dyna_id, 'state', timeslot[t] + 'T00:00:00Z', i, j, np.log(data[t, i, j, 0] + 1)])
            dyna_id += 1

od_file = open('./odcrn.geo', 'w')
writer = csv.writer(od_file)
writer.writerow(['geo_id', 'type', 'coordinates'])
for i in range(num_nodes):
    writer.writerow([i, 'Point', "[]"])

adj = np.load('./adjcache.npy')
rel_file = open('./odcrn.rel', 'w')
writer = csv.writer(rel_file)
writer.writerow(['rel_id', 'type', 'origin_id', 'destination_id', 'cost'])

rel_id = 0
for i in range(num_nodes):
    for j in range(num_nodes):
        if adj[i, j] != 0:
            writer.writerow([rel_id, 'geo', i, j])
            rel_id += 1
