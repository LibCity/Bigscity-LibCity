from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_executor, get_model, get_logger
from libcity.data.utils import generate_dataloader
# from geopy import distance
from math import radians, cos, sin, asin, sqrt
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
# import json
f = open('./raw_data/foursquare_cut_one_day.pkl', 'rb')
data = pickle.load(f)
# 要把它的数据放到 Batch 里面
"""
data_neural: {
    uid: {
        sessions: {
            session_id: [
                [loc, tim]
            ]
        }
    }
}

vid_lookup 来进行距离的计算，所以还是在这里完成 encode 操作吧
"""
data_neural = data['data_neural']
user_set = data['data_neural'].keys()
vid_lookup = data['vid_lookup']
tim_max = 47
pad_item = {
    'current_loc': 9296,
    'current_tim': tim_max+1
}


def geodistance(lat1, lng1, lat2, lng2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon = lng2-lng1
    dlat = lat2-lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance = 2*asin(sqrt(a))*6371*1000
    distance = round(distance/1000, 3)
    return distance


def _create_dilated_rnn_input(current_loc):
    current_loc.reverse()
    sequence_length = len(current_loc)
    session_dilated_rnn_input_index = [0] * sequence_length
    for i in range(sequence_length - 1):
        current_poi = current_loc[i]
        poi_before = current_loc[i + 1:]
        lon_cur, lat_cur = vid_lookup[current_poi][1], vid_lookup[current_poi][0]
        distance_row_explicit = []
        for target in poi_before:
            lon, lat = vid_lookup[target][1], vid_lookup[target][0]
            distance_row_explicit.append(geodistance(lat_cur, lon_cur, lat, lon))
        index_closet = np.argmin(distance_row_explicit).item()
        # reverse back
        session_dilated_rnn_input_index[sequence_length - i - 1] = sequence_length - 2 - index_closet - i
    current_loc.reverse()
    return session_dilated_rnn_input_index


def _gen_distance_matrix(current_loc, history_loc_central):
    # 使用 profile 计算当前位置与历史轨迹中心点之间的距离
    history_avg_distance = []  # history_session_count
    now_loc = current_loc[-1]
    lon_cur, lat_cur = vid_lookup[now_loc][1], vid_lookup[now_loc][0]
    for central in history_loc_central:
        dis = geodistance(central[0], central[1], lat_cur, lon_cur)
        if dis < 1:
            dis = 1
        history_avg_distance.append(dis)
    return history_avg_distance


encoded_data = {}

feature_dict = {'history_loc': 'array of int', 'history_tim': 'array of int',
                'current_loc': 'int', 'current_tim': 'int', 'dilated_rnn_input_index': 'no_pad_int',
                'history_avg_distance': 'no_pad_float',
                'target': 'int', 'uid': 'int'}

time_checkin_set = defaultdict(set)

for uid in tqdm(user_set, desc="encoding data"):
    history_loc = []
    history_tim = []
    history_loc_central = []
    encoded_trajectories = []
    sessions = data_neural[uid]['sessions']
    for session_id in sessions.keys():
        current_session = sessions[session_id]
        current_loc = []
        current_tim = []
        for p in current_session:
            current_loc.append(p[0])
            current_tim.append(p[1])
            if p[1] > tim_max:
                print('tim overleaf error')
                break
            if p[1] not in time_checkin_set:
                time_checkin_set[p[1]] = set()
            time_checkin_set[p[1]].add(p[0])
        if session_id == 0:
            history_loc.append(current_loc)
            history_tim.append(current_tim)
            lon = []
            lat = []
            for poi in current_loc:
                lon_cur = vid_lookup[poi][1]
                lat_cur = vid_lookup[poi][0]
                lon.append(lon_cur)
                lat.append(lat_cur)
            history_loc_central.append((np.mean(lat), np.mean(lon)))
            continue
        trace = []
        target = current_loc[-1]
        dilated_rnn_input_index = _create_dilated_rnn_input(current_loc[:-1])
        history_avg_distance = _gen_distance_matrix(current_loc[:-1], history_loc_central)
        trace.append(history_loc.copy())
        trace.append(history_tim.copy())
        trace.append(current_loc[:-1])
        trace.append(current_tim[:-1])
        trace.append(dilated_rnn_input_index)
        trace.append(history_avg_distance)
        trace.append(target)
        trace.append(uid)
        encoded_trajectories.append(trace)
        history_loc.append(current_loc)
        history_tim.append(current_tim)
        # calculate current_loc
        lon = []
        lat = []
        for poi in current_loc:
            lon_cur, lat_cur = vid_lookup[poi][1], vid_lookup[poi][0]
            lon.append(lon_cur)
            lat.append(lat_cur)
        history_loc_central.append((np.mean(lat), np.mean(lon)))
    encoded_data[str(uid)] = encoded_trajectories

sim_matrix = np.zeros((tim_max+1, tim_max+1))
for i in range(tim_max+1):
    sim_matrix[i][i] = 1
    for j in range(i+1, tim_max+1):
        set_i = time_checkin_set[i]
        set_j = time_checkin_set[j]
        if len(set_i | set_j) != 0:
            jaccard_ij = len(set_i & set_j) / len(set_i | set_j)
            sim_matrix[i][j] = jaccard_ij
            sim_matrix[j][i] = jaccard_ij


# with open('./lstpm_test_data.json', 'w') as f:
#     json.dump({
#         'encoded_data': encoded_data,
#         'sim_matrix': sim_matrix
#     }, f)


config = ConfigParser('traj_loc_pred', 'LSTPM', 'foursquare_tky', other_args={"history_type": 'cut_off', "gpu_id": 0,
                                                                              "metrics": ["Recall", "NDCG"], "topk": 5})
logger = get_logger(config)
dataset = get_dataset(config)
dataset.data = {
    'encoded_data': encoded_data
}
dataset.pad_item = pad_item
train_data, eval_data, test_data = dataset.divide_data()
train_data, eval_data, test_data = generate_dataloader(train_data, eval_data, test_data,
                                                       feature_dict, config['batch_size'],
                                                       config['num_workers'], pad_item,
                                                       {})

data_feature = {
    'loc_size': 9297,
    'tim_size': tim_max + 2,
    'uid_size': 934,
    'loc_pad': 9296,
    'tim_pad': tim_max + 1,
    'tim_sim_matrix': sim_matrix.tolist()
}

model = get_model(config, data_feature)
# batch = train_data.__iter__().__next__()
# batch.to_tensor(config['device'])
executor = get_executor(config, model)
executor.train(train_data, eval_data)
executor.evaluate(test_data)
