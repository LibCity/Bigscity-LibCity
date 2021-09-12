import pandas as pd
import numpy as np
import json
from sklearn.svm import SVR
from libcity.model.loss import *

config = {
    'dataset': 'METR_LA',
    'train_rate': 0.8,
    'seq_len': 12,
    'pre_len': 3
}


def preprocess_data(data, config):
    time_len = data.shape[0]
    num = config.get('num', time_len)
    train_rate = config.get('train_rate', 0.8)

    seq_len = config.get('seq_len', 12)
    pre_len = config.get('pre_len', 3)

    data = data[0:int(num)]

    time_len = data.shape[0]
    train_size = int(time_len * train_rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]

    length = data.shape[1]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data)):
        a = train_data[i]
        for j in range(length - seq_len - pre_len):
            a1 = a[j:j + seq_len + pre_len]
            trainX.append(a1[0:seq_len])
            trainY.append(a1[seq_len:seq_len + pre_len])
    for i in range(len(test_data)):
        b = test_data[i]
        for j in range(length - seq_len - pre_len):
            b1 = b[j:j + seq_len + pre_len]
            testX.append(b1[0:seq_len])
            testY.append(b1[seq_len:seq_len + pre_len])
    return trainX, trainY, testX, testY


def get_data(dataset):
    # path
    path = '../raw_data/' + dataset + '/'
    config_path = path + 'config.json'
    dyna_path = path + dataset + '.dyna'
    geo_path = path + dataset + '.geo'

    # read config
    with open(config_path, 'r') as f:
        json_obj = json.load(f)
        for key in json_obj:
            if key not in config:
                config[key] = json_obj[key]

    # read geo
    geo_file = pd.read_csv(geo_path)
    geo_ids = list(geo_file['geo_id'])

    # read dyna
    dyna_file = pd.read_csv(dyna_path)
    data_col = config.get('data_col', '')
    if data_col != '':  # 根据指定的列加载数据集
        if isinstance(data_col, list):
            data_col = data_col.copy()
        else:  # str
            data_col = [data_col].copy()
        data_col.insert(0, 'time')
        data_col.insert(1, 'entity_id')
        dyna_file = dyna_file[data_col]
    else:  # 不指定则加载所有列
        dyna_file = dyna_file[dyna_file.columns[2:]]  # 从time列开始所有列

    # 求时间序列
    time_slots = list(dyna_file['time'][:int(dyna_file.shape[0] / len(geo_ids))])

    idx_of_timeslots = dict()
    if not dyna_file['time'].isna().any():  # 时间没有空值
        time_slots = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), time_slots))
        time_slots = np.array(time_slots, dtype='datetime64[ns]')
        for idx, _ts in enumerate(time_slots):
            idx_of_timeslots[_ts] = idx

    # 转3-d数组
    feature_dim = len(dyna_file.columns) - 2
    df = dyna_file[dyna_file.columns[-feature_dim:]]
    len_time = len(time_slots)
    data = []
    for i in range(0, df.shape[0], len_time):
        data.append(df[i:i + len_time].values)
    data = np.array(data, dtype=float)  # (len(self.geo_ids), len_time, feature_dim)
    data = data.swapaxes(0, 1)  # (len_time, len(self.geo_ids), feature_dim)
    return data


def run_SVR(config, trainX, trainY, testX, testY):
    seq_len = config.get('seq_len', 12)
    pre_len = config.get('pre_len', 3)

    a_X = np.array(trainX)  # (, 12, 1)
    a_X = np.reshape(a_X, [-1, seq_len])  # (, 12)
    a_Y = np.array(trainY)  # (, 3, 1)
    a_Y = np.reshape(a_Y, [-1, pre_len])  # (, 3)
    a_Y = np.mean(a_Y, axis=1)  # (, 1)
    t_X = np.array(testX)  # (, 12, 1)
    t_X = np.reshape(t_X, [-1, seq_len])  # (, 12)
    t_Y = np.array(testY)  # (, 3, 1)
    t_Y = np.reshape(t_Y, [-1, pre_len])  # (, 3)
    t_Y = np.mean(t_Y, axis=1)

    print("----begin training----")
    svr_model = SVR(kernel='rbf')
    svr_model.fit(a_X, a_Y)
    print("----end training-----")
    print('=====================')
    print('=====================')
    print('=====================')
    print("----begin testing----")
    pre = svr_model.predict(t_X)  # (, 1)

    result1 = np.array(pre)
    testY1 = np.array(t_Y)

    y_true = np.array(testY1)  #(,3)
    y_pred = np.array(result1)  #(,3)

    # metric
    print('masked_MAE: ', masked_mae_np(y_pred, y_true, 0))
    print('masked_MAPE: ', masked_mape_np(y_pred, y_true, 0))
    print('masked_MSE: ', masked_mse_np(y_pred, y_true, 0))
    print('masked_RMSE: ', masked_rmse_np(y_pred, y_true, 0))

    print("----end testing----")


def main():
    data = get_data(config.get('dataset', ''))
    trainX, trainY, testX, testY = preprocess_data(data, config)
    run_SVR(config, trainX, trainY, testX, testY)


if __name__ == '__main__':
    main()
