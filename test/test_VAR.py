import pandas as pd
import numpy as np
import json
from statsmodels.tsa.api import VAR
from libcity.model.loss import *
import torch
from libcity.utils import StandardScaler
import time

config = {
    'dataset': 'METR_LA',
    'input_windows': 12,
    'output_windows': 12,
    'train_rate': 0.8,
    'seq_len': 12,
    'pre_len': 12,
    'metrics': ['masked_MAE', 'masked_MSE', 'masked_RMSE', 'masked_MAPE', 'MAE', 'MSE', 'RMSE', 'MAPE', 'R2', 'EVAR']
}


def preprocess_data(data, config):
    time_len = data.shape[0]
    num = config.get('num', time_len)
    train_rate = config.get('train_rate', 0.8)

    seq_len = config.get('seq_len', 12)
    pre_len = config.get('pre_len', 12)

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


def run_VAR(config, data, trainX, trainY, testX, testY):
    print("----begin training----")
    ts, points = data.shape[:2]
    data = data.reshape(ts, -1)[:int(ts * 0.7)] + np.random.randn(int(ts * 0.7), points) / 10000
    scaler = StandardScaler(data.mean(), data.std())
    data = scaler.transform(data)

    s = time.time()
    model = VAR(data)
    results = model.fit(maxlags=1, ic='aic')
    e = time.time()
    print(1, e - s)

    input_windows = config.get('input_windows', 12)
    output_windows = config.get('output_windows', 12)
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    trainX = trainX[:len(trainX) // points * points].reshape(-1, input_windows, points)
    trainY = trainY[:len(trainY) // points * points].reshape(-1, output_windows, points)
    print(trainX.shape, trainY.shape)  # B, T, N * F

    s = time.time()
    y_pred, y_true = [[] for i in range(12)], [[] for i in range(12)]
    for sample, target in zip(trainX, trainY):
        # print(sample.shape, target.shape)  T, N * F
        sample = scaler.transform(sample[-1:])
        out = results.forecast(sample, 12)
        # print(out.shape) T, N * F
        out = scaler.inverse_transform(out)
        for i in range(12):
            y_pred[i].append(out[i])
            y_true[i].append(target[i])
    e = time.time()
    print(2, e - s)
    y_pred = torch.Tensor(y_pred)  # T, B, N, F
    y_true = torch.Tensor(y_true)
    print("----end training-----")
    print('=====================')
    print('=====================')
    print('=====================')
    print("----begin testing----")
    df = []
    len_timeslots = 12
    for i in range(1, len_timeslots + 1):
        line = {}
        for metric in config['metrics']:
            if metric == 'masked_MAE':
                line[metric] = masked_mae_torch(y_pred[:, i - 1], y_true[:, i - 1], 0).item()
            elif metric == 'masked_MSE':
                line[metric] = masked_mse_torch(y_pred[:, i - 1], y_true[:, i - 1], 0).item()
            elif metric == 'masked_RMSE':
                line[metric] = masked_rmse_torch(y_pred[:, i - 1], y_true[:, i - 1], 0).item()
            elif metric == 'masked_MAPE':
                line[metric] = masked_mape_torch(y_pred[:, i - 1], y_true[:, i - 1], 0).item()
            elif metric == 'MAE':
                line[metric] = masked_mae_torch(y_pred[:, i - 1], y_true[:, i - 1]).item()
            elif metric == 'MSE':
                line[metric] = masked_mse_torch(y_pred[:, i - 1], y_true[:, i - 1]).item()
            elif metric == 'RMSE':
                line[metric] = masked_rmse_torch(y_pred[:, i - 1], y_true[:, i - 1]).item()
            elif metric == 'MAPE':
                line[metric] = masked_mape_torch(y_pred[:, i - 1], y_true[:, i - 1]).item()
            elif metric == 'R2':
                line[metric] = r2_score_torch(y_pred[:, i - 1], y_true[:, i - 1]).item()
            elif metric == 'EVAR':
                line[metric] = explained_variance_score_torch(y_pred[:, i - 1], y_true[:, i - 1]).item()
            else:
                raise ValueError('Error parameter evaluator_mode={}, please set `single` or `average`.'.format('single'))
        df.append(line)

    df = pd.DataFrame(df, columns=config['metrics'])
    print(df)
    df.to_csv("sz_metrics.csv")

    print("----end testing----")


def main():
    data = get_data(config.get('dataset', ''))
    trainX, trainY, testX, testY = preprocess_data(data, config)
    run_VAR(config, data, trainX, trainY, testX, testY)


if __name__ == '__main__':
    main()
