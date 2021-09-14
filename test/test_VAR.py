import pandas as pd
import numpy as np
import json
from statsmodels.tsa.api import VAR
from libcity.utils import StandardScaler
import time
import sys
import os
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
from libcity.model.loss import masked_mae_np, masked_mape_np, masked_mse_np, masked_rmse_np, r2_score_np, explained_variance_score_np

config = {
    'dataset': 'METR_LA',
    'train_rate': 0.8,
    'seq_len': 12,
    'pre_len': 12,
    'maxlags': 1,
    'metrics': ['masked_MAE', 'masked_MSE', 'masked_RMSE', 'masked_MAPE', 'MAE', 'MSE', 'RMSE', 'MAPE', 'R2', 'EVAR']
}


def preprocess_data(data, config):
    train_rate = config.get('train_rate', 0.8)
    seq_len = config.get('seq_len', 12)
    pre_len = config.get('pre_len', 12)

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
    path = 'raw_data/' + dataset + '/'
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

    # 转3-d数组
    feature_dim = len(dyna_file.columns) - 2
    df = dyna_file[dyna_file.columns[-feature_dim:]]
    len_time = len(time_slots)
    data = []
    for i in range(0, df.shape[0], len_time):
        data.append(df[i:i + len_time].values)
    data = np.array(data, dtype=float)  # (num_nodes, len_time, feature_dim)
    data = data.swapaxes(0, 1)  # (len_time, num_nodes, feature_dim)
    return data


def run_VAR(config, data, testX, testY):
    print("----begin training----")
    ts, points = data.shape[:2]
    data = data.reshape(ts, -1)[:int(ts * 0.7)] + np.random.randn(int(ts * 0.7), points) / 10000
    scaler = StandardScaler(data.mean(), data.std())
    data = scaler.transform(data)

    s = time.time()
    model = VAR(data)
    maxlags = config.get("maxlag", 1)
    results = model.fit(maxlags=maxlags, ic='aic')
    e = time.time()
    print(1, e - s)

    seq_len = config.get('seq_len', 12)
    pre_len = config.get('pre_len', 12)
    testX = np.array(testX)
    testY = np.array(testY)
    testX = testX[:len(testX) // points * points].reshape(-1, seq_len, points)
    testY = testY[:len(testY) // points * points].reshape(-1, pre_len, points)
    print(testX.shape, testY.shape)  # B, T, N * F

    s = time.time()
    y_pred, y_true = [[] for i in range(pre_len)], [[] for i in range(pre_len)]
    for sample, target in zip(testX, testY):
        # print(sample.shape, target.shape)  T, N * F
        sample = scaler.transform(sample[-maxlags:])
        out = results.forecast(sample, pre_len)
        # print(out.shape) T, N * F
        out = scaler.inverse_transform(out)
        for i in range(pre_len):
            y_pred[i].append(out[i])
            y_true[i].append(target[i])
    e = time.time()
    print(2, e - s)
    y_pred = np.array(y_pred)  # T, B, N, F
    y_true = np.array(y_true)
    print("----end training-----")
    return y_pred, y_true


def evaluate(result, testy):
    metrics = config.get('metrics',
                         ['MAE', 'MAPE', 'MSE', 'RMSE', 'masked_MAE', 'masked_MAPE', 'masked_MSE', 'masked_RMSE', 'R2', 'EVAR'])
    df = []
    line = {}
    for metric in metrics:
        if metric == 'masked_MAE':
            line[metric] = masked_mae_np(result, testy, 0)
        elif metric == 'masked_MSE':
            line[metric] = masked_mse_np(result, testy, 0)
        elif metric == 'masked_RMSE':
            line[metric] = masked_rmse_np(result, testy, 0)
        elif metric == 'masked_MAPE':
            line[metric] = masked_mape_np(result, testy, 0)
        elif metric == 'MAE':
            line[metric] = masked_mae_np(result, testy)
        elif metric == 'MSE':
            line[metric] = masked_mse_np(result, testy)
        elif metric == 'RMSE':
            line[metric] = masked_rmse_np(result, testy)
        elif metric == 'MAPE':
            line[metric] = masked_mape_np(result, testy)
        elif metric == 'R2':
            line[metric] = r2_score_np(result, testy)
        elif metric == 'EVAR':
            line[metric] = explained_variance_score_np(result, testy)
        else:
            raise ValueError(
                'Error parameter evaluator_mode={}.'.format(metric))
    df.append(line)

    df = pd.DataFrame(df, columns=metrics)
    print(df)
    df.to_csv("result.csv")


def main():
    data = get_data(config.get('dataset', ''))
    trainX, trainY, testX, testY = preprocess_data(data, config)
    y_pred, y_true = run_VAR(config, data, testX, testY)
    evaluate(y_pred, y_true)


if __name__ == '__main__':
    main()
