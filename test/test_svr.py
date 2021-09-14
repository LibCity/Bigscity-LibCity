import pandas as pd
import numpy as np
from sklearn.svm import SVR
import sys
import os
import json
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
from libcity.model.loss import *


config = {
    'dataset': 'METR_LA',
    'train_rate': 0.7,
    'eval_rate': 0.1,
    'input_window': 12,
    'output_window': 3,
    'metrics': ['MAE', 'MAPE', 'MSE', 'RMSE', 'masked_MAE', 'masked_MAPE', 'masked_MSE', 'masked_RMSE', 'R2', 'EVAR']
}


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

    if not dyna_file['time'].isna().any():  # 时间没有空值
        time_slots = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), time_slots))
        time_slots = np.array(time_slots, dtype='datetime64[ns]')

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


def preprocess_data(data, config):
    time_len = data.shape[0]
    train_rate = config.get('train_rate', 0.8)
    eval_rate = config.get('eval_rate',0.1)

    input_window = config.get('input_window', 12)
    output_window = config.get('output_window', 3)

    train_size = int(time_len * (train_rate + eval_rate))
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]

    trainX, trainY, testX, testy = [], [], [], []
    for i in range(len(train_data) - input_window - output_window):
        a = train_data[i: i + input_window + output_window]
        trainX.append(a[0: input_window])
        trainY.append(a[input_window: input_window + output_window])

    for i in range(len(test_data) - input_window - output_window):
        b = test_data[i: i + input_window + output_window]
        testX.append(b[0: input_window])
        testy.append(b[input_window: input_window + output_window])
    return trainX, trainY, testX, testy


def test(result, testy, config):
    metrics = config.get('metrics',
            ['MAE', 'MAPE', 'MSE', 'RMSE', 'masked_MAE', 'masked_MAPE', 'masked_MSE', 'masked_RMSE', 'R2', 'EVAR'])
    time_len = testy.shape[0]
    df = []
    for i in range(time_len):
        line = {}
        for metric in metrics:
            if metric == 'masked_MAE':
                line[metric] = masked_mae_np(result[i, :], testy[i, :], 0)
            elif metric == 'masked_MSE':
                line[metric] = masked_mse_np(result[i, :], testy[i, :], 0)
            elif metric == 'masked_RMSE':
                line[metric] = masked_rmse_np(result[i, :], testy[i, :], 0)
            elif metric == 'masked_MAPE':
                line[metric] = masked_mape_np(result[i, :], testy[i, :], 0)
            elif metric == 'MAE':
                line[metric] = masked_mae_np(result[i, :], testy[i, :])
            elif metric == 'MSE':
                line[metric] = masked_mse_np(result[i, :], testy[i, :])
            elif metric == 'RMSE':
                line[metric] = masked_rmse_np(result[i, :], testy[i, :])
            elif metric == 'MAPE':
                line[metric] = masked_mape_np(result[i, :], testy[i, :])
            elif metric == 'R2':
                line[metric] = r2_score_np(result[i, :], testy[i, :])
            elif metric == 'EVAR':
                line[metric] = explained_variance_score_np(result[i, :], testy[i, :])
                pass
            else:
                raise ValueError(
                    'Error parameter evaluator_mode={}.'.format(metric))
        df.append(line)

    df = pd.DataFrame(df, columns=metrics)
    print(df)
    df.to_csv("test_result.csv")


def train(data, config):
    num_nodes = data.shape[1]  # num_nodes
    input_window = config.get("input_window", 12)
    output_window = config.get("output_window", 3)

    result = []
    testy = []

    for i in range(num_nodes):
        data1 = np.mat(data)
        a = data1[:, i]  # (time_len, feature)
        ax, ay, tx, ty = preprocess_data(a, config)
        ax = np.array(ax)  # (train_size, input_window, feature)
        ax = np.reshape(ax, [-1, input_window])  # (train_size * feature, input_window)
        ay = np.array(ay)  # (train_size, output_window, feature)
        ay = np.reshape(ay, [-1, output_window])  # (train_size * feature, output_window)
        ay = np.mean(ay, axis=1)  # (train_size,)
        tx = np.array(tx)  # (test_size, input_window, feature)
        tx = np.reshape(tx, [-1, input_window])  # (test_size * feature, input_window)
        ty = np.array(ty)  # (test_size, output_window, feature)
        ty = np.reshape(ty, [-1, output_window])  # (test_size * feature, output_window)
        svr_model = SVR(kernel='rbf')
        svr_model.fit(ax, ay)
        pre = svr_model.predict(tx)  # (test_size, )
        pre = np.array(np.transpose(np.mat(pre))) # (test_size, 1)
        pre = pre.repeat(output_window, axis=1)  # (test_size, output_window)
        result.append(pre)
        testy.append(ty)
        break

    result = np.array(result)  # (num_nodes, test_size, output_window)
    testy = np.array(testy)  # (num_nodes, test_size, output_window)
    result = result.transpose(1, 0, 2)  # (test_size, num_nodes, output_window)
    testy = testy.transpose(1, 0, 2)  # (test_size, num_nodes, output_window)

    return result, testy


def main():
    data = get_data(config.get('dataset', ''))
    result, testy = train(data, config)
    test(result, testy, config)


if __name__ == '__main__':
    main()
