import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

from libcity.model.loss import masked_mae_np, masked_mape_np, masked_mse_np, masked_rmse_np, r2_score_np, \
    explained_variance_score_np

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

config = {
    'p_range': [0, 4],
    'd_range': [0, 3],
    'q_range': [0, 4],

    'dataset': 'METR_LA',
    'train_rate': 0.7,
    'eval_rate': 0.1,
    'input_window': 12,
    'output_window': 3,

    'metrics': ['MAE', 'MAPE', 'MSE', 'RMSE', 'masked_MAE', 'masked_MAPE', 'masked_MSE', 'masked_RMSE', 'R2', 'EVAR']
}


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


def preprocess_data(data):
    train_rate = config.get('train_rate', 0.7)
    eval_rate = config.get('eval_rate', 0.1)

    input_window = config.get('input_window', 12)
    output_window = config.get('output_window', 3)

    x, y = [], []
    for i in range(len(data) - input_window - output_window):
        a = data[i: i + input_window + output_window]
        x.append(a[0: input_window])
        y.append(a[input_window: input_window + output_window])
    x = np.array(x)
    y = np.array(y)

    train_size = int(x.shape[0] * (train_rate + eval_rate))
    trainX = x[:train_size]
    trainY = y[:train_size]
    testX = x[train_size:x.shape[0]]
    testY = y[train_size:x.shape[0]]
    return trainX, trainY, testX, testY


def evaluate(result, testy):
    metrics = config.get('metrics',
                         ['MAE', 'MAPE', 'MSE', 'RMSE', 'masked_MAE', 'masked_MAPE', 'masked_MSE', 'masked_RMSE', 'R2',
                          'EVAR'])
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


# Try to find the best (p,d,q) parameters for ARIMA
def order_select_pred(data):
    # data: (T, F)
    res = ARIMA(data, order=(0, 0, 0)).fit()
    bic = res.bic
    p_range = config.get('p_range', [0, 4])
    d_range = config.get('d_range', [0, 3])
    q_range = config.get('q_range', [0, 4])
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=ConvergenceWarning)
        warnings.simplefilter("error", category=RuntimeWarning)
        for p in range(p_range[0], p_range[1]):
            for d in range(d_range[0], d_range[1]):
                for q in range(q_range[0], q_range[1]):
                    try:
                        cur_res = ARIMA(data, order=(p, d, q)).fit()
                    except:
                        continue
                    if cur_res.bic < bic:
                        bic = cur_res.bic
                        res = cur_res
    return res  # (T, F)


def arima(data):
    output_window = config.get('output_window', 3)
    y_pred = []  # (num_sequences, num_nodes, len_time, num_features)
    data = data.swapaxes(1, 2)  # (num_sequences, num_nodes, len_time, feature_dim)
    for time_slot in data:
        y_pred_ele = []  # (num_nodes, len_time, num_features)
        # Different nodes should be predict by different ARIMA models instance.
        for seq in time_slot:
            pred = order_select_pred(seq).forecast(steps=output_window)
            pred = pred.reshape((-1, seq.shape[1]))  # (len_time, num_features)
            y_pred_ele.append(pred)
        y_pred.append(y_pred_ele)
    return np.array(y_pred).swapaxes(1, 2)  # (num_sequences,  len_time, num_nodes, num_features)


def main():
    data = get_data(config.get('dataset', ''))
    trainX, trainY, testX, testY = preprocess_data(data)
    y_pred = arima(testX)
    evaluate(y_pred, testY)


if __name__ == '__main__':
    main()
