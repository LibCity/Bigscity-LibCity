import pandas as pd
import numpy as np
import json
import sys
import os
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
from libcity.model.loss import masked_mae_np, masked_mape_np, masked_mse_np, masked_rmse_np, r2_score_np, explained_variance_score_np

config = {
    'dataset': 'METR_LA',
    'lag': [7*24*12, 24*12, 1],
    'weight': [0.2, 0, 0.8],
    'train_rate': 0.7,
    'eval_rate': 0.1,
    'n_sample': 4,
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
    data = data.swapaxes(0, 1)  # (len_time, len(self.geo_ids), feature_dim)
    return data


def historical_average(data):
    t, p, f = data.shape
    lag = config.get('lag', 7 * 24 * 12)
    train_rate = config.get('train_rate', 0.7)
    eval_rate = config.get('eval_rate', 0.1)
    weight = config.get('weight', 1)
    n_sample = config.get('n_sample', 4)
    if isinstance(lag, int):
        lag = [lag]
        weight = [1]
    else:
        assert isinstance(weight, list)
        assert sum(weight) == 1
    y_true = []
    y_pred = []
    for i in range(int(t * (train_rate + eval_rate)), t):
        # y_true
        y_true.append(data[i, :, :])

        # y_pred
        y_pred_i = 0
        for j in range(len(lag)):
            y_pred_i += weight[j] * np.mean(data[i - n_sample * lag[j]:i:lag[j], :, :], axis=0)
        y_pred.append(y_pred_i)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
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
    y_pred, y_true = historical_average(data)
    evaluate(y_pred, y_true)


if __name__ == '__main__':
    main()
