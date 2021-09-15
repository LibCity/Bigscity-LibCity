import json
import os

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from libcity.model.loss import masked_mae_np, masked_mape_np, masked_mse_np, masked_rmse_np, r2_score_np, \
    explained_variance_score_np

root_path = os.path.abspath(__file__)

config = {
    'dataset': 'METR_LA',
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
    return data


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
    res = ARIMA(data, (0, 0, 0)).fit()
    bic = res.bic
    for p in range(0, 4):
        for d in range(0, 3):
            for q in range(0, 4):
                try:
                    cur_res = ARIMA(data, (p, d, q)).fit()
                    if cur_res.bic < bic:
                        bic = cur_res.bic
                        res = cur_res
                except:
                    pass
    return res  # (T, F)


def arima(data: np.ndarray):
    y_pred = []
    y_true = []

    # Different nodes should be predict by different ARIMA models instance.
    for row in data:
        y_true.append(row)
        y_pred.append(order_select_pred(row))
    return y_pred, y_true  # (N, T, F)


def main():
    data = get_data(config.get('dataset', ''))
    y_pred, y_true = arima(data)
    evaluate(y_pred, y_true)


if __name__ == '__main__':
    main()
