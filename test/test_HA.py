import pandas as pd
import numpy as np
import json
from libcity.model.loss import masked_mae_np, masked_mape_np, masked_mse_np, masked_rmse_np

config = {
    'dataset': 'METR_LA',
    'lag': [7*24*12, 24*12, 1],
    'weight': [0.2, 0, 0.8]
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


def historical_average(data):
    t, p, f = data.shape
    lag = config.get('lag', 7 * 24 * 12)
    weight = config.get('weight', 1)
    if isinstance(lag, int):
        lag = [lag]
        weight = [1]
    else:
        assert isinstance(weight, list)
        assert sum(weight) == 1
    y_true = []
    y_pred = []
    for i in range(int(t * 0.8), t):
        # y_true
        y_true.append(data[i, :, :])

        # y_pred
        y_pred_i = 0
        for j in range(len(lag)):
            y_pred_i += weight[j] * np.mean(data[i - 4 * lag[j]:i:lag[j], :, :], axis=0)
        y_pred.append(y_pred_i)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # metric
    print('masked_MAE: ', masked_mae_np(y_pred, y_true, 0))
    print('masked_MAPE: ', masked_mape_np(y_pred, y_true, 0))
    print('masked_MSE: ', masked_mse_np(y_pred, y_true, 0))
    print('masked_RMSE: ', masked_rmse_np(y_pred, y_true, 0))


def main():
    data = get_data(config.get('dataset', ''))
    historical_average(data)


if __name__ == '__main__':
    main()
