import pandas as pd
import numpy as np
import json
import sys
import os
from statsmodels.tsa.api import VAR
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
from libcity.utils import StandardScaler
from libcity.evaluator.utils import evaluate_model
from libcity.utils import preprocess_data


config = {
    'model': 'VAR',
    'maxlags': 1,
    'dataset': 'METR_LA',
    'train_rate': 0.7,
    'eval_rate': 0.1,
    'input_window': 12,
    'output_windows': 3,
    'metrics': ['MAE', 'MAPE', 'MSE', 'RMSE', 'masked_MAE',
                'masked_MAPE', 'masked_MSE', 'masked_RMSE', 'R2', 'EVAR']
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
    data = np.array(data, dtype=float)  # (N, T, F)
    data = data.swapaxes(0, 1)  # (T, N, F)
    return data


def run_VAR(data, inputs):
    ts, points, f = data.shape
    train_rate = config.get('train_rate', 0.7)
    eval_rate = config.get('eval_rate', 0.1)
    output_window = config.get('output_window', 3)
    maxlags = config.get("maxlag", 1)

    data = data.reshape(ts, -1)[:int(ts * (train_rate + eval_rate))]  # (train_size, N * F)
    scaler = StandardScaler(data.mean(), data.std())
    data = scaler.transform(data)

    model = VAR(data)
    results = model.fit(maxlags=maxlags, ic='aic')

    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1)  # (num_samples, out, N * F)
    y_pred = []  # (num_samples, out, N, F)
    for sample in inputs:  # (out, N * F)
        sample = scaler.transform(sample[-maxlags:])  # (T, N, F)
        out = results.forecast(sample, output_window)  # (out, N * F)
        out = scaler.inverse_transform(out)  # (out, N * F)
        y_pred.append(out.reshape(output_window, points, f))
    y_pred = np.array(y_pred)  # (num_samples, out, N, F)
    return y_pred


def main():
    print(config)
    data = get_data(config.get('dataset', ''))
    trainx, trainy, testx, testy = preprocess_data(data, config)
    y_pred = run_VAR(data, testx)
    evaluate_model(y_pred=y_pred, y_true=testy, metrics=config['metrics'],
                   path=config['model']+'_'+config['dataset']+'_metrics.csv')


if __name__ == '__main__':
    main()
