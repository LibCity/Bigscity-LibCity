import sys
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVR
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
from libcity.evaluator.utils import evaluate_model
from libcity.utils import preprocess_data


config = {
    'model': 'SVR',
    'kernel': 'rbf',
    'dataset': 'METR_LA',
    'train_rate': 0.7,
    'eval_rate': 0.1,
    'input_window': 12,
    'output_window': 3,
    'metrics': ['MAE', 'MAPE', 'MSE', 'RMSE', 'masked_MAE',
                'masked_MAPE', 'masked_MSE', 'masked_RMSE', 'R2', 'EVAR']}


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


def run_SVR(data):
    ts, num_nodes, f = data.shape
    output_window = config.get("output_window", 3)
    kernel = config.get('kernel', 'rbf')

    y_pred = []
    y_true = []
    for i in tqdm(range(num_nodes), 'num_nodes'):
        trainx, trainy, testx, testy = preprocess_data(data[:, i, :], config)  # (T, F)
        # (train_size, in/out, F), (test_size, in/out, F)
        trainx = np.reshape(trainx, (trainx.shape[0], -1))  # (train_size, in * F)
        trainy = np.reshape(trainy, (trainy.shape[0], -1))  # (train_size, out * F)
        trainy = np.mean(trainy, axis=1)  # (train_size,)
        testx = np.reshape(testx, (testx.shape[0], -1))  # (test_size, in * F)
        print(trainx.shape, trainy.shape, testx.shape, testy.shape)

        svr_model = SVR(kernel=kernel)
        svr_model.fit(trainx, trainy)
        pre = svr_model.predict(testx)  # (test_size, )
        pre = np.expand_dims(pre, axis=1)  # (test_size, 1)
        pre = pre.repeat(output_window * f, axis=1)  # (test_size, out * F)
        y_pred.append(pre.reshape(pre.shape[0], output_window, f))
        y_true.append(testy)

    y_pred = np.array(y_pred)  # (N, test_size, out, F)
    y_true = np.array(y_true)  # (N, test_size, out, F)
    y_pred = y_pred.transpose((1, 2, 0, 3))  # (test_size, out, N, F)
    y_true = y_true.transpose((1, 2, 0, 3))  # (test_size, out, N, F)
    return y_pred, y_true


def main():
    print(config)
    data = get_data(config.get('dataset', ''))
    y_pred, y_true = run_SVR(data)
    evaluate_model(y_pred=y_pred, y_true=y_true, metrics=config['metrics'],
                   path=config['model']+'_'+config['dataset']+'_metrics.csv')


if __name__ == '__main__':
    main()
