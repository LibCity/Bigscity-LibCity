import json
import warnings
import os
import sys
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
from libcity.evaluator.utils import evaluate_model
from libcity.utils import preprocess_data


config = {
    'model': 'ARIMA',
    'p_range': [0, 4],
    'd_range': [0, 3],
    'q_range': [0, 4],
    'dataset': 'METR_LA',
    'train_rate': 0.7,
    'eval_rate': 0.1,
    'input_window': 12,
    'output_window': 3,
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
    return res


def arima(data):
    output_window = config.get('output_window', 3)
    y_pred = []  # (num_samples, N, out, F)
    data = data.swapaxes(1, 2)  # (num_samples, N, out, F)
    for time_slot in tqdm(data, 'ts'):  # (N, out, F)
        y_pred_ele = []  # (N, out, F)
        # Different nodes should be predict by different ARIMA models instance.
        for seq in time_slot:  # (out, F)
            pred = order_select_pred(seq).forecast(steps=output_window)
            pred = pred.reshape((-1, seq.shape[1]))  # (out, F)
            y_pred_ele.append(pred)
        y_pred.append(y_pred_ele)
    return np.array(y_pred).swapaxes(1, 2)  # (num_samples, out, N, F)


def main():
    print(config)
    data = get_data(config.get('dataset', ''))
    trainx, trainy, testx, testy = preprocess_data(data, config)
    y_pred = arima(testx)
    evaluate_model(y_pred=y_pred, y_true=testy, metrics=config['metrics'],
                   path=config['model']+'_'+config['dataset']+'_metrics.csv')


if __name__ == '__main__':
    main()