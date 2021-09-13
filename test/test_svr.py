import pandas as pd
import numpy as np
from sklearn.svm import SVR
from libcity.model.loss import *


def preprocess_data(data, config):
    time_len = data.shape[0]
    train_rate = config.get('train_rate', 0.8)

    input_window = config.get('input_window', 12)
    output_window = config.get('output_window', 3)

    train_size = int(time_len * train_rate)
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


def evaluate(result, testy):
    metrics = ['MAE', 'MAPE', 'MSE', 'RMSE', 'masked_MAE', 'masked_MAPE', 'masked_MSE', 'masked_RMSE', 'R2', 'EVAR']
    time_len = testy.shape[0]
    df = []
    print("----begin testing----")
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
    df.to_csv("sz_metrics.csv")


def train(data, config):
    num_nodes = data.shape[1]  # num_nodes 156
    input_window = config.get("input_window", 12)
    output_window = config.get("output_window", 3)

    result = []
    testy = []
    print("----begin training----")
    for i in range(num_nodes):
        data1 = np.mat(data)
        a = data1[:, i]  # (time_len, feature) = (2365,1)
        ax, ay, tx, ty = preprocess_data(a, config)
        ax = np.array(ax)  # (train_size, input_window, feature) = (2365,12,1)
        ax = np.reshape(ax, [-1, input_window])  # (train_size, input_window * feature) = (156,12)
        ay = np.array(ay)  # (train_size, output_window, feature) = (2365, 3, 1)
        ay = np.reshape(ay, [-1, output_window])  # (train_size, output_window * feature) = (2365, 3)
        ay = np.mean(ay, axis=1)  # (train_size,) = (2365,)
        tx = np.array(tx)  # (test_size, input_window, feature) = (581, 12, 1)
        tx = np.reshape(tx, [-1, input_window])  # (test_size, input_window * feature) = (581, 12)
        ty = np.array(ty)  # (test_size, output_window, feature) = (581, 3 ,1)
        ty = np.reshape(ty, [-1, output_window])  # (test_size, output_window * feature) = (581, 3)
        svr_model = SVR(kernel='rbf')
        svr_model.fit(ax, ay)
        pre = svr_model.predict(tx)  # (test_size, feature) = (581, )
        pre = np.array(np.transpose(np.mat(pre)))  # (test_size, 1)
        ty = np.mean(ty, axis=1)  # (test_size, )
        ty = np.array(np.transpose(np.mat(ty)))  # (test_size, 1)
        result.append(pre)
        testy.append(ty)

    print("----end training-----")

    print('=====================')
    print('=====================')
    print('=====================')

    result = np.array(result)  # (num_nodes, test_size, feature) = (156, 581, 1)
    testy = np.array(testy)  # (num_nodes, test_size, feature) = (156, 581, 1)

    result = result.transpose(1, 0, 2)  # (test_size, num_nodes, feature) = (581, 156, 1)
    testy = testy.transpose(1, 0, 2)  # (test_size, num_nodes, feature) = (581, 156, 1)

    return result, testy


def main():
    config = {
        'dataset': 'METR_LA',
        'train_rate': 0.8,
        'input_window': 12,
        'output_window': 3
    }
    data = pd.read_csv("./sz_speed.csv")
    # trainX, trainY, testX, testy = preprocess_data(data, config)
    # print(len(trainX), len(trainY), len(testX), len(testy))
    # 2365 2365 581 581
    # print(trainX[0].shape, trainY[0].shape, testX[0].shape, testy[0].shape)
    # (12, 156) (3, 156) (12, 156) (3, 156)
    result, testy = train(data, config)
    evaluate(result, testy)


if __name__ == '__main__':
    main()
