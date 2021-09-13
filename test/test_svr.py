import pandas as pd
import numpy as np
from sklearn.svm import SVR
from libcity.model.loss import *
from libcity.data import get_dataset


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


def evaluate(result, testy, config):
    metrics =  config.get('matrics', ['MAE', 'MAPE', 'MSE', 'RMSE', 'masked_MAE', 'masked_MAPE', 'masked_MSE', 'masked_RMSE', 'R2', 'EVAR'])
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
        a = data1[:, i]  # (time_len, feature)
        ax, ay, tx, ty = preprocess_data(a, config)
        ax = np.array(ax)  # (train_size, input_window, feature)
        ax = np.reshape(ax, [-1, input_window])  # (train_size, input_window * feature)
        ay = np.array(ay)  # (train_size, output_window, feature)
        ay = np.reshape(ay, [-1, output_window])  # (train_size, output_window * feature)
        ay = np.mean(ay, axis=1)  # (train_size,)
        tx = np.array(tx)  # (test_size, input_window, feature)
        tx = np.reshape(tx, [-1, input_window])  # (test_size, input_window * feature)
        ty = np.array(ty)  # (test_size, output_window, feature)
        ty = np.reshape(ty, [-1, output_window])  # (test_size, output_window * feature)
        svr_model = SVR(kernel='rbf')
        svr_model.fit(ax, ay)
        pre = svr_model.predict(tx)  # (test_size, feature)
        pre = np.array(np.transpose(np.mat(pre)))
        ty = np.mean(ty, axis=1)  # (test_size, )
        ty = np.array(np.transpose(np.mat(ty)))  # (test_size, 1)
        result.append(pre)
        testy.append(ty)


    print("----end training-----")

    print('=====================')
    print('=====================')
    print('=====================')

    result = np.array(result)  # (num_nodes, test_size, feature)
    testy = np.array(testy)  # (num_nodes, test_size, feature)

    result = result.transpose(1, 0, 2)  # (test_size, num_nodes, feature)
    testy = testy.transpose(1, 0, 2)  # (test_size, num_nodes, feature)

    return result, testy


def main():
    config = {
        'dataset': 'METR_LA',
        'train_rate': 0.8,
        'input_window': 12,
        'output_window': 3,
        'dataset_class': 'TrafficStatePointDataset',

        'metrics' : ['MAE', 'MAPE', 'MSE', 'RMSE', 'masked_MAE', 'masked_MAPE', 'masked_MSE', 'masked_RMSE', 'R2', 'EVAR']
    }
    dataset = get_dataset(config)
    data = dataset._load_dyna_3d("METR_LA")
    data = data.squeeze()
    # trainX, trainY, testX, testy = preprocess_data(data, config)
    # print(len(trainX), len(trainY), len(testX), len(testy))
    #27402 27402 6840 6840
    # print(trainX[0].shape, trainY[0].shape, testX[0].shape, testy[0].shape)
    #(12, 207) (3, 207) (12, 207) (3, 207)
    # exit(0)

    result, testy = train(data, config)
    evaluate(result, testy, config)


if __name__ == '__main__':
    main()
