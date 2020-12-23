import numpy as np
import torch

# 均方误差（Mean Square Error）
def MSE(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'MSE: 预测数据与真实数据大小不一致'
    return np.mean(sum(pow(loc_pred - loc_true, 2)))


# 平均绝对误差（Mean Absolute Error）
def MAE(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'MAE: 预测数据与真实数据大小不一致'
    return np.mean(sum(loc_pred - loc_true))


# 均方根误差（Root Mean Square Error）
def RMSE(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'RMSE: 预测数据与真实数据大小不一致'
    return np.sqrt(np.mean(sum(pow(loc_pred - loc_true, 2))))


# 平均绝对百分比误差（Mean Absolute Percentage Error）
def MAPE(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'MAPE: 预测数据与真实数据大小不一致'
    assert 0 not in loc_true, "MAPE: 真实数据有0，该公式不适用"
    return np.mean(abs(loc_pred - loc_true) / loc_true)


# 平均绝对和相对误差（Mean Absolute Relative Error）
def MARE(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), "MARE：预测数据与真实数据大小不一致"
    assert np.sum(loc_true) != 0, "MARE：真实位置全为0，该公式不适用"
    return np.sum(np.abs(loc_pred - loc_true)) / np.sum(loc_true)


# 对称平均绝对百分比误差（Symmetric Mean Absolute Percentage Error）
def SMAPE(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'SMAPE: 预测数据与真实数据大小不一致'
    assert 0 in (loc_pred + loc_true), "SMAPE: 预测数据与真实数据有0，该公式不适用"
    return 2.0 * np.mean(np.abs(loc_pred - loc_true) / (np.abs(loc_pred) + np.abs(loc_true)))


# 对比真实位置与预测位置获得预测准确率
def ACC(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), "accuracy: 预测数据与真实数据大小不一致"
    loc_diff = loc_pred - loc_true
    loc_diff[loc_diff != 0] = 1
    return loc_diff, np.mean(loc_diff == 0)


# 对比真实位置与模型预测的前k个位置获得预测准确率
def top_k(loc_pred, loc_true, topK):
    '''
    Args:
        loc_pred (batch_size * output_dim)
        loc_true (batch_size * 1)
    Return:
        For a single data in batch, if loc_true is in topk(loc_pred) return 1, else return 0
        res: (batch_size * 1)
    '''
    assert topK > 0, "top-k ACC评估方法：k值应不小于1"
    loc_pred = torch.LongTensor(loc_pred)
    val, index = torch.topk(loc_pred, topK, 1) # 使用 torch 的 topk 来实现
    index = index.numpy()
    res = []
    for i, p in enumerate(index):
        target = loc_true[i]
        if target in p:
            res.append(1)
        else:
            res.append(0)
    return res

