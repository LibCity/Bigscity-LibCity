import numpy as np
import torch


# 均方误差（Mean Square Error）
def mse(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'MSE: 预测数据与真实数据大小不一致'
    return np.mean(sum(pow(loc_pred - loc_true, 2)))


# 平均绝对误差（Mean Absolute Error）
def mae(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'MAE: 预测数据与真实数据大小不一致'
    return np.mean(sum(loc_pred - loc_true))


# 均方根误差（Root Mean Square Error）
def rmse(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'RMSE: 预测数据与真实数据大小不一致'
    return np.sqrt(np.mean(sum(pow(loc_pred - loc_true, 2))))


# 平均绝对百分比误差（Mean Absolute Percentage Error）
def mape(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'MAPE: 预测数据与真实数据大小不一致'
    assert 0 not in loc_true, "MAPE: 真实数据有0，该公式不适用"
    return np.mean(abs(loc_pred - loc_true) / loc_true)


# 平均绝对和相对误差（Mean Absolute Relative Error）
def mare(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), "MARE：预测数据与真实数据大小不一致"
    assert np.sum(loc_true) != 0, "MARE：真实位置全为0，该公式不适用"
    return np.sum(np.abs(loc_pred - loc_true)) / np.sum(loc_true)


# 对称平均绝对百分比误差（Symmetric Mean Absolute Percentage Error）
def smape(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'SMAPE: 预测数据与真实数据大小不一致'
    assert 0 in (loc_pred + loc_true), "SMAPE: 预测数据与真实数据有0，该公式不适用"
    return 2.0 * np.mean(np.abs(loc_pred - loc_true) / (np.abs(loc_pred) +
                                                        np.abs(loc_true)))


# 对比真实位置与预测位置获得预测准确率
def acc(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), "accuracy: 预测数据与真实数据大小不一致"
    loc_diff = loc_pred - loc_true
    loc_diff[loc_diff != 0] = 1
    return loc_diff, np.mean(loc_diff == 0)


def top_k(loc_pred, loc_true, topk):
    """
    count the hit numbers of loc_true in topK of loc_pred, used to calculate
    Precision, Recall and F1-score
    calculate the reciprocal rank, used to calcualte MRR
    calculate the sum of DCG@K of the batch, used to calculate NDCG
    Args:
        loc_pred (batch_size * output_dim)
        loc_true (batch_size * 1)
    Return:
        hit (int): the hit numbers
        rank (float): the sum of the reciprocal rank of input batch
        dcg (float)
    """
    assert topk > 0, "top-k ACC评估方法：k值应不小于1"
    loc_pred = torch.FloatTensor(loc_pred)
    val, index = torch.topk(loc_pred, topk, 1)
    index = index.numpy()
    hit = 0
    rank = 0.0
    dcg = 0.0
    for i, p in enumerate(index):
        target = loc_true[i]
        if target in p:
            hit += 1
            rank_list = list(p)
            rank_index = rank_list.index(target)
            # rank_index is start from 0, so need plus 1
            rank += 1.0 / (rank_index + 1)
            dcg += 1.0 / np.log2(rank_index + 2)
    return hit, rank, dcg
