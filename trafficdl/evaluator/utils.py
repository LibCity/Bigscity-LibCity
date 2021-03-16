import json
from heapq import nlargest


def output(method, value, field):
    """
    :param method: 评估方法
    :param value: 对应评估方法的评估结果值
    :param field: 评估的范围, 对一条轨迹或是整个模型
    """
    if method == 'ACC':
        if field == 'model':
            print('---- 该模型在 {} 评估方法下 avg_acc={:.3f} ----'.format(method,
                                                                  value))
        else:
            print('{} avg_acc={:.3f}'.format(method, value))
    elif method in ['MSE', 'RMSE', 'MAE', 'MAPE', 'MARE', 'SMAPE']:
        if field == 'model':
            print('---- 该模型在 {} 评估方法下 avg_loss={:.3f} ----'.format(method,
                                                                   value))
        else:
            print('{} avg_loss={:.3f}'.format(method, value))
    else:
        if field == 'model':
            print('---- 该模型在 {} 评估方法下 avg_acc={:.3f} ----'.format(method,
                                                                  value))
        else:
            print('{} avg_acc={:.3f}'.format(method, value))


def transfer_data(data, model, maxk):
    """
    Here we transform specific data types to standard input type
    """
    if type(data) == str:
        data = json.loads(data)
    assert type(data) == dict, "待评估数据的类型/格式不合法"
    if model == 'DeepMove':
        user_idx = data.keys()
        for user_id in user_idx:
            trace_idx = data[user_id].keys()
            for trace_id in trace_idx:
                trace = data[user_id][trace_id]
                loc_pred = trace['loc_pred']
                new_loc_pred = []
                for t_list in loc_pred:
                    new_loc_pred.append(sort_confidence_ids(t_list, maxk))
                data[user_id][trace_id]['loc_pred'] = new_loc_pred
    return data


def sort_confidence_ids(confidence_list, threshold):
    """
    Here we convert the prediction results of the DeepMove model
    DeepMove model output: confidence of all locations
    Evaluate model input: location ids based on confidence
    :param threshold: maxK
    :param confidence_list:
    :return: ids_list
    """
    """sorted_list = sorted(confidence_list, reverse=True)
    mark_list = [0 for i in confidence_list]
    ids_list = []
    for item in sorted_list:
        for i in range(len(confidence_list)):
            if confidence_list[i] == item and mark_list[i] == 0:
                mark_list[i] = 1
                ids_list.append(i)
                break
        if len(ids_list) == threshold:
            break
    return ids_list"""
    max_score_with_id = nlargest(
        threshold, enumerate(confidence_list), lambda x: x[1])
    return list(map(lambda x: x[0], max_score_with_id))
