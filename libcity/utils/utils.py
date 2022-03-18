import importlib
import logging
import datetime
import os
import sys
import numpy as np
import random
import torch


def get_executor(config, model, data_feature):
    """
    according the config['executor'] to create the executor

    Args:
        config(ConfigParser): config
        model(AbstractModel): model

    Returns:
        AbstractExecutor: the loaded executor
    """
    try:
        return getattr(importlib.import_module('libcity.executor'),
                       config['executor'])(config, model, data_feature)
    except AttributeError:
        raise AttributeError('executor is not found')


def get_model(config, data_feature):
    """
    according the config['model'] to create the model

    Args:
        config(ConfigParser): config
        data_feature(dict): feature of the data

    Returns:
        AbstractModel: the loaded model
    """
    if config['task'] == 'traj_loc_pred':
        try:
            return getattr(importlib.import_module('libcity.model.trajectory_loc_prediction'),
                           config['model'])(config, data_feature)
        except AttributeError:
            raise AttributeError('model is not found')
    elif config['task'] == 'traffic_state_pred':
        try:
            return getattr(importlib.import_module('libcity.model.traffic_flow_prediction'),
                           config['model'])(config, data_feature)
        except AttributeError:
            try:
                return getattr(importlib.import_module('libcity.model.traffic_speed_prediction'),
                               config['model'])(config, data_feature)
            except AttributeError:
                try:
                    return getattr(importlib.import_module('libcity.model.traffic_demand_prediction'),
                                   config['model'])(config, data_feature)
                except AttributeError:
                    try:
                        return getattr(importlib.import_module('libcity.model.traffic_od_prediction'),
                                       config['model'])(config, data_feature)
                    except AttributeError:
                        try:
                            return getattr(importlib.import_module('libcity.model.traffic_accident_prediction'),
                                           config['model'])(config, data_feature)
                        except AttributeError:
                            raise AttributeError('model is not found')
    elif config['task'] == 'map_matching':
        try:
            return getattr(importlib.import_module('libcity.model.map_matching'),
                           config['model'])(config, data_feature)
        except AttributeError:
            raise AttributeError('model is not found')
    elif config['task'] == 'road_representation':
        try:
            return getattr(importlib.import_module('libcity.model.road_representation'),
                           config['model'])(config, data_feature)
        except AttributeError:
            raise AttributeError('model is not found')
    elif config['task'] == 'eta':
        try:
            return getattr(importlib.import_module('libcity.model.eta'),
                           config['model'])(config, data_feature)
        except AttributeError:
            raise AttributeError('model is not found')
    else:
        raise AttributeError('task is not found')


def get_evaluator(config):
    """
    according the config['evaluator'] to create the evaluator

    Args:
        config(ConfigParser): config

    Returns:
        AbstractEvaluator: the loaded evaluator
    """
    try:
        return getattr(importlib.import_module('libcity.evaluator'),
                       config['evaluator'])(config)
    except AttributeError:
        raise AttributeError('evaluator is not found')


def get_logger(config, name=None):
    """
    获取Logger对象

    Args:
        config(ConfigParser): config
        name: specified name

    Returns:
        Logger: logger
    """
    log_dir = './libcity/log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = '{}-{}-{}-{}.log'.format(config['exp_id'],
                                            config['model'], config['dataset'], get_local_time())
    logfilepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)

    log_level = config.get('log_level', 'INFO')

    if log_level.lower() == 'info':
        level = logging.INFO
    elif log_level.lower() == 'debug':
        level = logging.DEBUG
    elif log_level.lower() == 'error':
        level = logging.ERROR
    elif log_level.lower() == 'warning':
        level = logging.WARNING
    elif log_level.lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(logfilepath)
    file_handler.setFormatter(formatter)

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info('Log directory: %s', log_dir)
    return logger


def get_local_time():
    """
    获取时间

    Return:
        datetime: 时间
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur


def ensure_dir(dir_path):
    """Make sure the directory exists, if it does not exist, create it.

    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def trans_naming_rule(origin, origin_rule, target_rule):
    """
    名字转换规则

    Args:
        origin (str): 源命名格式下的变量名
        origin_rule (str): 源命名格式，枚举类
        target_rule (str): 目标命名格式，枚举类

    Return:
        target (str): 转换之后的结果
    """
    # TODO: 请确保输入是符合 origin_rule，这里目前不做检查
    target = ''
    if origin_rule == 'upper_camel_case' and target_rule == 'under_score_rule':
        for i, c in enumerate(origin):
            if i == 0:
                target = c.lower()
            else:
                target += '_' + c.lower() if c.isupper() else c
        return target
    else:
        raise NotImplementedError(
            'trans naming rule only support from upper_camel_case to \
                under_score_rule')


def preprocess_data(data, config):
    """
    split by input_window and output_window

    Args:
        data: shape (T, ...)

    Returns:
        np.ndarray: (train_size/test_size, input_window, ...)
                    (train_size/test_size, output_window, ...)

    """
    train_rate = config.get('train_rate', 0.7)
    eval_rate = config.get('eval_rate', 0.1)

    input_window = config.get('input_window', 12)
    output_window = config.get('output_window', 3)

    x, y = [], []
    for i in range(len(data) - input_window - output_window):
        a = data[i: i + input_window + output_window]  # (in+out, ...)
        x.append(a[0: input_window])  # (in, ...)
        y.append(a[input_window: input_window + output_window])  # (out, ...)
    x = np.array(x)  # (num_samples, in, ...)
    y = np.array(y)  # (num_samples, out, ...)

    train_size = int(x.shape[0] * (train_rate + eval_rate))
    trainx = x[:train_size]  # (train_size, in, ...)
    trainy = y[:train_size]  # (train_size, out, ...)
    testx = x[train_size:x.shape[0]]  # (test_size, in, ...)
    testy = y[train_size:x.shape[0]]  # (test_size, out, ...)
    return trainx, trainy, testx, testy


def set_random_seed(seed):
    """
    重置随机数种子

    Args:
        seed(int): 种子数
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
