import importlib
import logging
import datetime
import os
import sys


def get_executor(config, model):
    try:
        return getattr(importlib.import_module('trafficdl.executor'),
                       config['executor'])(config, model)
    except AttributeError:
        raise AttributeError('executor is not found')


def get_model(config, data_feature):
    try:
        return getattr(importlib.import_module('trafficdl.model'),
                       config['model'])(config, data_feature)
    except AttributeError:
        raise AttributeError('model is not found')


def get_evaluator(config):
    try:
        return getattr(importlib.import_module('trafficdl.evaluator'),
                       config['evaluator'])(config)
    except AttributeError:
        raise AttributeError('evaluator is not found')


def get_logger(config, name=None):
    log_dir = './trafficdl/log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = '{}-{}.log'.format(config['model'], get_local_time())
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
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur


def ensure_dir(dir_path):
    r"""Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def trans_naming_rule(origin, origin_rule, target_rule):
    """
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
