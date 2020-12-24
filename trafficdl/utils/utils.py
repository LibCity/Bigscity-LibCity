import importlib
import logging
import datetime
import os
import sys

def get_executor(config, model):
    try:
        return getattr(importlib.import_module('trafficdl.executor'), config['executor'])(config, model)
    except AttributeError:
        raise AttributeError('executor is not found')

def get_model(config, data_feature):
    try:
        return getattr(importlib.import_module('trafficdl.model'), config['model'])(config, data_feature)
    except AttributeError:
        raise AttributeError('model is not found')

def get_evaluator(config):
    try:
        return getattr(importlib.import_module('trafficdl.evaluator'), config['evaluator'])(config)
    except AttributeError:
        raise AttributeError('evaluator is not found')

def get_logger(config, name=None):
    log_dir = './trafficdl/log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = '{}-{}.log'.format(config['model'], get_local_time())
    logfilepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)

    if config['log_level'] is None or config['log_level'].lower() == 'info':
        level = logging.INFO
    elif config['log_level'].lower() == 'debug':
        level = logging.DEBUG
    elif config['log_level'].lower() == 'error':
        level = logging.ERROR
    elif config['log_level'].lower() == 'warning':
        level = logging.WARNING
    elif config['log_level'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(logfilepath)
    file_handler.setFormatter(formatter)

    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
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


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def ensure_dir(dir_path):
    r"""Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
