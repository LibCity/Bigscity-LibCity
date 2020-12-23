import importlib

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
