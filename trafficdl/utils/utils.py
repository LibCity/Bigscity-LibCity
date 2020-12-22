import importlib

def get_executor(config):
    try:
        return getattr(importlib.import_module('trafficdl.executor'), config['executor'])(config)
    except AttributeError:
        raise AttributeError('executor is not found')

def get_model(config):
    try:
        return getattr(importlib.import_module('trafficdl.model'), config['model'])(config)
    except AttributeError:
        raise AttributeError('model is not found')

def get_evaluator(config):
    try:
        return getattr(importlib.import_module('trafficdl.evaluator'), config['evaluator'])(config)
    except AttributeError:
        raise AttributeError('evaluator is not found')
