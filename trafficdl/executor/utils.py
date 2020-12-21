import importlib

def get_executor(config):
    try:
        return getattr(importlib.import_module('trafficdl.executor'), config['executor'])(config)
    except AttributeError:
        raise AttributeError('executor is not found')
