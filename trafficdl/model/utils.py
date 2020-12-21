import importlib

def get_model(config):
    try:
        return getattr(importlib.import_module('trafficdl.model'), config['model'])(config)
    except AttributeError:
        raise AttributeError('model is not found')
