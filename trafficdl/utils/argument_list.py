"""
store the arguments can be modified by the user
"""
import argparse

general_arguments = {
    "gpu": "bool",
    "batch_size": "int",
    "train_rate": "float",
    "eval_rate": "float",
    "learning_rate": "float",
    "max_epoch": "int",
    "gpu_id": "int"
}


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true'):
        return True
    elif s.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('bool value expected.')


def str2float(s):
    if isinstance(s, float):
        return s
    try:
        x = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError('float value expected.')
    return x
