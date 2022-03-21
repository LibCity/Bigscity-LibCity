"""
store the arguments can be modified by the user
"""
import argparse

general_arguments = {
    "gpu": {
        "type": "bool",
        "default": None,
        "help": "whether use gpu"
    },
    "gpu_id": {
        "type": "int",
        "default": None,
        "help": "the gpu id to use"
    },
    "train_rate": {
        "type": "float",
        "default": None,
        "help": "the train set rate"
    },
    "eval_rate": {
        "type": "float",
        "default": None,
        "help": "the validation set rate"
    },
    "batch_size": {
        "type": "int",
        "default": None,
        "help": "the batch size"
    },
    "learning_rate": {
        "type": "float",
        "default": None,
        "help": "learning rate"
    },
    "max_epoch": {
        "type": "int",
        "default": None,
        "help": "the maximum epoch"
    },
    "dataset_class": {
        "type": "str",
        "default": None,
        "help": "the dataset class name"
    },
    "executor": {
        "type": "str",
        "default": None,
        "help": "the executor class name"
    },
    "evaluator": {
        "type": "str",
        "default": None,
        "help": "the evaluator class name"
    },
}

hyper_arguments = {
    "gpu": {
        "type": "bool",
        "default": None,
        "help": "whether use gpu"
    },
    "gpu_id": {
        "type": "int",
        "default": None,
        "help": "the gpu id to use"
    },
    "train_rate": {
        "type": "float",
        "default": None,
        "help": "the train set rate"
    },
    "eval_rate": {
        "type": "float",
        "default": None,
        "help": "the validation set rate"
    },
    "batch_size": {
        "type": "int",
        "default": None,
        "help": "the batch size"
    }
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


def add_general_args(parser):
    for arg in general_arguments:
        if general_arguments[arg]['type'] == 'int':
            parser.add_argument('--{}'.format(arg), type=int,
                                default=general_arguments[arg]['default'], help=general_arguments[arg]['help'])
        elif general_arguments[arg]['type'] == 'bool':
            parser.add_argument('--{}'.format(arg), type=str2bool,
                                default=general_arguments[arg]['default'], help=general_arguments[arg]['help'])
        elif general_arguments[arg]['type'] == 'float':
            parser.add_argument('--{}'.format(arg), type=str2float,
                                default=general_arguments[arg]['default'], help=general_arguments[arg]['help'])
        elif general_arguments[arg]['type'] == 'str':
            parser.add_argument('--{}'.format(arg), type=str,
                                default=general_arguments[arg]['default'], help=general_arguments[arg]['help'])
        elif general_arguments[arg]['type'] == 'list of int':
            parser.add_argument('--{}'.format(arg), nargs='+', type=int,
                                default=general_arguments[arg]['default'], help=general_arguments[arg]['help'])


def add_hyper_args(parser):
    for arg in hyper_arguments:
        if hyper_arguments[arg]['type'] == 'int':
            parser.add_argument('--{}'.format(arg), type=int,
                                default=hyper_arguments[arg]['default'], help=hyper_arguments[arg]['help'])
        elif hyper_arguments[arg]['type'] == 'bool':
            parser.add_argument('--{}'.format(arg), type=str2bool,
                                default=hyper_arguments[arg]['default'], help=hyper_arguments[arg]['help'])
        elif hyper_arguments[arg]['type'] == 'float':
            parser.add_argument('--{}'.format(arg), type=str2float,
                                default=hyper_arguments[arg]['default'], help=hyper_arguments[arg]['help'])
        elif hyper_arguments[arg]['type'] == 'str':
            parser.add_argument('--{}'.format(arg), type=str,
                                default=hyper_arguments[arg]['default'], help=hyper_arguments[arg]['help'])
        elif hyper_arguments[arg]['type'] == 'list of int':
            parser.add_argument('--{}'.format(arg), nargs='+', type=int,
                                default=hyper_arguments[arg]['default'], help=hyper_arguments[arg]['help'])

