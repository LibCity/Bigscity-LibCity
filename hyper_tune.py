"""
训练并评估单一模型的脚本
"""

import argparse

from trafficdl.pipeline import hyper_parameter
from trafficdl.utils import hyper_arguments, str2bool, str2float


def add_other_args(parser):
    for arg in hyper_arguments:
        if hyper_arguments[arg]['type'] == 'int':
            parser.add_argument('--{}'.format(arg), type=int, default=hyper_arguments[arg]['default'],
                                help=hyper_arguments[arg]['help'])
        elif hyper_arguments[arg] == 'bool':
            parser.add_argument('--{}'.format(arg),
                                type=str2bool, default=hyper_arguments[arg]['default'],
                                help=hyper_arguments[arg]['help'])
        elif hyper_arguments[arg] == 'float':
            parser.add_argument('--{}'.format(arg),
                                type=str2float, default=hyper_arguments[arg]['default'],
                                help=hyper_arguments[arg]['help'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 增加指定的参数
    parser.add_argument('--task', type=str,
                        default='traj_loc_pred', help='the name of task')
    parser.add_argument('--model', type=str,
                        default='DeepMove', help='the name of model')
    parser.add_argument('--dataset', type=str,
                        default='foursquare_tky', help='the name of dataset')
    parser.add_argument('--config_file', type=str,
                        default=None, help='the file name of config file')
    parser.add_argument('--space_file', type=str,
                        default=None, help='the file which specifies the parameter search space')
    parser.add_argument('--scheduler', type=str,
                        default='FIFO', help='the trial sheduler which will be used in ray.tune.run')
    parser.add_argument('--search_alg', type=str,
                        default='GridSearch', help='the search algorithm')
    # 增加其他可选的参数
    add_other_args(parser)
    # 解析参数
    args = parser.parse_args()
    dict_args = vars(args)
    other_args = {key: val for key, val in dict_args.items() if key not in [
        'task', 'model', 'dataset', 'config_file', 'space_file', 'scheduler', 'search_alg'] and
        val is not None}
    hyper_parameter(task=args.task, model_name=args.model, dataset_name=args.dataset,
                    config_file=args.config_file, space_file=args.space_file,
                    scheduler=args.scheduler, search_alg=args.search_alg, other_args=other_args)
