"""
训练并评估单一模型的脚本
"""

import argparse

from libcity.pipeline import hyper_parameter
from libcity.utils import hyper_arguments, str2bool, str2float


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
                        default='traffic_state_pred', help='the name of task')
    parser.add_argument('--model', type=str,
                        default='GRU', help='the name of model')
    parser.add_argument('--dataset', type=str,
                        default='METR_LA', help='the name of dataset')
    parser.add_argument('--config_file', type=str,
                        default=None, help='the file name of config file')
    parser.add_argument('--space_file', type=str,
                        default=None, help='the file which specifies the parameter search space')
    parser.add_argument('--scheduler', type=str,
                        default='FIFO', help='the trial sheduler which will be used in ray.tune.run')
    parser.add_argument('--search_alg', type=str,
                        default='BasicSearch', help='the search algorithm')
    parser.add_argument('--num_samples', type=int,
                        default=5, help='the number of times to sample from hyperparameter space.')
    parser.add_argument('--max_concurrent', type=int,
                        default=1, help='maximum number of trails running at the same time')
    parser.add_argument('--cpu_per_trial', type=int,
                        default=1, help='the number of cpu which per trial will allocate')
    parser.add_argument('--gpu_per_trial', type=int,
                        default=1, help='the number of gpu which per trial will allocate')
    # 增加其他可选的参数
    add_other_args(parser)
    # 解析参数
    args = parser.parse_args()
    dict_args = vars(args)
    other_args = {key: val for key, val in dict_args.items() if key not in [
        'task', 'model', 'dataset', 'config_file', 'space_file', 'scheduler', 'search_alg',
        'num_samples', 'max_concurrent', 'cpu_per_trial', 'gpu_per_trial'] and
        val is not None}
    hyper_parameter(task=args.task, model_name=args.model, dataset_name=args.dataset,
                    config_file=args.config_file, space_file=args.space_file,
                    scheduler=args.scheduler, search_alg=args.search_alg,
                    num_samples=args.num_samples, max_concurrent=args.max_concurrent,
                    cpu_per_trial=args.cpu_per_trial, gpu_per_trial=args.gpu_per_trial,
                    other_args=other_args)
