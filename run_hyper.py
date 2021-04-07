"""
单一模型调参脚本
"""

import argparse

from trafficdl.pipeline import objective_function
from trafficdl.executor import HyperTuning
from trafficdl.utils import general_arguments, str2bool, str2float, get_logger


def add_other_args(parser):
    for arg in general_arguments:
        if general_arguments[arg] == 'int':
            parser.add_argument('--{}'.format(arg), type=int, default=None)
        elif general_arguments[arg] == 'bool':
            parser.add_argument('--{}'.format(arg),
                                type=str2bool, default=None)
        elif general_arguments[arg] == 'float':
            parser.add_argument('--{}'.format(arg),
                                type=str2float, default=None)


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
    parser.add_argument('--saved_model', type=str2bool,
                        default=True, help='whether save the trained model')
    parser.add_argument('--train', type=str2bool, default=True,
                        help='whether re-train model if the model is \
                             trained before')
    parser.add_argument('--params_file', type=str,
                        default=None, help='the file which specify the \
                             hyper-parameters and ranges to be adjusted')
    parser.add_argument('--hyper_algo', type=str,
                        default='grid_search', help='hyper-parameters search algorithm')
    parser.add_argument('--max_evals', type=int,
                        default=100, help='Allow up to this many function \
                             evaluations before returning.')
    # 增加其他可选的参数
    add_other_args(parser)
    # 解析参数
    args = parser.parse_args()
    dict_args = vars(args)
    other_args = {key: val for key, val in dict_args.items() if key not in [
        'task', 'model', 'dataset', 'config_file', 'saved_model', 'train',
        'params_file', 'hyper_algo'] and val is not None}

    logger = get_logger({'model': args.model})
    hp = HyperTuning(objective_function, params_file=args.params_file, algo=args.hyper_algo,
                     max_evals=args.max_evals, task=args.task, model_name=args.model,
                     dataset_name=args.dataset, config_file=args.config_file,
                     saved_model=args.saved_model, train=args.train, other_args=other_args)
    hp.run()
    hp.export_result(output_file='hyper_example.result')
    logger.info('best params: ' + str(hp.best_params))
    logger.info('best result: ')
    logger.info(str(hp.params2result[hp.params2str(hp.best_params)]))
