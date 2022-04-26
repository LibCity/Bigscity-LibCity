"""
模型调参脚本 (based on the hyperopt)
"""

import argparse
import random
from libcity.pipeline import objective_function
from libcity.executor import HyperTuning
from libcity.utils import str2bool, get_logger, set_random_seed, add_general_args


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
    parser.add_argument('--saved_model', type=str2bool,
                        default=True, help='whether save the trained model')
    parser.add_argument('--train', type=str2bool, default=True,
                        help='whether re-train model if the model is \
                             trained before')
    parser.add_argument('--params_file', type=str,
                        default='hyper_example.txt', help='the file which specify the \
                             hyper-parameters and ranges to be adjusted')
    parser.add_argument('--hyper_algo', type=str,
                        default='grid_search', help='hyper-parameters search algorithm')
    parser.add_argument('--max_evals', type=int,
                        default=100, help='Allow up to this many function \
                             evaluations before returning.')
    parser.add_argument('--exp_id', type=str, default=None, help='id of experiment')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # 增加其他可选的参数
    add_general_args(parser)
    # 解析参数
    args = parser.parse_args()
    dict_args = vars(args)
    other_args = {key: val for key, val in dict_args.items() if key not in [
        'task', 'model', 'dataset', 'config_file', 'saved_model', 'train',
        'params_file', 'hyper_algo'] and val is not None}
    # exp_id
    exp_id = dict_args.get('exp_id', None)
    if exp_id is None:
        # Make a new experiment ID
        exp_id = int(random.SystemRandom().random() * 100000)
        other_args['exp_id'] = exp_id
    # logger
    logger = get_logger({'model': args.model, 'dataset': args.dataset, 'exp_id': exp_id})
    # seed
    seed = dict_args.get('seed', 0)
    set_random_seed(seed)
    other_args['seed'] = seed
    hp = HyperTuning(objective_function, params_file=args.params_file, algo=args.hyper_algo,
                     max_evals=args.max_evals, task=args.task, model_name=args.model,
                     dataset_name=args.dataset, config_file=args.config_file,
                     saved_model=args.saved_model, train=args.train, other_args=other_args)
    hp.start()
    hp.save_result(filename='hyper.result')
    logger.info('best params: ' + str(hp.best_params))
    logger.info('best result: ')
    logger.info(str(hp.params2result[hp.params2str(hp.best_params)]))
