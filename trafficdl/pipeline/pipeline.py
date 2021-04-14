import os
from ray import tune
import json
import torch

from trafficdl.config import ConfigParser
from trafficdl.data import get_dataset
from trafficdl.utils import get_executor, get_model, get_logger


def run_model(task=None, model_name=None, dataset_name=None, config_file=None,
              save_model=True, train=True, other_args=None):
    """
    Args:
        task(str): task name
        model_name(str): model name
        dataset_name(str): dataset name
        config_file(str): config filename used to modify the pipeline's
            settings. the config file should be json.
        save_model(bool): whether to save the model
        train(bool): whether to train the model
        other_args(dict): the rest parameter args, which will be pass to the Config
    """
    # load config
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, other_args)
    # logger
    logger = get_logger(config)
    logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}'.
                format(str(task), str(model_name), str(dataset_name)))
    # 加载数据集
    dataset = get_dataset(config)
    # 转换数据，并划分数据集
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    # 加载执行器
    model_cache_file = './trafficdl/cache/model_cache/{}_{}.m'.format(
        model_name, dataset_name)
    model = get_model(config, data_feature)
    executor = get_executor(config, model)
    # 训练
    if train or not os.path.exists(model_cache_file):
        executor.train(train_data, valid_data)
        if save_model:
            executor.save_model(model_cache_file)
    else:
        executor.load_model(model_cache_file)
    # 评估，评估结果将会放在 cache/evaluate_cache 下
    executor.evaluate(test_data)


def parse_search_space(space_file):
    search_space = {}
    if os.path.exists('./{}.json'.format(space_file)):
        with open('./{}.json'.format(space_file), 'r') as f:
            paras_dict = json.load(f)
            for name in paras_dict:
                paras_type = paras_dict[name]['type']
                if paras_type == 'uniform':
                    # name type low up
                    try:
                        search_space[name] = tune.uniform(paras_dict[name]['lower'], paras_dict[name]['upper'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing uniform type.')
                elif paras_type == 'randn':
                    # name type mean sd
                    try:
                        search_space[name] = tune.randn(paras_dict[name]['mean'], paras_dict[name]['sd'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing randn type.')
                elif paras_type == 'randint':
                    # name type lower upper
                    try:
                        if 'lower' not in paras_dict[name]:
                            search_space[name] = tune.randint(paras_dict[name]['upper'])
                        else:
                            search_space[name] = tune.randint(paras_dict[name]['lower'], paras_dict[name]['upper'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing randint type.')
                elif paras_type == 'choice':
                    # name type list
                    try:
                        search_space[name] = tune.choice(paras_dict[name]['list'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing choice type.')
                elif paras_type == 'grid_search':
                    # name type list
                    try:
                        search_space[name] = tune.grid_search(paras_dict[name]['list'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing grid_search type.')
                else:
                    raise TypeError('The space file does not meet the format requirements,\
                            when parsing an undefined type.')
    else:
        raise FileNotFoundError('The space file {}.json is not found. Please ensure \
            the config file is in the root dir and is a txt.'.format(space_file))
    return search_space


def hyper_parameter(task=None, model_name=None, dataset_name=None, config_file=None, space_file=None,
                    scheduler=None, search_alg=None, other_args=None):
    """ Use Ray tune to hyper parameter tune

    Args:
        task(str): task name
        model_name(str): model name
        dataset_name(str): dataset name
        config_file(str): config filename used to modify the pipeline's
            settings. the config file should be json.
        space_file(str): the file which specifies the parameter search space
        scheduler(str): the trial sheduler which will be used in ray.tune.run
        search_alg(str): the search algorithm
        other_args(dict): the rest parameter args, which will be pass to the Config
    """
    # load config
    experiment_config = ConfigParser(task, model_name, dataset_name, config_file, other_args)
    # logger
    logger = get_logger(experiment_config)
    # check space_file
    if space_file is None:
        logger.error('the space_file should not be None when hyperparameter tune.')
        exit(0)
    # parse space_file
    search_sapce = parse_search_space(space_file)

    def train(config, checkpoint_dir=None):
        """trainable function which meets ray tune API

        Args:
            config (dict): A dict of hyperparameter.
        """
        # modify experiment_config
        for key in config:
            if key in experiment_config:
                experiment_config[key] = config[key]
        # load dataset
        dataset = get_dataset(experiment_config)
        # get train valid test data
        train_data, valid_data, test_data = dataset.get_data()
        data_feature = dataset.get_data_feature()
        # load model
        model = get_model(experiment_config, data_feature)
        # load executor
        executor = get_executor(experiment_config, model)
        # checkpoint by ray tune
        if checkpoint_dir:
            checkpoint = os.path.join(checkpoint_dir, 'checkpoint')
            executor.load_model(checkpoint)
        # train
        experiment_config['hyper_tune'] = True
        experiment_config['load_best_epoch'] = False
        executor.train(train_data, valid_data)

    # init search algorithm and scheduler
    if search_alg == 'BasicSearch':
        algorithm = tune.suggest.basic_variant.BasicVariantGenerator()
    elif search_alg == 'BayesOptSearch':
        algorithm = tune.suggest.bayesopt.BayesOptSearch(metric='loss', mode='min')
    elif search_alg == 'HyperOpt':
        algorithm = tune.suggest.hyperopt.HyperOptSearch(metric='loss', mode='min')
    else:
        raise ValueError('the search_alg is illegal.')
    if scheduler == 'FIFO':
        tune_scheduler = tune.schedulers.FIFOScheduler()
    elif scheduler == 'ASHA':
        tune_scheduler = tune.schedulers.ASHAScheduler()
    elif scheduler == 'MedianStoppingRule':
        tune_scheduler = tune.schedulers.MedianStoppingRule()
    else:
        raise ValueError('the scheduler is illegal')
    # ray tune run
    result = tune.run(train, resources_per_trial={'cpu': 1, 'gpu': 1}, config=search_sapce,
                      metric='loss', mode='min', scheduler=tune_scheduler, search_alg=algorithm,
                      local_dir='./trafficdl/tmp')
    best_trial = result.get_best_trial("loss", "min", "last")
    logger.info("Best trial config: {}".format(best_trial.config))
    logger.info("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    # save best
    best_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
    model_state, optimizer_state = torch.load(best_path)
    model_cache_file = './trafficdl/cache/model_cache/{}_{}.m'.format(
        model_name, dataset_name)
    if not os.path.exists('./trafficdl/cache/model_cache'):
        os.makedirs('./trafficdl/cache/model_cache')
    torch.save((model_state, optimizer_state), model_cache_file)
