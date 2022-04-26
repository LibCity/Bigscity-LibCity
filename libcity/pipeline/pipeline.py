import os
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.schedulers import FIFOScheduler, ASHAScheduler, MedianStoppingRule
from ray.tune.suggest import ConcurrencyLimiter
import json
import torch
import random
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_executor, get_model, get_logger, ensure_dir, set_random_seed


def run_model(task=None, model_name=None, dataset_name=None, config_file=None,
              saved_model=True, train=True, other_args=None):
    """
    Args:
        task(str): task name
        model_name(str): model name
        dataset_name(str): dataset name
        config_file(str): config filename used to modify the pipeline's
            settings. the config file should be json.
        saved_model(bool): whether to save the model
        train(bool): whether to train the model
        other_args(dict): the rest parameter args, which will be pass to the Config
    """
    # load config
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, saved_model, train, other_args)
    exp_id = config.get('exp_id', None)
    if exp_id is None:
        # Make a new experiment ID
        exp_id = int(random.SystemRandom().random() * 100000)
        config['exp_id'] = exp_id
    # logger
    logger = get_logger(config)
    logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(task), str(model_name), str(dataset_name), str(exp_id)))
    logger.info(config.config)
    # seed
    seed = config.get('seed', 0)
    set_random_seed(seed)
    # 加载数据集
    dataset = get_dataset(config)
    # 转换数据，并划分数据集
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    # 加载执行器
    model_cache_file = './libcity/cache/{}/model_cache/{}_{}.m'.format(
        exp_id, model_name, dataset_name)
    model = get_model(config, data_feature)
    executor = get_executor(config, model, data_feature)
    # 训练
    if train or not os.path.exists(model_cache_file):
        executor.train(train_data, valid_data)
        if saved_model:
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
                    scheduler=None, search_alg=None, other_args=None, num_samples=5, max_concurrent=1,
                    cpu_per_trial=1, gpu_per_trial=1):
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
    experiment_config = ConfigParser(task, model_name, dataset_name, config_file=config_file,
                                     other_args=other_args)
    # exp_id
    exp_id = experiment_config.get('exp_id', None)
    if exp_id is None:
        exp_id = int(random.SystemRandom().random() * 100000)
        experiment_config['exp_id'] = exp_id
    # logger
    logger = get_logger(experiment_config)
    logger.info('Begin ray-tune, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(task), str(model_name), str(dataset_name), str(exp_id)))
    logger.info(experiment_config.config)
    # check space_file
    if space_file is None:
        logger.error('the space_file should not be None when hyperparameter tune.')
        exit(0)
    # seed
    seed = experiment_config.get('seed', 0)
    set_random_seed(seed)
    # parse space_file
    search_sapce = parse_search_space(space_file)
    # load dataset
    dataset = get_dataset(experiment_config)
    # get train valid test data
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()

    def train(config, checkpoint_dir=None, experiment_config=None,
              train_data=None, valid_data=None, data_feature=None):
        """trainable function which meets ray tune API

        Args:
            config (dict): A dict of hyperparameter.
        """
        # modify experiment_config
        for key in config:
            if key in experiment_config:
                experiment_config[key] = config[key]
        experiment_config['hyper_tune'] = True
        logger = get_logger(experiment_config)
        # exp_id
        exp_id = int(random.SystemRandom().random() * 100000)
        experiment_config['exp_id'] = exp_id
        logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                    format(str(task), str(model_name), str(dataset_name), str(exp_id)))
        logger.info('running parameters: ' + str(config))
        # load model
        model = get_model(experiment_config, data_feature)
        # load executor
        executor = get_executor(experiment_config, model, data_feature)
        # checkpoint by ray tune
        if checkpoint_dir:
            checkpoint = os.path.join(checkpoint_dir, 'checkpoint')
            executor.load_model(checkpoint)
        # train
        executor.train(train_data, valid_data)

    # init search algorithm and scheduler
    if search_alg == 'BasicSearch':
        algorithm = BasicVariantGenerator()
    elif search_alg == 'BayesOptSearch':
        algorithm = BayesOptSearch(metric='loss', mode='min')
        # add concurrency limit
        algorithm = ConcurrencyLimiter(algorithm, max_concurrent=max_concurrent)
    elif search_alg == 'HyperOpt':
        algorithm = HyperOptSearch(metric='loss', mode='min')
        # add concurrency limit
        algorithm = ConcurrencyLimiter(algorithm, max_concurrent=max_concurrent)
    else:
        raise ValueError('the search_alg is illegal.')
    if scheduler == 'FIFO':
        tune_scheduler = FIFOScheduler()
    elif scheduler == 'ASHA':
        tune_scheduler = ASHAScheduler()
    elif scheduler == 'MedianStoppingRule':
        tune_scheduler = MedianStoppingRule()
    else:
        raise ValueError('the scheduler is illegal')
    # ray tune run
    ensure_dir('./libcity/cache/hyper_tune')
    result = tune.run(tune.with_parameters(train, experiment_config=experiment_config, train_data=train_data,
                      valid_data=valid_data, data_feature=data_feature),
                      resources_per_trial={'cpu': cpu_per_trial, 'gpu': gpu_per_trial}, config=search_sapce,
                      metric='loss', mode='min', scheduler=tune_scheduler, search_alg=algorithm,
                      local_dir='./libcity/cache/hyper_tune', num_samples=num_samples)
    best_trial = result.get_best_trial("loss", "min", "last")
    logger.info("Best trial config: {}".format(best_trial.config))
    logger.info("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    # save best
    best_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
    model_state, optimizer_state = torch.load(best_path)
    model_cache_file = './libcity/cache/{}/model_cache/{}_{}.m'.format(
        exp_id, model_name, dataset_name)
    ensure_dir('./libcity/cache/{}/model_cache'.format(exp_id))
    torch.save((model_state, optimizer_state), model_cache_file)


def objective_function(task=None, model_name=None, dataset_name=None, config_file=None,
                       saved_model=True, train=True, other_args=None, hyper_config_dict=None):
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, saved_model, train, other_args, hyper_config_dict)
    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()

    model = get_model(config, data_feature)
    executor = get_executor(config, model, data_feature)
    best_valid_score = executor.train(train_data, valid_data)
    test_result = executor.evaluate(test_data)

    return {
        'best_valid_score': best_valid_score,
        'test_result': test_result
    }
