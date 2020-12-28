import os

from trafficdl.config import ConfigParser
from trafficdl.data import get_dataset
from trafficdl.utils import get_executor, get_model

def run_model(task=None, model=None, dataset=None, config_file=None, save_model=True, train=True, other_args=None):
    '''
    Args:
        task (str): task name
        model (str): model name
        dataset (str): dataset name
        config_file (str): config filename used to modify the pipeline's settings. the config file should be json.
        save_model (bool): whether to save the model
        train (bool): whether to train the model
        other_args (dict): the rest parameter args, which will be pass to the Config 
    '''

    # load config
    config = ConfigParser(task, model, dataset, config_file, other_args)
    # 加载数据集
    dataset = get_dataset(config)
    # 转换数据，并划分数据集
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    # 加载执行器
    model_cache_file = './trafficdl/cache/model_cache/{}_{}.m'.format(model, dataset)
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
