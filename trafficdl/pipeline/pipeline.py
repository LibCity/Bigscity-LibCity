from trafficdl.data import get_dataset

config = {
    'dataset_class': 'TrajectoryDataset',
    "min_session_len": 5,
    "min_sessions": 2,
    "time_length": 72,
    "history_len": 50,
    "batch_size": 20,
    "num_workers": 0,
    "train_rate": 0.6,
    "eval_rate": 0.2
}

def run_model(task=None, model=None, dataset=None, config_file=None, saved=True, train=True, other_args=None):
    '''
    Args:
        task (str): task name
        model (str): model name
        dataset (str): dataset name
        config_file (str): config filename used to modify the pipeline's settings. the config file should be json.
        saved (bool): whether to save the model
        train (bool): whether to train the model
        other_args (dict): the rest parameter args, which will be pass to the Config 
    '''

    # load config
    # TODO: 先不实现 config
    config['dataset'] = dataset
    config['model'] = model

    # 加载数据集
    dataset = get_dataset(config)
    # 转换数据，并划分数据集
    train_data, valid_data, test_data = dataset.get_data()
    
