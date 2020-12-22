import os

from trafficdl.data import get_dataset
from trafficdl.utils import get_executor
config = {
    "dataset_class": "TrajectoryDataset",
    "executor": "TrajLocPredExecutor",
    "evaluator": "TrajLocPredEvaluator",
    "min_session_len": 5,
    "min_sessions": 2,
    "time_length": 72,
    "history_len": 50,
    "batch_size": 1,
    "num_workers": 0,
    "cache_dataset": True,
    "train_rate": 0.6,
    "eval_rate": 0.2,
    "use_cuda": True,
    "lr": 5e-4,
    "L2": 1e-5, 
    "max_epoch": 1,
    "lr_step": 2,
    "lr_decay": 0.1,
    "clip": 5.0,
    "schedule_threshold": 1e-3,
    "verbose": 10,
    "loc_emb_size": 500,
    "uid_emb_size": 40,
    "tim_emb_size": 10,
    "hidden_size": 500,
    "attn_type": "dot",
    "rnn_type": "LSTM",
    "dropout_p": 0.3
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
    config['dataset'] = 'foursquare_tky' # foursquare_tky
    config['model'] = 'DeepMove' # DeepMove

    # 加载数据集
    dataset = get_dataset(config)
    # 转换数据，并划分数据集
    train_data, valid_data, test_data = dataset.get_data()
    config['data_feature'] = dataset.get_data_feature()
    # 加载执行器
    model_cache_file = './trafficdl/cache/model_cache/{}_{}.m'.format(model, dataset)
    executor = get_executor(config)
    # 训练
    if train or not os.path.exists(model_cache_file):
        executor.train(train_data, valid_data)
        if saved:
            executor.save_model(model_cache_file)
    else:
        executor.load_model(model_cache_file)
    # 评估，评估结果将会放在 cache/evaluate_cache 下
    executor.evaluate(test_data)
