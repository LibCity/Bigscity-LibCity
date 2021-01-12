import os

from trafficdl.data import get_dataset
from trafficdl.utils import get_executor, get_evaluator, get_model
config = {
    "dataset_class": "TrajectoryDataset",
    "executor": "TrajLocPredExecutor",
    "evaluator": "TrajLocPredEvaluator",
    "metrics": ["topk"],
    "topk": 1,
    "min_session_len": 5,
    "min_sessions": 2,
    "time_window_size": 72,
    "history_len": 50,
    "batch_size": 20,
    "num_workers": 0,
    "cache_dataset": True,
    "train_rate": 0.6,
    "eval_rate": 0.2,
    "gpu": True,
    "learning_rate": 5e-4,
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
    "dropout_p": 0.3,
    'history_type': 'cut_off'
}

config['dataset'] = 'foursquare_tky' # foursquare_tky
config['model'] = 'DeepMove' # DeepMove
# 加载数据集
dataset = get_dataset(config)
# 转换数据，并划分数据集
train_data, valid_data, test_data = dataset.get_data()
# batch = train_data.__iter__().__next__()
# batch.to_tensor(gpu=True)
data_feature = dataset.get_data_feature()
# 加载执行器
model_cache_file = './trafficdl/cache/model_cache/DeepMove_foursquare_tky.m'
model = get_model(config, data_feature)
executor = get_executor(config, model)
# self = executor.model
# 训练
# executor.train(train_data, valid_data)
# executor.save_model(model_cache_file)
executor.load_model(model_cache_file)
# 评估，评估结果将会放在 cache/evaluate_cache 下
executor.evaluate(test_data)

# loc = batch['current_loc']
# tim = batch['current_tim']
# history_loc = batch['history_loc']
# history_tim = batch['history_tim']
# loc_len = batch.get_origin_len('current_loc')
# history_len = batch.get_origin_len('history_loc')
# batch_size = loc.shape[0]
