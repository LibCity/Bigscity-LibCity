from libcity.data import get_dataset
from libcity.utils import get_executor
from libcity.utils import get_model
from libcity.utils import get_logger

config = {
    'log_level': 'INFO',

    'dataset': 'PEMSD4',
    'model': 'ASTGCN',
    'dataset_class': 'ASTGCNDataset',
    'evaluator': 'TrafficStateEvaluator',
    'executor': 'TrafficStateExecutor',
    'metrics': ['MAE', 'MAPE', 'MSE', 'RMSE', 'masked_MAE', 'masked_MSE', 'masked_RMSE', 'masked_MAPE', 'R2', 'EVAR'],
    'weight_col': 'cost',
    'calculate_weight': False,
    'add_time_in_day': False,
    'add_day_in_week': False,
    'pad_with_last_sample': True,
    'scaler': 'standard',

    'num_workers': 1,
    'cache_dataset': True,
    'gpu': True,
    'batch_size': 64,
    'train_rate': 0.6,
    'eval_rate': 0.2,
    'len_trend': 0,
    'len_period': 0,
    'len_closeness': 2,
    'output_window': 12,
    'output_dim': 1,

    'nb_block': 2,
    'K': 3,
    'nb_chev_filter': 64,
    'nb_time_filter': 64,

    'learning_rate': 0.0001,
    'learner': 'adam',
    'weight_decay': 0,
    'epoch': 0,
    'epochs': 100,
    'clip_grad_norm': False,
    'max_grad_norm': 5,
    'lr_decay': False,
    'patience': 50,

    "info": {
        "data_col": [
          "traffic_flow"
        ],
        "weight_col": "cost",
        "data_files": [
          "PEMSD4"
        ],
        "geo_file": "PEMSD4",
        "rel_file": "PEMSD4",
        "output_dim": 1,
        "init_weight_inf_or_zero": "zero",
        "set_weight_link_or_dist": "link",
        "calculate_weight_adj": False,
        "weight_adj_epsilon": 0.1
      }
}

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
config['device'] = torch.device("cuda" if torch.cuda.is_available() and config['gpu'] else "cpu")

logger = get_logger(config)
# 加载数据集
dataset = get_dataset(config)
# 转换数据，并划分数据集
train_data, valid_data, test_data = dataset.get_data()
print(len(train_data), len(train_data.dataset), train_data.dataset[0][0].shape, train_data.dataset[0][1].shape, train_data.batch_size)
print(len(valid_data), len(valid_data.dataset), valid_data.dataset[0][0].shape, valid_data.dataset[0][1].shape, valid_data.batch_size)
print(len(test_data), len(test_data.dataset), test_data.dataset[0][0].shape, test_data.dataset[0][1].shape, test_data.batch_size)

data_feature = dataset.get_data_feature()
print(data_feature['adj_mx'].shape)
print(data_feature['adj_mx'].sum())

model = get_model(config, data_feature)

# 加载执行器
model_cache_file = './libcity/cache/model_cache/' + config['model'] + '_' + config['dataset'] + '.m'
executor = get_executor(config, model)

# 训练
executor.train(train_data, valid_data)
executor.save_model(model_cache_file)
executor.load_model(model_cache_file)
# 评估，评估结果将会放在 cache/evaluate_cache 下
executor.evaluate(test_data)
