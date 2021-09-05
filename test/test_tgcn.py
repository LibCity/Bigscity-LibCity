from libcity.data import get_dataset
from libcity.utils import get_executor
from libcity.utils import get_model
from libcity.utils import get_logger
import sys
import time

config = {
    'log_level': 'INFO',

    'dataset': 'SZ_TAXI',
    'model': 'TGCN',
    'evaluator': 'TrafficStateEvaluator',
    'executor': 'TrafficStateExecutor',
    'dataset_class': 'TrafficStatePointDataset',
    'metrics': ['MAE', 'MSE', 'RMSE', 'masked_MAPE', 'R2', 'EVAR'],
    'weight_col': 'link_weight',
    'calculate_weight': False,
    'adj_epsilon': 0.1,
    'add_time_in_day': False,
    'add_day_in_week': False,
    'scaler': "normal",

    'num_workers': 0,
    'cache_dataset': True,
    'gpu': True,
    'gpu_id': '1',
    'batch_size': 32,

    'input_window': 12,
    'output_window': 12,
    'rnn_units': 100,
    'train_rate': 0.8,
    'eval_rate': 0.2,

    'learning_rate': 0.001,
    'learner': 'adam',
    'epoch': 0,
    'max_epoch': 5000,
    'lr_decay': False,
    'lr_decay_ratio': 0.1,
    'max_grad_norm': 5,
    'clip_grad_norm': False,
    'lr_scheduler': 'multisteplr',
    'use_early_stop': False,
    'steps': [20, 30, 40, 50],
    'lambda': 0.0015
}

import os

os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']
import torch

config['device'] = torch.device("cuda" if torch.cuda.is_available() and config['gpu'] else "cpu")

logger = get_logger(config)
# 加载数据集
dataset = get_dataset(config)
# 转换数据，并划分数据集
train_data, valid_data, test_data = dataset.get_data()

print(len(train_data.dataset), train_data.dataset[0][0].shape, train_data.dataset[0][1].shape, train_data.batch_size)
print(len(valid_data.dataset), valid_data.dataset[0][0].shape, valid_data.dataset[0][1].shape, valid_data.batch_size)
print(len(test_data.dataset), test_data.dataset[0][0].shape, test_data.dataset[0][1].shape, test_data.batch_size)

data_feature = dataset.get_data_feature()
print(data_feature['adj_mx'].shape)
print(data_feature['adj_mx'].sum())

model = get_model(config, data_feature)

# 加载执行器
model_cache_file = './libcity/cache/model_cache/' + config['model'] + '_' + config['dataset'] + '.m'
executor = get_executor(config, model)
start_time = time.time()
# 训练
executor.train(train_data, valid_data)
end_time = time.time()
executor.save_model(model_cache_file)
executor.load_model(model_cache_file)
# 评估，评估结果将会放在 cache/evaluate_cache 下
executor.evaluate(valid_data)

print('------------total   ', end_time - start_time, '-------------')
