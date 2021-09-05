from libcity.data import get_dataset
from libcity.utils import get_executor
from libcity.utils import get_model
from libcity.utils import get_logger

config = {
    'log_level': 'INFO',

    'dataset': 'METR_LA',
    'model': 'DCRNN',
    'evaluator': 'TrafficStateEvaluator',
    'executor': 'DCRNNExecutor',
    'dataset_class': 'TrafficStatePointDataset',
    'metrics': ['masked_MAE', 'masked_MSE', 'masked_RMSE', 'masked_MAPE', 'R2', 'EVAR'],
    'weight_col': 'cost',
    'data_col': ['traffic_speed'],
    'calculate_weight': True,
    'adj_epsilon': 0.1,
    'add_time_in_day': True,
    'add_day_in_week': False,
    'scaler': "standard",

    'num_workers': 1,
    'cache_dataset': True,
    'gpu': True,
    'gpu_id': '1',
    'batch_size': 64,

    'cl_decay_steps': 2000,
    'filter_type': 'dual_random_walk',
    'input_window': 12,
    'output_window': 12,
    'max_diffusion_step': 2,
    'num_rnn_layers': 2,
    'rnn_units': 64,
    'use_curriculum_learning': True,
    'train_rate': 0.7,
    'eval_rate': 0.1,

    'learning_rate': 0.01,
    'learner': 'adam',
    'weight_decay': 0,
    'epoch': 0,
    'max_epoch': 100,
    'lr_epsilon': 1.0e-3,
    'lr_decay': True,
    'lr_decay_ratio': 0.1,
    'max_grad_norm': 5,
    'clip_grad_norm': True,
    'lr_scheduler': 'multisteplr',
    'use_early_stop': True,
    'patience': 50,
    'steps': [20, 30, 40, 50],
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

# 训练
executor.train(train_data, valid_data)
executor.save_model(model_cache_file)
executor.load_model(model_cache_file)
# 评估，评估结果将会放在 cache/evaluate_cache 下
executor.evaluate(test_data)
