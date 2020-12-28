from trafficdl.data import get_dataset
from trafficdl.utils import get_executor
from trafficdl.utils import get_model
from trafficdl.utils import get_logger

config = {
    'log_level': 'INFO',

    'dataset': 'METR_LA',
    'model': 'DCRNN',
    'evaluator': 'TrafficSpeedPredEvaluator',
    'executor': 'TrafficSpeedPredExecutor',
    'dataset_class': 'TrafficSpeedDataset',
    'metrics': ['MAE', 'MSE', 'RESE', 'MAPE'],

    'num_workers': 1,
    'cache_dataset': True,
    'gpu': True,
    'batch_size': 64,
    'test_batch_size': 64,
    'val_batch_size': 64,

    'cl_decay_steps': 2000,
    'filter_type': 'dual_random_walk',
    'seq_len': 12,
    'horizon': 12,
    'input_dim': 2,
    'l1_decay': 0,
    'max_diffusion_step': 2,
    'num_nodes': 207,
    'num_rnn_layers': 2,
    'output_dim': 1,
    'rnn_units': 64,
    'use_curriculum_learning': True,
    'split_ratio': [0.7, 0.1, 0.2],


    'base_lr': 0.01,
    'dropout': 0,
    'epoch': 0,
    'epochs': 100,
    'epsilon': 1.0e-3,
    'global_step': 0,
    'lr_decay_ratio': 0.1,
    'max_grad_norm': 5,
    'clip_grad_norm': True,
    'max_to_keep': 100,
    'min_learning_rate': 2.0e-06,
    'optimizer': 'adam',
    'patience': 50,
    'steps': [20, 30, 40, 50],
}

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
model_cache_file = './trafficdl/cache/model_cache/DCRNN_METR_LA.m'
if config['model'] == 'DCRNN':
    executor = get_executor(config, model)

# 训练
executor.train(train_data, valid_data)
executor.save_model(model_cache_file)
executor.load_model(model_cache_file)
# 评估，评估结果将会放在 cache/evaluate_cache 下
executor.evaluate(test_data)
