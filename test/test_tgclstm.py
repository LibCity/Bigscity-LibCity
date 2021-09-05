from libcity.data import get_dataset
from libcity.utils import get_executor
from libcity.utils import get_model
from libcity.utils import get_logger

config = {
    'log_level': 'INFO',

    'dataset': 'Loop_Seattle',
    'model': 'TGCLSTM',
    'evaluator': 'TrafficStateEvaluator',
    'executor': 'TrafficStateExecutor',
    'dataset_class': 'TGCLSTMDataset',
    'metrics': ['masked_MAE', 'masked_MSE', 'masked_RMSE', 'masked_MAPE', 'R2', 'EVAR'],
    'weight_col': 'adj',
    'calculate_weight': False,
    'add_time_in_day': False,
    'add_day_in_week': False,
    'scaler': 'normal',

    'num_workers': 1,
    'cache_dataset': True,
    'gpu': True,
    'gpu_id': '1',
    'batch_size': 64,

    'K_hop_numbers': 3,
    'back_length': 3,
    'Clamp_A': False,
    'input_window': 12,
    'output_window': 12,
    'train_rate': 0.7,
    'eval_rate': 0.1,

    'learning_rate': 1e-5,
    'learner': 'RMSProp',
    'lr_decay': False,
    'lr_decay_ratio': 0.5,
    'step_size': 25,
    'lr_scheduler': 'steplr',
    'max_epoch': 100,
    'use_early_stop': False,
    'patience': 50,
}

import os
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']
import torch
config['device'] = torch.device("cuda" if torch.cuda.is_available() and config['gpu'] else "cpu")

logger = get_logger(config)

dataset = get_dataset(config)

train_data, valid_data, test_data = dataset.get_data()
print(len(train_data.dataset), train_data.dataset[0][0].shape, train_data.dataset[0][1].shape, train_data.batch_size)
print(len(valid_data.dataset), valid_data.dataset[0][0].shape, valid_data.dataset[0][1].shape, valid_data.batch_size)
print(len(test_data.dataset), test_data.dataset[0][0].shape, test_data.dataset[0][1].shape, test_data.batch_size)

data_feature = dataset.get_data_feature()
print(data_feature['adj_mx'].shape)

model = get_model(config, data_feature)

model_cache_file = './libcity/cache/model_cache/' + config['model'] + '_' + config['dataset'] + '.m'
executor = get_executor(config, model)

executor.train(train_data, valid_data)
executor.save_model(model_cache_file)

executor.load_model(model_cache_file)
executor.evaluate(test_data)
