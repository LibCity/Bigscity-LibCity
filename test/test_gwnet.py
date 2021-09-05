from libcity.data import get_dataset
from libcity.utils import get_logger, get_executor, get_model

if __name__ == '__main__':
    config = {
        'log_level': 'INFO',
        'input_window': 12,
        'output_window': 12,
        'train_rate': 0.7,
        'eval_rate': 0.1,
        'cache_dataset': True,
        'batch_size': 64,
        'num_workers': 1,

        'evaluator': 'TrafficStateEvaluator',
        'dataset_class': 'TrafficStatePointDataset',
        'executor': 'TrafficStateExecutor',
        'model': 'GWNET',

        'learning_rate': 0.001,
        'learner': 'adam',
        'lr_decay': False,
        'weight_decay': 0.0001,
        'dropout': 0.3,
        'max_epoch': 100,
        'epoch': 0,
        'max_grad_norm': 5,
        'clip_grad_norm': True,

        'metrics': ['MAE', 'MSE', 'RMSE', 'MAPE', 'masked_MAE', 'masked_MSE', 'masked_RMSE', 'masked_MAPE', 'R2', 'EVAR'],
        'gpu': True,
        'gpu_id': '1',
        'dataset': 'METR_LA',
        'weight_col': 'cost',
        'data_col': ['traffic_speed'],
        'calculate_weight': True,
        'add_time_in_day': False,
        'add_day_in_week': False,
        'scaler': "standard",
        'use_early_stop': False,
    }
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']
    import torch
    config['device'] = torch.device("cuda" if torch.cuda.is_available() and config['gpu'] else "cpu")

    logger = get_logger(config)
    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    print(len(train_data.dataset), train_data.dataset[0][0].shape, train_data.dataset[0][1].shape,
          train_data.batch_size)
    print(len(valid_data.dataset), valid_data.dataset[0][0].shape, valid_data.dataset[0][1].shape,
          valid_data.batch_size)
    print(len(test_data.dataset), test_data.dataset[0][0].shape, test_data.dataset[0][1].shape, test_data.batch_size)

    data_feature = dataset.get_data_feature()
    print(data_feature['adj_mx'].shape)
    print(data_feature['adj_mx'].sum())
    model = get_model(config, data_feature)
    executor = get_executor(config, model)
    executor.train(train_data, valid_data)
    model_cache_file = './libcity/cache/model_cache/' + config['model'] + '_' + config['dataset'] + '.m'
    executor.save_model(model_cache_file)
    executor.load_model(model_cache_file)
    # 评估，评估结果将会放在 cache/evaluate_cache 下
    executor.evaluate(test_data)
