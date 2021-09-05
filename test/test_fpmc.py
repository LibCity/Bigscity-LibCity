import os
import torch
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_executor, get_model

model_name = 'FPMC'
dataset_name = 'foursquare_tky'
# load config
config = ConfigParser('traj_loc_pred', 'FPMC', 'foursquare_tky', None, None)
# 加载数据集
dataset = get_dataset(config)
# 转换数据，并划分数据集
train_data, valid_data, test_data = dataset.get_data()
data_feature = dataset.get_data_feature()
batch = train_data.__iter__().__next__()
batch.to_tensor(gpu=True)
# 加载执行器
model_cache_file = './libcity/cache/model_cache/{}_{}.m'.format(model_name, dataset_name)
model = get_model(config, data_feature)
executor = get_executor(config, model)
# 训练
# if train or not os.path.exists(model_cache_file):
#     executor.train(train_data, valid_data)
#     if save_model:
#         executor.save_model(model_cache_file)
# else:
#     executor.load_model(model_cache_file)
# 评估，评估结果将会放在 cache/evaluate_cache 下
# executor.evaluate(test_data)
