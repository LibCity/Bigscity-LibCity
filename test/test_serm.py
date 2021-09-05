import os
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_executor, get_model
# import pandas as pd

# raw_data = pd.read_csv('./raw_data/serm_foursquare.txt', sep='', header=None)

model_name = 'SERM'
dataset_name = 'foursquare_tky'

config = ConfigParser('traj_loc_pred', 'SERM', 'foursquare_serm', None, None)
dataset = get_dataset(config)
train_data, valid_data, test_data = dataset.get_data()
data_feature = dataset.get_data_feature()
batch = valid_data.__iter__().__next__()
batch.to_tensor(config['device'])

# 加载执行器
model_cache_file = './libcity/cache/model_cache/{}_{}.m'.format(
    model_name, dataset_name)
model = get_model(config, data_feature)
executor = get_executor(config, model)

# 训练
train = True
save_model = True
if train or not os.path.exists(model_cache_file):
    executor.train(train_data, valid_data)
    if save_model:
        executor.save_model(model_cache_file)
    else:
        executor.load_model(model_cache_file)
# 评估，评估结果将会放在 cache/evaluate_cache 下
executor.evaluate(test_data)
