from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_executor, get_model

config = ConfigParser('traj_loc_pred', 'DeepMove', 'foursquare_tky',
                      other_args={'evaluate_method': 'sample', 'dataset_class': 'PBSTrajectoryDataset'})
# 加载数据集
dataset = get_dataset(config)
# 转换数据，并划分数据集
train_data, valid_data, test_data = dataset.get_data()
# batch = train_data.__iter__().__next__()
# batch.to_tensor(gpu=True)
data_feature = dataset.get_data_feature()
# 加载执行器
model_cache_file = './libcity/cache/model_cache/DeepMove_foursquare_tky.m'
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
