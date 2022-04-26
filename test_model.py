from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_model, get_executor, get_logger, set_random_seed
import random

"""
取一个batch的数据进行初步测试
Take the data of a batch for preliminary testing
"""

# 加载配置文件
config = ConfigParser(task='traffic_state_pred', model='RNN',
                      dataset='METR_LA', other_args={'batch_size': 2})
exp_id = config.get('exp_id', None)
if exp_id is None:
    exp_id = int(random.SystemRandom().random() * 100000)
    config['exp_id'] = exp_id
# logger
logger = get_logger(config)
logger.info(config.config)
# seed
seed = config.get('seed', 0)
set_random_seed(seed)
# 加载数据模块
dataset = get_dataset(config)
# 数据预处理，划分数据集
train_data, valid_data, test_data = dataset.get_data()
data_feature = dataset.get_data_feature()
# 抽取一个 batch 的数据进行模型测试
batch = train_data.__iter__().__next__()
# 加载模型
model = get_model(config, data_feature)
model = model.to(config['device'])
# 加载执行器
executor = get_executor(config, model, data_feature)
# 模型预测
batch.to_tensor(config['device'])
res = model.predict(batch)
logger.info('Result shape is {}'.format(res.shape))
logger.info('Success test the model!')
