import warnings

warnings.filterwarnings('ignore')

from libcity.config import ConfigParser

from libcity.data import get_dataset

from libcity.utils import get_executor

from libcity.utils import get_model

# 加载模型config文件
task = 'route_planning'
model = 'MPR'
dataset = 'BJTaxi_201511'
evaluator = 'RoutePlanningEvaluator'
config = ConfigParser(task, model, dataset, config_file=None)

# 加载路段和轨迹数据， 分配训练集和验证集
dataset = get_dataset(config)
train_dataloader, test_dataloader = dataset.get_data()
data_feature = dataset.get_data_feature()

# #加载MPR模型
model = get_model(config, data_feature)

# 加载执行器
executor = get_executor(config, model)

# 生成预测轨迹
executor.train()
