import warnings
warnings.filterwarnings('ignore')

from libcity.config import ConfigParser

from libcity.data import get_dataset

from libcity.utils import get_executor

from libcity.utils import get_model


#加载模型config文件
task = 'road_representation'
model = 'Node2Vec'
dataset = 'BJ_roadmap'
evaluator = 'RoadRepresentationEvaluator'
config = ConfigParser(task, model, dataset, config_file=None)

# 加载路网数据（rel文件），生成networkx图
dataset = get_dataset(config)
dataset.get_data()
data_feature = dataset.get_data_feature()

# #加载node2vec模型
model = get_model(config, data_feature)

# 加载执行器
executor = get_executor(config, model)

# 生成node2vec游走 结果为由num_walks个长度为walk_length的一维list合成的二维list
executor.train()

