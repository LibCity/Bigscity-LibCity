import warnings
warnings.filterwarnings('ignore')

from libcity.data import get_dataset

from libcity.utils import get_executor

from libcity.utils import get_model


config = {
    'task' : 'road_representation',
    'dataset': 'BJ_roadmap',
    'dataset_class' : 'Node2VecDataset',
    'model' : 'Node2vec',
    'executor': 'Node2VecExecutor',
    'p': 0.25,
    'q': 4,
    'walks': 80,
    'length': 80,
    'dimension': 128,
    'window': 5,
    'iter': 1,
    'workers': 8
}

# 加载路网数据（rel文件），生成networkx图
dataset = get_dataset(config)
dataset.get_data()
data_feature = dataset.get_data_feature()

# #加载node2vec模型
model = get_model(config, data_feature)

# 加载执行器
executor = get_executor(config, model)

# 生成node2vec游走 结果为由num_walks个长度为walk_length的一维list合成的二维list
executor.run_model()

