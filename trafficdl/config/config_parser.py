import os
import json
import torch


class ConfigParser(object):
    """
    use to parse the user defined parameters and use these to modify the
    pipeline's parameter setting.
    值得注意的是，目前各阶段的参数是放置于同一个 dict 中的，因此需要编程时保证命名空间不冲突
    config 优先级：命令行 > config file > default config
    """

    def __init__(self, task, model, dataset, config_file=None,
                 other_args=None):
        """
        Args:
            task, model, dataset (str): 用户在命令行必须指明的三个参数
            config_file (str): 配置文件的文件名，将在项目根目录下进行搜索
            other_args (dict): 通过命令行传入的其他参数
        """
        self.config = {}
        self._parse_external_config(task, model, dataset, other_args)
        self._parse_config_file(config_file)
        self._load_default_config()
        self._init_device()

    def _parse_external_config(self, task, model, dataset, other_args=None):
        if task is None:
            raise ValueError('the parameter task should not be None!')
        if model is None:
            raise ValueError('the parameter model should not be None!')
        if dataset is None:
            raise ValueError('the parameter dataset should not be None!')
        # 目前暂定这三个参数必须由用户指定
        self.config['task'] = task
        self.config['model'] = model
        self.config['dataset'] = dataset
        if other_args is not None:
            # TODO: 这里可以设计加入参数检查，哪些参数是允许用户通过命令行修改的
            for key in other_args:
                self.config[key] = other_args[key]

    def _parse_config_file(self, config_file):
        if config_file is not None:
            # TODO: 对 config file 的格式进行检查
            if os.path.exists('./{}.json'.format(config_file)):
                with open('./{}.json'.format(config_file), 'r') as f:
                    x = json.load(f)
                    for key in x:
                        if key not in self.config:
                            self.config[key] = x[key]
            else:
                raise FileNotFoundError(
                    'Config file {}.json is not found. Please ensure \
                    the config file is in the root dir and is a JSON \
                    file.'.format(config_file))

    def _load_default_config(self):
        # 首先加载 task config
        with open('./trafficdl/config/task_config.json', 'r') as f:
            task_config = json.load(f)
            if self.config['task'] not in task_config:
                raise ValueError(
                    'task {} is not supported.'.format(self.config['task']))
            task_config = task_config[self.config['task']]
            # check model and dataset
            if self.config['model'] not in task_config['allowed_model']:
                raise ValueError('task {} do not support model {}'.format(
                    self.config['task'], self.config['model']))
            model = self.config['model']
            # 加载 dataset、executor、evaluator 的模块
            if 'dataset_class' not in self.config:
                self.config['dataset_class'] = \
                    task_config[model]['dataset_class']
            if 'executor' not in self.config:
                self.config['executor'] = task_config[model]['executor']
            if 'evaluator' not in self.config:
                self.config['evaluator'] = task_config[model]['evaluator']
            # 对于 LSTM RNN GRU 使用的都是同一个类，只是 RNN 模块不一样而已，这里做一下修改
            if self.config['model'] in ['LSTM', 'GRU', 'RNN']:
                self.config['model'] = 'RNN'
                self.config['rnn_type'] = self.config['model']
            if self.config['dataset'] not in task_config['allowed_dataset']:
                raise ValueError('task {} do not support dataset {}'.format(
                    self.config['task'], self.config['dataset']))
        # 接着加载每个阶段的 default config
        default_file_list = []
        default_file_list.append(
            'data/{}.json'.format(self.config['dataset_class']))
        # executor
        default_file_list.append(
            'executor/{}.json'.format(self.config['executor']))
        # evaluator
        default_file_list.append(
            'evaluator/{}.json'.format(self.config['evaluator']))
        # model
        default_file_list.append('model/{}.json'.format(self.config['model']))
        # 加载所有默认配置
        for file_name in default_file_list:
            with open('./trafficdl/config/{}'.format(file_name), 'r') as f:
                x = json.load(f)
                for key in x:
                    if key not in self.config:
                        self.config[key] = x[key]

    def _init_device(self):
        use_gpu = self.config.get('gpu', True)
        gpu_id = self.config.get('gpu_id', 0)
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.config['device'] = torch.device(
            "cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    def __getitem__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise KeyError('{} is not in the config'.format(key))

    def __setitem__(self, key, value):
        if key in self.config:
            self.config[key] = value
        else:
            raise KeyError('{} is not in the config'.format(key))

    def __contains__(self, key):
        return key in self.config

    # 支持迭代操作
    def __iter__(self):
        return self.config.__iter__()
