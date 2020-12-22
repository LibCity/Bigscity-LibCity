import json
import time

import numpy as np
import os
import shutil
import re
import sys

from trafficdl.evaluator.eval_funcs import ACC, top_k, SMAPE, RMSE, MAPE, MARE, MSE, MAE
from trafficdl.evaluator.utils import output, transfer_data


class TrajLocPredEvaluator(object):

    def __init__(self, config):
        """
        Initialize the creation of the Evaluate Class
        :param config: 用于传递 global_config
        """
        # 从配置文件中读取相关参数
        self.config = config
        self.output_switch = False
        self.mode_list = ["ACC", "RMSE", "MSE", "MAE", "MAPE", "MARE", "SMAPE", "top-k"]
        self.data_path = ""
        self.model = self.config['model']
        self.mode = ['ACC']
        # 初始化类的内部变量
        self.topK = 1
        self.maxK = 1
        self.topK_pattern = re.compile("top-[1-9]\\d*$")
        self.data = None
        self.data_batch = 0
        self.metrics = {}
        self.trace_metrics = {}
        self.model_metrics = {}
        self.para_metrics = {}
        # 检查是否有不支持的配置
        self.check_config()

    def check_config(self):
        # check mode
        for mode in self.mode:
            if mode in self.mode_list:
                self.trace_metrics[mode] = []
            elif re.match(self.topK_pattern, mode) is not None:
                k = int(mode.split('-')[1])
                self.trace_metrics[mode] = []
                self.maxK = k if k > self.maxK else self.maxK
            else:
                raise ValueError("{} 是不支持的评估方法".format(mode))

    def evaluate(self, data=None):
        """
        The entrance of evaluation (user-oriented)
        :param data: 待评估数据, 可以直接是dict类型或者str形式的dict类型，也可以是列表类型(分batch)
        """
        if data is not None:
            self.data = data
        else:
            try:
                with open(self.data_path) as f:
                    self.data = json.load(f)
            except Exception:
                raise ValueError('待评估数据的路径无效')
        if isinstance(self.data, list):
            data_list = self.data
            for batch_data in data_list:
                self.data = batch_data
                self.evaluate_data()
        else:
            self.evaluate_data()

    def save_result(self, result_path=None):
        """
        :param result_path: 绝对路径，存放结果json
        :return: 文件名
        """
        if result_path is None:
            raise ValueError('请正确指定保存评估结果的绝对路径')
        self.calculate_mode_metrics()
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        now = time.strftime("%Y-%m-%d", time.localtime(time.time()))
        filename = result_path + '/res_' + self.model + '_' + now + '.txt'
        with open(filename, "w") as f:
            metrics = {'model': self.model_metrics, 'data': self.metrics}
            f.write(json.dumps(metrics, indent=1))
        return filename

    def evaluate_data(self):
        """
        evaluate data batch (internal)
        """
        self.metrics[self.generate_name()] = {}
        self.para_metrics[self.generate_name()] = {}
        self.data = transfer_data(self.data, self.model, self.maxK)
        loc_true = []
        loc_pred = []
        user_ids = self.data.keys()
        for user_id in user_ids:
            user = self.data[user_id]
            trace_ids = user.keys()
            for trace_id in trace_ids:
                trace = user[trace_id]
                t_loc_true = trace['loc_true']
                t_loc_pred = trace['loc_pred']
                loc_true.extend(t_loc_true)
                loc_pred.extend(t_loc_pred)
                self.run_mode(t_loc_pred, t_loc_true, 'trace')
        self.para_metrics[self.generate_name()]['data_size'] = len(loc_true)
        self.run_mode(loc_pred, loc_true, 'model')
        self.data_batch = self.data_batch + 1

    def run_mode(self, loc_pred, loc_true, field):
        """
        The method of run evaluate (internal)
        :param loc_pred: 模型预测出位置的结果
        :param loc_true: 数据集的真实位置
        :param field: 是对轨迹进行评估还是对整个模型进行评估 (internal)
        """
        assert len(loc_pred) == len(loc_true), "评估的预测数据与真实数据大小不一致"
        t_loc_pred = [[] for i in range(self.maxK)]
        for i in range(len(loc_true)):
            assert len(loc_pred[i]) >= self.maxK, "模型的位置预测结果少于top-{}评估方法的k值".format(self.maxK)
            for j in range(self.maxK):
                t_loc_pred[j].append(loc_pred[i][j])
        for mode in self.mode:
            if mode == 'ACC':
                t, avg_acc = ACC(np.array(t_loc_pred[0]), np.array(loc_true))
                self.add_metrics(mode, field, avg_acc)
                self.para_metrics[self.generate_name()][mode] = np.sum(t == 0)
            elif re.match(self.topK_pattern, mode) is not None:
                t, avg_acc = top_k(np.array(t_loc_pred, dtype=object), np.array(loc_true, dtype=object),
                                   int(mode.split('-')[1]))
                self.add_metrics(mode, field, avg_acc)
                self.para_metrics[self.generate_name()][mode] = np.sum(t < int(mode.split('-')[1]))
            else:
                avg_loss = 0
                if mode == "SMAPE":
                    avg_loss = SMAPE(np.array(t_loc_pred[0]), np.array(loc_true))
                elif mode == 'RMSE':
                    avg_loss = RMSE(np.array(t_loc_pred[0]), np.array(loc_true))
                elif mode == "MAPE":
                    avg_loss = MAPE(np.array(t_loc_pred[0]), np.array(loc_true))
                elif mode == "MARE":
                    avg_loss = MARE(np.array(t_loc_pred[0]), np.array(loc_true))
                elif mode == 'MSE':
                    avg_loss = MSE(np.array(t_loc_pred[0]), np.array(loc_true))
                elif mode == "MAE":
                    avg_loss = MAE(np.array(t_loc_pred[0]), np.array(loc_true))
                self.add_metrics(mode, field, avg_loss)

    def add_metrics(self, method, field, avg):
        """
        save every trace metrics or the whole model metrics
        :param method: evaluate method
        :param field: trace or model
        :param avg: avg_acc or avg_loss
        """
        if self.output_switch:
            output(method, avg, field)
        if field == 'model':
            self.metrics['data batch ' + str(self.data_batch)][method] = avg
        else:
            self.trace_metrics[method].append(avg)

    def calculate_mode_metrics(self):
        if self.data_batch == 1:
            self.model_metrics = {}
            for data in self.metrics.keys():
                for mode in self.metrics[data].keys():
                    self.model_metrics[mode] = self.metrics[data][mode]
        else:
            self.model_metrics = {}
            self.para_metrics['model'] = {}
            self.para_metrics['model']['data_size'] = 0
            for data in self.para_metrics.keys():
                for mode in self.para_metrics[data].keys():
                    if mode == 'data_size':
                        self.para_metrics['model']['data_size'] += self.para_metrics[data][mode]
                    elif mode in self.para_metrics['model'].keys():
                        self.para_metrics['model'][mode] += self.para_metrics[data][mode]
                    else:
                        self.para_metrics['model'][mode] = self.para_metrics[data][mode]
            for mode in self.para_metrics['model'].keys():
                if mode == 'data_size':
                    continue
                self.model_metrics[mode] = self.para_metrics['model'][mode] / self.para_metrics['model']['data_size']

    def generate_name(self):
        return 'data batch ' + str(self.data_batch)
