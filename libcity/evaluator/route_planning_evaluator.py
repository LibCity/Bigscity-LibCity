import os
import json
import time
from libcity.evaluator.abstract_evaluator import AbstractEvaluator

allowed_metrics = ['Precision', 'Recall', 'F1', 'EDT']


class RoutePlanningEvaluator(AbstractEvaluator):
    """
    evaluator's metrics is referred from paper 'Empowering A* Search Algorithms with Neural Networks for
    Personalized Route Recommendation'
    """
    def __init__(self, config):
        self.metrics = config['metrics']
        self.result = {}
        self.intermediate_result = {
            'true_trace_len': 0,
            'gen_trace_len': 0,
            'hit': 0,
            'edit_distance': 0,
            'total_trace': 0
        }
        self._check_config()

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        for i in self.metrics:
            if i not in allowed_metrics:
                raise ValueError('the metric is not allowed in RoutePlanningEvaluator')

    def collect(self, batch):
        """
        collect a batch of input

        Args:
            batch(dict): the input dict, which contains two key: generate_trace and true_trace
        """
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict of user')
        total = len(batch['true_trace'])
        self.intermediate_result['total_trace'] += total
        for i in range(total):
            generate_trace_i = [x[0] for x in batch['generate_trace'][i][1]]
            true_trace_i = batch['true_trace'][i]
            self.intermediate_result['true_trace_len'] += len(true_trace_i)
            self.intermediate_result['gen_trace_len'] += len(generate_trace_i)
            set_generate = set(generate_trace_i)
            set_true = set(true_trace_i)
            set_intersection = set_generate.intersection(set_true)
            self.intermediate_result['hit'] += len(set_intersection)
            self.intermediate_result['edit_distance'] += self.edit_distance(generate_trace_i, true_trace_i)

    def evaluate(self):
        """
        return evaluate result
        """
        precision = self.intermediate_result['hit'] / self.intermediate_result['gen_trace_len']
        recall = self.intermediate_result['hit'] / self.intermediate_result['true_trace_len']
        f1_score = (2 * precision * recall) / (precision + recall)
        avg_edit_distance = self.intermediate_result['edit_distance'] / self.intermediate_result['total_trace']
        if 'Precision' in self.metrics:
            self.result['Precision'] = precision
        if 'Recall' in self.metrics:
            self.result['Recall'] = recall
        if 'F1' in self.metrics:
            self.result['F1'] = f1_score
        if 'EDT' in self.metrics:
            self.result['EDT'] = avg_edit_distance
        return self.result

    def save_result(self, save_path, filename=None):
        """
        save evaluate result to save_path

        Args:
            save_path: 保存路径
            filename: 保存文件名
        """
        self.evaluate()
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if filename is None:
            # 使用时间戳
            filename = time.strftime(
                "%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
        print('evaluate result is ', json.dumps(self.result, indent=1))
        with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') \
                as f:
            json.dump(self.result, f)

    def clear(self):
        """
        reset intermediate_result
        """
        self.result = {}
        self.intermediate_result = {
            'true_trace_len': 0,
            'gen_trace_len': 0,
            'hit': 0,
            'edit_distance': 0,
            'total_trace': 0
        }

    @staticmethod
    def edit_distance(trace1, trace2):
        """
        the edit distance between two trajectory

        Args:
            trace1:
            trace2:

        Returns:
            edit_distance
        """
        matrix = [[i + j for j in range(len(trace2) + 1)] for i in range(len(trace1) + 1)]
        for i in range(1, len(trace1) + 1):
            for j in range(1, len(trace2) + 1):
                if trace1[i - 1] == trace2[j - 1]:
                    d = 0
                else:
                    d = 1
                matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
        return matrix[len(trace1)][len(trace2)]
