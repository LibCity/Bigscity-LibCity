from libcity.executor.abstract_executor import AbstractExecutor
from logging import getLogger
from libcity.utils import get_evaluator, ensure_dir
import numpy as np
import torch
import time
import os


class AbstractTraditionExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        self.evaluator = get_evaluator(config)
        self.config = config
        self.data_feature = data_feature
        self.device = self.config.get('device', torch.device('cpu'))
        self.model = model
        self.exp_id = self.config.get('exp_id', None)

        self.cache_dir = './libcity/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/evaluate_cache'.format(self.exp_id)

        ensure_dir(self.cache_dir)
        ensure_dir(self.evaluate_res_dir)

        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        self.output_dim = self.config.get('output_dim', 1)

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')

        y_truths = []
        y_preds = []
        for batch in test_dataloader:
            batch.to_ndarray()
            output = self.model.run(batch)
            y_true = self._scaler.inverse_transform(batch['y'][..., :self.output_dim])
            y_pred = self._scaler.inverse_transform(output[..., :self.output_dim])
            y_truths.append(y_true)
            y_preds.append(y_pred)

        y_preds = np.concatenate(y_preds, axis=0)
        y_truths = np.concatenate(y_truths, axis=0)  # concatenate on batch
        outputs = {'prediction': y_preds, 'truth': y_truths}
        filename = \
            time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + '_' \
            + self.config['model'] + '_' + self.config['dataset'] + '_predictions.npz'
        np.savez_compressed(os.path.join(self.evaluate_res_dir, filename), **outputs)
        self.evaluator.clear()
        self.evaluator.collect({'y_true': torch.tensor(y_truths), 'y_pred': torch.tensor(y_preds)})
        test_result = self.evaluator.save_result(self.evaluate_res_dir)
        return test_result

    def train(self, train_dataloader, eval_dataloader):
        """
        train model

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        raise NotImplementedError

    def save_model(self, cache_name):
        """
        对于传统模型，不需要模型保存

        Args:
            cache_name(str): 保存的文件名
        """
        assert True  # do nothing

    def load_model(self, cache_name):
        """
        对于传统模型，不需要模型加载

        Args:
            cache_name(str): 保存的文件名
        """
        assert True  # do nothing
