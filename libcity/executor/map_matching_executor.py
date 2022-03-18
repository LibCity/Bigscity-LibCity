from logging import getLogger
from libcity.executor.abstract_tradition_executor import AbstractTraditionExecutor
from libcity.utils import get_evaluator


class MapMatchingExecutor(AbstractTraditionExecutor):

    def __init__(self, config, model, data_feature):
        self.model = model
        self.config = config
        self.evaluator = get_evaluator(config)
        self.exp_id = self.config.get('exp_id', None)
        self.cache_dir = './libcity/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/evaluate_cache'.format(self.exp_id)
        self._logger = getLogger()

    def evaluate(self, test_data):
        """
        use model to test data

        Args:
            test_data
        """
        result = self.model.run(test_data)
        batch = {'route': test_data['route'], 'result': result, 'rd_nwk': test_data['rd_nwk']}
        self.evaluator.collect(batch)
        self.evaluator.save_result(self.evaluate_res_dir)

    def train(self, train_dataloader, eval_dataloader):
        """
        对于传统模型，不需要训练

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        pass  # do nothing
