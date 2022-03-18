from libcity.utils import get_evaluator, ensure_dir
from libcity.executor.abstract_executor import AbstractExecutor


class GensimExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        self.evaluator = get_evaluator(config)
        self.config = config
        self.model = model
        self.exp_id = config.get('exp_id', None)

        self.cache_dir = './libcity/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/evaluate_cache'.format(self.exp_id)
        ensure_dir(self.cache_dir)
        ensure_dir(self.evaluate_res_dir)

    def evaluate(self, test_dataloader):
        """
        use model to test data
        """
        self.evaluator.evaluate()

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config
        """
        self.model.run()

    def load_model(self, cache_name):
        pass

    def save_model(self, cache_name):
        pass
