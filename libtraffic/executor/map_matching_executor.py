from logging import getLogger
from libtraffic.executor.abstract_tradition_executor import AbstractTraditionExecutor
from libtraffic.utils import get_evaluator


class MapMatchingExecutor(AbstractTraditionExecutor):

    def __init__(self, config, model):
        self.model = model
        self.config = config
        self.evaluator = get_evaluator(config)
        self.evaluate_res_dir = './libtraffic/cache/evaluate_cache'
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

