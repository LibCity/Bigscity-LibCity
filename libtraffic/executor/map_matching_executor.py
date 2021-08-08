from libtraffic.executor.abstract_executor import AbstractExecutor
from libtraffic.utils.utils import ensure_dir
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter


class MapMatchingExecutor(AbstractExecutor):

    def __init__(self, config, model):
        self.model = model
        self.config = config
        self.res_dir = './libtraffic/cache/result_cache'
        self._logger = getLogger()

    def evaluate(self, test_data):
        """
        use model to test data

        Args:
            test_data
        """
        result = self.model.run(test_data)
        self._save_result(result)

    def _save_result(self, result):
        ensure_dir(self.res_dir)
        file_name = self.config.get('dataset', '') + '.out'
        with open(self.res_dir + '/' + file_name, 'w') as f:
            f.write('dyna_id,rel_id\n')
            for line in result:
                f.write(str(line[0]) + ',' + str(line[1]) + '\n')

    def train(self, train_dataloader, eval_dataloader):
        assert True  # do nothing

    def load_model(self, cache_name):
        assert True  # do nothing

    def save_model(self, cache_name):
        assert True  # do nothing
