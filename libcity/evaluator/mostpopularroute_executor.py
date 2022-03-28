from libcity.executor import AbstractExecutor


class MPRExecutor(AbstractExecutor):
    def __init__(self, config, model):
        self.model = model
        self.config = config

    def train(self, train_dataloader, eval_dataloader=None):
        '''
        valid_data 仅保持接口一致性，不会在本方法中调用
        :param train_dataloader: Dataloader
        :param eval_dataloader:
        '''
        self.model.fit(train_dataloader.dataset.data)

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        for traj_id, trace in enumerate(test_dataloader.dataset.data):
            start = trace[0]
            end = trace[-1]
            route = self.model.predict((start, end))

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        pass

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件

        Args:
            cache_name(str): 保存的文件名
        """
        pass