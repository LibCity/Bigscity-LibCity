class AbstractExecutor(object):

    def __init__(self, config, model):
        raise NotImplementedError("Executor not implemented")

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config
        """
        raise NotImplementedError("Executor train not implemented")

    def evaluate(self, test_dataloader):
        """
        use model to test data
        评估结果将会保存到 cache 文件下
        """
        raise NotImplementedError("Executor evaluate not implemented")

    def load_model(self, cache_name):
        """
        加载对应模型的 cache
        """
        raise NotImplementedError("Executor load cache not implemented")

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件内
        """
        raise NotImplementedError("Executor save cache not implemented")
