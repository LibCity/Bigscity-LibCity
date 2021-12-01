from libcity.executor import AbstractExecutor


class Node2VecExecutor(AbstractExecutor):
    def __init__(self, config, model):
        self.model = model

        self.config = config
        self.p = self.config.get('p', '')
        self.q = self.config.get('q', '')
        self.walks = self.config.get('walks', '')
        self.length = self.config.get('length', '')
        self.dimension = self.config.get('dimension', '')
        self.window = self.config.get('window', '')
        self.iter = self.config.get('iter', '')
        self.workers = self.config.get('workers', '')
    def train(self, train_dataloader = None, eval_dataloader = None):
        self.model.preprocess_transition_probs()
        self.model.simulate_walks(num_walks=self.walks, walk_length=self.length)
        #将游走结果带入word2vec模型进行训练，输出最终结果
        self.model.learn_embeddings(vector_size=self.dimension, window=self.window, min_count=0, sg=1, workers=self.workers,
                            epochs=self.iter)

    #train_data 与 valid_data 仅保持接口一致性，不会在本方法中调用
    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        pass

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


