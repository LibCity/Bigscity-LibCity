
class Node2VecExecutor():
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
    def run_model(self):
        self.model.preprocess_transition_probs()
        self.model.simulate_walks(num_walks=self.walks, walk_length=self.length)
        #将游走结果带入word2vec模型进行训练，输出最终结果
        self.model.learn_embeddings(vector_size=self.dimension, window=self.window, min_count=0, sg=1, workers=self.workers,
                            epochs=self.iter)

