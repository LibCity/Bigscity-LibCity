# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import argparse as arg
import warnings
warnings.filterwarnings('ignore')
from node2vec import Node2vec
from node2vec_dataset import ReadGraph, Node2vecDataset
from gensim.models import Word2Vec

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = arg.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='',
                        help='Input graph path')
    parser.add_argument('--p', type=float, default=0.25,help='Return hyperparameter. Default is 0.25.')
    parser.add_argument('--q', type=float, default=4, help='Return hyperparameter. Default is 4.')
    parser.add_argument('--walks', type=int, default=80,
                        help='Length of walk per source. Default is 80.')
    parser.add_argument('--length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')
    parser.add_argument('--dimension', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    parser.add_argument('--window', type=int, default=5,
                        help='Context size for optimization. Default is 10.')
    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')
    return parser.parse_args()

#word2vec模型
def learn_embeddings(self, walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''

    model = Word2Vec(walks, vector_size=args.dimension, window=args.window, min_count=0, sg=1, workers=args.workers,
                     epochs=args.iter)
    save_path = self.cache_file_folder + '{}_embedding.bin'.format(self.dataset)
    model.wv.save_word2vec_format(save_path)
    return

if __name__ == '__main__':
    args = parse_args()
    config = {'dataset': 'BJ_roadmap'}
    #根据路网数据（rel文件）生成networkx图
    RG = ReadGraph(config)
    G = RG._load_graph()

    #生成node2vec游走 结果为由num_walks个长度为walk_length的一维list合成的二维list
    rw = Node2vec(G, p=args.p, q=args.q, use_rejection_sampling=0)
    rw.preprocess_transition_probs()
    sentences = rw.simulate_walks(num_walks=args.walks, walk_length=args.length)

    data = Node2vecDataset(config, sentences)

    #将游走结果分为训练集(train)、验证集(valid)、测试集(test)三个部分，并带入word2vec模型进行训练，输出最终结果
    data.train_dataloader, data.eval_dataloader, data.test_dataloader = data.get_data()
    train_walk = data.train_dataloader['mask']
    learn_embeddings(data, train_walk)
























