# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import argparse as arg
import warnings
warnings.filterwarnings('ignore')
import random
import node2vec
from node2vec import RandomWalker
from node2vec_dataset import ReadGraph, Node2vecDataset

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = arg.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='',
                        help='Input graph path')
    parser.add_argument('--p', type=float, default=0.25,help='Return hyperparameter. Default is 0.25.')
    parser.add_argument('--q', type=float, default=4, help='Return hyperparameter. Default is 4.')
    # parser.add_argument('--dimension', type=int, default=128,
    #                     help='Number of dimensions. Default is 128.')
    # parser.add_argument('--window', type=int, default=5,
    #                     help='Context size for optimization. Default is 10.')
    parser.add_argument('--walks', type=int, default=80,
                        help='Length of walk per source. Default is 80.')
    parser.add_argument('--length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    config = {'dataset': 'BJ_roadmap'}
    #根据路网数据（rel文件）生成networkx图
    RG = ReadGraph(config)
    G = RG._load_graph()

    #生成node2vec游走 结果为由num_walks个长度为walk_length的一维list合成的二维list
    rw = RandomWalker(G, p=args.p, q=args.q, use_rejection_sampling=0)
    rw.preprocess_transition_probs()
    sentences = rw.simulate_walks(num_walks=args.walks, walk_length=args.length)
    #将游走结果导入，转换为rel文件
    NV = Node2vecDataset(config,sentences)
    NV.sentence_form_rel()
    #生成邻接矩阵
    NV._load_rel()















