# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import argparse as arg
import warnings
warnings.filterwarnings('ignore')
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import networkx as nx
from node2vec import Node2Vec

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = arg.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')
    parser.add_argument('--p', type=float, default=1,help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=1, help='Return hyperparameter. Default is 1.')
    parser.add_argument('--dimension', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    parser.add_argument('--window', type=int, default=5,
                        help='Context size for optimization. Default is 10.')
    parser.add_argument('--walks', type=int, default=80,
                        help='Length of walk per source. Default is 80.')
    parser.add_argument('--length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')

    return parser.parse_args()

def read_graph():
    file = open(args.input, 'r')
    list=[]
    try:
        while True:
            text_line = file.readline()
            if text_line:
                text=text_line.split( )
                list.append(text)
            else:
                break
    finally:
        file.close()
    G = nx.Graph()
    for node in list:
        i=node[0]
        j=node[1]
        if i not in G.nodes():
            G.add_node(i)
        elif j not in G.nodes():
            G.add_node(j)
        G.add_weighted_edges_from([(i, j, 1.0)])
    return G


if __name__ == '__main__':
    args = parse_args()
    G = read_graph()
    nv = Node2Vec(p=args.p, q=args.q, weight_reverse=False, dimension=args.dimension, window=args.window, walks=args.walks, length=args.length, lr=0.01,
                    sampling='negative', negative=10)
    nv.train_data(G, epochs=5, batch_size=1000, verbose_step=1)
    nodes = [node for node in nv.embedding.keys()]
    emb = [nv.embedding[node] for node in nodes]
    tsne = TSNE(n_components=2, init='pca', random_state=2)
    result = tsne.fit_transform(emb)
    nx.draw(G, with_labels=True)
    plt.show()
    plt.scatter(result[:, 0], result[:, 1], 200)
    for ind, node in enumerate(nodes):
        plt.text(result[ind, 0], result[ind, 1], node)
    plt.title('轨迹图')
    plt.show()
