import random
import json
import networkx as nx
import numpy as np
from gensim.models import Word2Vec

from logging import getLogger
from libcity.model.abstract_traffic_tradition_model import AbstractTraditionModel


# Reference: https://github.com/aditya-grover/node2vec
class Graph():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self._logger = getLogger()

    def node2vec_walk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                                               alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        """
        Repeatedly simulate random walks from each node.
        """
        G = self.G
        walks = []
        nodes = list(G.nodes())

        self._logger.info('Walk iteration:')
        for walk_iter in range(num_walks):
            self._logger.info(str(walk_iter + 1) + '/' + str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to
    https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def learn_embeddings(walks, dimensions, window_size, workers, iters, min_count=0, sg=1, hs=0):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, vector_size=dimensions, window=window_size, min_count=min_count, sg=sg, hs=hs,
        workers=workers, epochs=iters)
    return model


class Node2Vec(AbstractTraditionModel):

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.adj_mx = data_feature.get('adj_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.geo_to_ind = data_feature.get('geo_to_ind', None)
        self.ind_to_geo = data_feature.get('ind_to_geo', None)
        self._logger = getLogger()

        self.output_dim = config.get('output_dim', 64)
        self.is_directed = config.get('is_directed', True)
        self.p = config.get('p', 2)
        self.q = config.get('q', 1)
        self.num_walks = config.get('num_walks', 100)
        self.walk_length = config.get('walk_length', 80)
        self.window_size = config.get('window_size', 10)
        self.num_workers = config.get('num_workers', 10)
        self.iter = config.get('max_epoch', 1000)

        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)
        self.txt_cache_file = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.txt'.\
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.model_cache_file = './libcity/cache/{}/model_cache/embedding_{}_{}_{}.m'.\
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.npy_cache_file = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.npy'.\
            format(self.exp_id, self.model, self.dataset, self.output_dim)

    def run(self, data=None):
        nx_g = nx.from_numpy_matrix(self.adj_mx, create_using=nx.DiGraph())
        g = Graph(nx_g, self.is_directed, self.p, self.q)
        g.preprocess_transition_probs()

        walks = g.simulate_walks(self.num_walks, self.walk_length)

        model = learn_embeddings(walks=walks, dimensions=self.output_dim,
                                 window_size=self.window_size, workers=self.num_workers, iters=self.iter)
        model.wv.save_word2vec_format(self.txt_cache_file)
        model.save(self.model_cache_file)

        assert len(model.wv) == self.num_nodes
        assert len(model.wv[0]) == self.output_dim

        node_embedding = np.zeros(shape=(self.num_nodes, self.output_dim), dtype=np.float32)
        f = open(self.txt_cache_file, mode='r')
        lines = f.readlines()
        for line in lines[1:]:
            temp = line.split(' ')
            index = int(temp[0])
            node_embedding[index] = temp[1:]
        np.save(self.npy_cache_file, node_embedding)

        self._logger.info('词向量和模型保存完成')
        self._logger.info('词向量维度：(' + str(len(model.wv)) + ',' + str(len(model.wv[0])) + ')')
        json.dump(self.ind_to_geo, open('./libcity/cache/{}/evaluate_cache/ind_to_geo_{}.json'.format(
            self.exp_id, self.dataset), 'w'))
        json.dump(self.geo_to_ind, open('./libcity/cache/{}/evaluate_cache/geo_to_ind_{}.json'.format(
            self.exp_id, self.dataset), 'w'))
