import numpy as np
from gensim.models import Word2Vec
import json
from logging import getLogger
from libcity.model.abstract_traffic_tradition_model import AbstractTraditionModel

from io import open
from time import time
from collections import defaultdict, Iterable
import random


# Reference: https://github.com/phanein/deepwalk
class Graph(defaultdict):

    def __init__(self):
        super(Graph, self).__init__(list)
        self._logger = getLogger()

    def nodes(self):
        return self.keys()

    def adjacency_iter(self):
        return self.iteritems()

    def subgraph(self, nodes={}):
        subgraph = Graph()

        for n in nodes:
            if n in self:
                subgraph[n] = [x for x in self[n] if x in nodes]

        return subgraph

    def make_undirected(self):

        t0 = time()

        for v in list(self):
            for other in self[v]:
                if v != other:
                    self[other].append(v)

        t1 = time()
        self._logger.info('make_directed: added missing edges {}s'.format(t1 - t0))

        self.make_consistent()
        return self

    def make_consistent(self):
        t0 = time()
        # for k in iterkeys(self):
        #     self[k] = list(sorted(set(self[k])))
        for k in self.keys():
            self[k] = list(sorted(set(self[k])))

        t1 = time()
        self._logger.info('make_consistent: made consistent in {}s'.format(t1 - t0))

        self.remove_self_loops()

        return self

    def remove_self_loops(self):

        removed = 0
        t0 = time()

        for x in self:
            if x in self[x]:
                self[x].remove(x)
                removed += 1

        t1 = time()

        self._logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1 - t0)))
        return self

    def check_self_loops(self):
        for x in self:
            for y in self[x]:
                if x == y:
                    return True

        return False

    def has_edge(self, v1, v2):
        if v2 in self[v1] or v1 in self[v2]:
            return True
        return False

    def degree(self, nodes=None):
        if isinstance(nodes, Iterable):
            return {v: len(self[v]) for v in nodes}
        else:
            return len(self[nodes])

    def order(self):
        "Returns the number of nodes in the graph"
        return len(self)

    def number_of_edges(self):
        "Returns the number of nodes in the graph"
        return sum([self.degree(x) for x in self.keys()]) / 2

    def number_of_nodes(self):
        "Returns the number of nodes in the graph"
        return self.order()

    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        """ Returns a truncated random walk.
            path_length: Length of the random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        G = self
        if start:
            path = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(G[cur]))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]


def build_deepwalk_corpus(G, num_paths, path_length, alpha=0, rand=random.Random(0)):
    walks = []

    nodes = list(G.nodes())

    logger = getLogger()
    logger.info('Walk iteration:')
    for cnt in range(num_paths):
        logger.info(str(cnt + 1) + '/' + str(num_paths))
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))

    return walks


def from_numpy(x, directed=False):
    G = Graph()

    for i in range(x.shape[0]):
        G[i] = []

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j]:
                G[i].append(j)

    if not directed:
        G.make_undirected()

    G.make_consistent()
    return G


def learn_embeddings(walks, dimensions, window_size, workers, iters, min_count=0, sg=1, hs=0):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, vector_size=dimensions, window=window_size, min_count=min_count, sg=sg, hs=hs,
        workers=workers, epochs=iters)
    return model


class DeepWalk(AbstractTraditionModel):

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.adj_mx = data_feature.get('adj_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.geo_to_ind = data_feature.get('geo_to_ind', None)
        self.ind_to_geo = data_feature.get('ind_to_geo', None)
        self._logger = getLogger()

        self.output_dim = config.get('output_dim', 64)
        self.is_directed = config.get('is_directed', False)
        self.num_walks = config.get('num_walks', 100)
        self.walk_length = config.get('walk_length', 80)
        self.window_size = config.get('window_size', 10)
        self.num_workers = config.get('num_workers', 10)
        self.iter = config.get('max_epoch', 1000)
        self.alpha = config.get('alpha', 0.0)
        self.seed = config.get('seed', 0)

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
        g = from_numpy(self.adj_mx, self.is_directed)

        walks = build_deepwalk_corpus(g, num_paths=self.num_walks, path_length=self.walk_length,
                                      alpha=self.alpha, rand=random.Random(self.seed))

        model = learn_embeddings(walks=walks, dimensions=self.output_dim,
                                 window_size=self.window_size, workers=self.num_workers, iters=self.iter, hs=1)
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
