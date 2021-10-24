import random

import networkx as nx

from walkpro import AliasSamplingForNet

class Node2VecWalk(AliasSamplingForNet):
    """
    To generate a series of paths with node2vec method.
    p, q: hyper-parameters that control the walk way, large p will cause the walk less likely to step back while large q
    will cause the walk step nearby the start node
    weight_reverse: whether reverse the weight, which means that larger edge weight will less likely to be visited.
    names: if the form of net is adjacent matrix, the names will be the label of all the nodes, the index is corresponded
    to the index in the matrix
    """
    def __init__(self, p, q, walks=10, length=10, weight_reverse=False, names=None):
        super(Node2VecWalk, self).__init__(p=p, q=q, weight_reverse=weight_reverse)
        self.p = p
        self.q = q
        self.weight_reverse = weight_reverse
        self.__walks = walks
        self.__length = length
        self.__names = names
        self.__extrowalked = 0

    def trans_net(self, G):
        """
        transform G to networkx form
        :param G: matrix or networkx form
        :return: networkx form G
        """
        if isinstance(G, nx.classes.graph.Graph):
            return G
        elif G.shape[0] == G.shape[1]:
            if self.__names is not None:
                if len(self.__names) != G.shape[0]:
                    raise ValueError("Adjacency matrix should contain number of nodes equals to names.")
                else:
                    graph = nx.Graph()
                    for i in range(G.shape[0]):
                        for j in range(G.shape[1]):
                            graph.add_weighted_edges_from([(self.__names[i], self.__names[j], 1)])
                return graph

    def rand_path(self, G, v):
        """
        get random paths start from v
        :param G: network
        :param v: start node
        :return:
        """
        path = [v]
        length = 0
        while length < self.__length:
            if length < 1:
                v_ = self.takepoint(G, path[-1])
            else:
                v_ = self.takepoint(G, path[-2], path[-1])
            path.append(v_)
            length += 1
        return path

    def gen_walks(self, G, batch_size=1000):
        num = 0
        allwalks = []
        G = self.trans_net(G)
        node = [n for n in nx.nodes(G)]
        for i in range(self.__walks):
            random.shuffle(node)
            for n in node:
                allwalks.append(self.rand_path(G, n))
                num += 1
                if num % batch_size == 0:
                    yield allwalks
                    allwalks = []
                    num = 0
        if len(allwalks) > 0:
            yield allwalks