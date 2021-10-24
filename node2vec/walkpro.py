import networkx as nx

import numpy as np

import aliasmethod

class AliasSamplingForNet:
    """
    A method to quickly sample from non-uniform discrete distribution
    Specially designed for net
    :return:
    """
    def __init__(self, p, q, weight_reverse=False, name=None):
        self.p = p
        self.q = q
        self.weight_reverse = weight_reverse
        self.name = name
        self.fitted = False

    def fit_net(self, net):
        """
        net is a networkx.Graph or networkx.DiGraph
        :param net:
        :return:
        """
        if isinstance(net, nx.Graph):
            directed = False
        elif isinstance(net, nx.DiGraph):
            directed = True
        else:
            raise TypeError('Only support input type "networkx.Graph" or "networkx.DiGraph". '
                            'But the input type is {}'.format(type(net)))
        self._2nd_option = {}  # if node is from the start, use this to sample the node
        for node in net.nodes:
            if not self.weight_reverse:
                edge_weight = [net[node][neig]['weight'] for neig in sorted(net.neighbors(node))]
            else:
                edge_weight = [net[node][neig]['weight'] for neig in sorted(net.neighbors(node))]
                ave = np.mean(edge_weight)
                edge_weight = [v - (ave - v) for v in edge_weight]
            neig_probs = self.normalize(edge_weight)
            self._2nd_option[node] = aliasmethod.alias_setup(neig_probs)
        self._3rd_option = {}  # if node is after the 2rd node along the path, use this to sample the node.
        for src, _2nd in net.edges:
            if directed:
                self._3rd_option[(src, _2nd)] = aliasmethod.alias_setup(self._get_edge_probs(net, src, _2nd))
            else:
                self._3rd_option[(src, _2nd)] = aliasmethod.alias_setup(self._get_edge_probs(net, src, _2nd))
                self._3rd_option[(_2nd, src)] = aliasmethod.alias_setup(self._get_edge_probs(net, _2nd, src))
        self.fitted = True

    def _get_edge_probs(self, net, src, _2nd):
        """
        get probs of an edge
        :param net:
        :param src:
        :param _2nd:
        :return:
        """
        ngrs = sorted(net.neighbors(_2nd))
        if not self.weight_reverse:
            edge_weight = [net[_2nd][neig]['weight'] for neig in ngrs]
        else:
            edge_weight = [net[_2nd][neig]['weight'] for neig in ngrs]
            ave = np.mean(edge_weight)
            edge_weight = [v - (ave - v) for v in edge_weight]
        neig_probs = []
        for ind, n in enumerate(ngrs):
            if n == src:
                neig_probs.append(edge_weight[ind] / self.p)
            elif net.has_edge(_2nd, n):
                neig_probs.append(edge_weight[ind])
            else:
                neig_probs.append(edge_weight[ind] / self.q)
        neig_probs = self.normalize(neig_probs)
        return neig_probs

    def normalize(self, x):
        """
        normalize x
        :param x:
        :return:
        """
        return np.array(x) / np.sum(x)

    def takepoint(self, net, v1, v2=None):
        """
        sample the next node for a path
        :param net: should be networkx.Graph or networkx.DiGraph
        :param v1:
        :param v2:
        :return:
        """
        if isinstance(net, nx.Graph) or isinstance(net, nx.DiGraph):
            pass
        else:
            raise TypeError('Only support input type "networkx.Graph" or "networkx.DiGraph". '
                            'But the input type is {}'.format(type(net)))
        if not self.fitted:
            self.fit_net(net)
        if v2 is None:
            ngrs = sorted(net.neighbors(v1))
            return ngrs[aliasmethod.alias_draw(self._2nd_option[v1][0], self._2nd_option[v1][1])]
        else:
            ngrs = sorted(net.neighbors(v2))
            return ngrs[aliasmethod.alias_draw(self._3rd_option[(v1, v2)][0], self._3rd_option[(v1, v2)][1])]