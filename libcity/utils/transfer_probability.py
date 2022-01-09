import math
import sys
import numpy as np
import scipy as sp


def check_node_absorbing(nodes, edges, node):
    node_set = [(node, t) for t in list(nodes.keys())]
    for node in node_set:
        if node in list(edges.keys()):
            return False
    return True


class TransferProbability:
    def __init__(self, nodes, edges, trajectories):
        self.nodes = nodes
        self.edges = edges
        self.trajectories = trajectories
        self.p = None
        self.q = None
        self.s = None
        self.derive()

    def derive(self):
        """
        Derive matrix P ,matrix Q ,matrix S and column vector V of each node.

        """
        for node in self.nodes:
            self.p = self.create_transition_matrix(node)
            self.q, self.s = self.reorganize(self.p, node)
            node.vector = self.cal_vector(node, self.p, self.q)

    def create_transition_matrix(self, d):
        """
        Construct the transition matrix P by function transition_probability.

        :param Point d: the destination node
        :return: matrix P

        """
        nodes_len = len(self.nodes)
        p_row = []
        p_col = []
        p_data = []
        for row in range(nodes_len):
            for col in range(nodes_len):
                p = self.transition_probability(d, row, col)
                if p != 0:
                    p_row.append(row)
                    p_col.append(col)
                    p_data.append(p)
        p_mx = sp.coo_matrix((p_data, (p_row, p_col)), shape=(nodes_len, nodes_len), dtype=int)
        return p_mx

    def transition_probability(self, d, nodei, nodej):
        """
        Get the transition probability of moving from nodei to nodej
        through the state of nodei and the subscripts of both nodes.

        :param Point d: the destination node
        :param Point nodei: the starting node of transition
        :param Point nodej: the ending node of transition
        :return: the transition probability

        """

        if (nodei == d or check_node_absorbing(self.nodes, self.edges, nodei) == True) \
                and nodei == nodej:
            return 1
        elif (not (nodei == d or check_node_absorbing(self.nodes, self.edges, nodei) == True)) \
                and nodei != nodej:
            return self.prd(d, nodei, nodej)
        else:
            return 0

    def prd(self, d, nodei, nodej):
        """
        Get the turning probability of moving from nodei to nodej
        through the ratio of adding func values of all the trajectories
        on (nodei,nodej) and all the trajectories starting from nodei.

        :param Point d: the destination node
        :param Point nodei: the starting node of transition
        :param Point nodej: the ending node of transition
        :return: the turning probability

        """
        sum_ij, sum_i = 0, 0

        # add func values of all the trajectories on (nodei,nodej)
        if (nodei, nodej) in list(self.edges.keys()):
            for t in self.edges[(nodei, nodej)]:
                sum_ij += self.func(t, d, nodei)

        # add func values of all the trajectories starting from nodei
        col_set = []
        for edge in list(self.edges.keys()):
            if edge[0] == nodei:
                col_set.append(edge)
        for col in col_set:
            for t in self.edges[col]:
                sum_i += self.func(t, d, nodei)

        if sum_i == 0:
            return 0
        return sum_ij / sum_i

    def func(self, traj, d, nodei):
        """
        Estimate the likelihood that a trajectory traj might
        suggest a correct route to d.

        :param Point.trajectory_id traj: the trajectory
        :param Point d: the destination node
        :param Point nodei: the starting node
        :return: the likelihood

        """
        dists = sys.maxsize
        flag = 0
        traj_value = self.trajectories[traj]
        nodei_index_in_traj = \
            self.trajectories[traj].index(nodei)

        # trajectory traj passes node d ,dists = 0
        if d in traj_value[nodei_index_in_traj::]:
            dists = 0
            flag = 1
        # trajectory traj only has one node rather than edge
        elif len(traj_value[nodei_index_in_traj::]) == 1:
            dists = math.pow(
                ((self.nodes[d][0] - self.nodes[traj_value[0]][0]) ** 2
                 + (self.nodes[d][1] - self.nodes[traj_value[0]][1]) ** 2),
                0.5)
            flag = 1
        # trajectory traj has one edge at least
        elif len(traj_value[nodei_index_in_traj::]) >= 2:
            for index in range(nodei_index_in_traj, len(traj_value) - 1):
                new_dist = self.get_dist(d, traj_value[index],traj_value[index + 1])
                if new_dist < dists:
                    dists = new_dist
                    flag = 1

        if flag == 0:
            return 0
        return math.exp(-dists)

    def get_dist(self, d, point1, point2):
        """
        Get the shortest Euclidean/network distance between d and
        the segment from point1 to point2.

        :param Point d: the point outside the segment
        :param Point point1: the endpoint of the segment
        :param Point point2: another endpoint of the segment
        :return: the distance

        """
        d_x = self.nodes[d][0]
        d_y = self.nodes[d][1]
        point1_x = self.nodes[point1][0]
        point1_y = self.nodes[point1][1]
        point2_x = self.nodes[point2][0]
        point2_y = self.nodes[point2][1]
        cross = (point2_x - point1_x) * (d_x - point1_x) \
                + (point2_y - point1_y) * (d_y - point1_y)
        dist2 = (point2_x - point1_x) ** 2 + (point2_y - point1_y) ** 2

        if cross <= 0:
            return math.sqrt((d_x - point1_x) ** 2 + (d_y - point1_y) ** 2)
        if cross >= dist2:
            return math.sqrt((d_x - point2_x) ** 2 + (d_y - point2_y) ** 2)
        r = cross / dist2
        p_x = point1_x + (point2_x - point1_x) * r
        p_y = point1_y + (point2_y - point1_y) * r
        return math.sqrt((d_x - p_x) ** 2 + (d_y - p_y) ** 2)

    def reorganize(self, p, d):
        """
        Reorganize matrix P to canonical form by grouping
        absorbing states into ABS and transient states into TR.

        :param np.array p: matrix P
        :param Point d: the destination node
        :return: matrix Q(TR * TR), matrix S(TR * ABS)

        """
        ABS = []
        TR = []
        for node in list(self.nodes.keys()):
            if node == d or check_node_absorbing(self.nodes, self.edges, node) == True:
                ABS.append(node)
            else:
                TR.append(node)

        p_left_top = p[np.ix_(TR, TR)]
        p_left_bottom = p[np.ix_(ABS, TR)]
        p_right_top = p[np.ix_(TR, ABS)]
        p_right_bottom = p[np.ix_(ABS, ABS)]

        return p_left_top, p_right_top

    def matrix_multiply(self, a, n):
        """
        Calculate n power of matrix A.

        :param np.array a: matrix A
        :param int n: n power
        :return: matrix result

        """
        result = np.identity(a.shape[0])
        for i in range(n):
            result = np.dot(result, a)
        return result

    def step_t(self):
        """
        Set step t as the diameter of the transfer network by Floyd.

        :return: the diameter

        """
        edge_matrix_len = len(self.edges)
        weight = [[sys.maxsize for j in range(edge_matrix_len)]
                  for i in range(edge_matrix_len)]
        for i in range(edge_matrix_len):
            weight[i][i] = 0
            for j in range(edge_matrix_len):
                if self.edges[i][j] != -1 and i != j:
                    weight[i][j] = 1

        for k in range(edge_matrix_len):
            for i in range(edge_matrix_len):
                for j in range(edge_matrix_len):
                    if weight[i][j] > weight[i][k] + weight[k][j]:
                        weight[i][j] = weight[i][k] + weight[k][j]
        return max(max(weight))

    def cal_vector(self, d, p, q):
        """
        Get the column vector V of each node through matrix P and matrix Q.

        :param Point d: the node d
        :param np.array p: matrix P
        :param np.array q: matrix Q
        :return: column vector V

        """
        TR = []
        absorbing_state = [-1 for j in range(len(self.edges))]

        # D=S[*,d]
        for node in list(self.nodes.keys()):
            if not (node == d or check_node_absorbing(self.nodes, self.edges, node) == True):
                TR.append(node)
        D = p[np.ix_(TR, d)]

        # V=D+Q·D+Q^2·D+...+Q^(t-1)·D
        v = np.zeros(D.shape)
        for j in range(0, self.step_t()):
            v = v + np.dot(self.matrix_multiply(q, j), D)
        return v
