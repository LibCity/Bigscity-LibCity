import math
import sys
import scipy as sp


def check_node_absorbing(nodes, edges, node):
    node_set = [(node, t) for t in list(nodes.keys())]
    for node in node_set:
        if node in list(edges.keys()):
            return False
    return True


def check_coomatrix_value(i, j, row, col, data):
    for n in range(len(row)):
        if row[n] == i and col[n] == j:
            return data[n]
        else:
            return 0


def SparseMatrixAdd(A, B):
    row = []
    col = []
    data = []
    data_set = set()
    for i in range(A.data):
        row.append(A.row[i])
        col.append(A.col[i])
        data_set.add((A.row[i], A.col[i]))
        data.append(A.data[i])
    for i in range(B.data):
        if (B.row[i], B.col[i]) not in data_set:
            row.append(B.row[i])
            col.append(B.col[i])
            data.append(A.data[i])
        else:
            m = zip(row, col)
            n = m.index((B.row[i], B.col[i]))
            data[n] += B.data[i]
    mx = sp.coo_matrix((data, (row, col)), shape=(len(A.row), len(B.col)), dtype=int)
    return mx


def SparseMatrixMultiply(A, B):
    row = []
    col = []
    data = []
    data_set = set()
    data_dict = dict()
    for i in range(A.col):
        for j in range(B.row):
            if A.col[i] == B.row[j]:
                if (A.row[i], B.col[j]) not in data_set:
                    data_set.add((A.row[i], B.col[j]))
                    data_dict[(A.row[i], B.col[j])] = A.data[i] * B.data[j]
                else:
                    data_dict[(A.row[i], B.col[j])] += A.data[i] * B.data[j]
    for key in data_dict.keys():
        row.append(key[0])
        col.append(key[1])
        data.append(data_dict[key])
    mx = sp.coo_matrix((data, (row, col)), shape=(len(A.row), len(B.col)), dtype=int)
    return mx


class TransferProbability:
    def __init__(self, nodes, edges, trajectories):
        self.nodes = nodes
        self.edges = edges
        self.trajectories = trajectories
        self.vector = None
        self.p = None
        self.q = None
        self.s = None
        self.derive()

    def derive(self):
        """
        Derive matrix P ,matrix Q ,matrix S and column vector V of each node.

        """
        self.vector = []
        for node in self.nodes:
            self.p = self.create_transition_matrix(node)
            self.q = self.reorganize(self.p, node)
            self.vector[node] = self.cal_vector(node, self.p, self.q)

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
        if (nodei, nodej) in self.edges:
            for t in self.edges[(nodei, nodej)]:
                sum_ij += self.func(t, d, nodei)

        # add func values of all the trajectories starting from nodei
        col_set = []
        for edge in self.edges:
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
                new_dist = self.get_dist(d, traj_value[index], traj_value[index + 1])
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

        m = zip(p.row, p.col)
        row = []
        col = []
        data = []
        for i in range(len(TR)):
            for j in range(len(TR)):
                if (TR[i], TR[j]) in m:
                    n = m.index((TR[i], TR[j]))
                    row.append(TR[i])
                    col.append(TR[j])
                    data.append(p.data[n])
                else:
                    continue
        p_left_top = sp.coo_matrix((data, (row, col)), shape=(len(TR), len(TR)), dtype=int)
        return p_left_top

    def matrix_multiply(self, a, n):
        """
        Calculate n power of matrix A.

        :param np.array a: matrix A
        :param int n: n power
        :return: matrix result

        """
        result = sp.coo_matrix(
            ([1 for i in range(len(a.data))], ([i for i in range(len(a.data))], [i for i in range(len(a.data))])),
            shape=(len(a.row), len(a.col)), dtype=int)
        for i in range(n):
            result = SparseMatrixMultiply(result, a)
        return result

    def step_t(self):
        """
        Set step t as the diameter of the transfer network by Floyd.

        :return: the diameter

        """
        edge_matrix_len = len(self.edges)
        weight_row = []
        weight_col = []
        weight_data = []
        weight_set = ()
        for i in range(edge_matrix_len):
            for j in range(edge_matrix_len):
                if (i, j) in list(self.edges.keys()) and (i, j) not in weight_set and i != j:
                    weight_set.add((i, j))
                    weight_row.append(i)
                    weight_col.append(j)
                    weight_data.append(1)
        for k in range(edge_matrix_len):
            for i in range(edge_matrix_len):
                for j in range(edge_matrix_len):
                    if check_coomatrix_value(i, j, weight_row, weight_col, weight_data) \
                            > check_coomatrix_value(i, k, weight_row, weight_col, weight_data) + \
                            check_coomatrix_value(k, j, weight_row, weight_col, weight_data):
                        if (i, j) not in weight_set:
                            weight_set.add((i, j))
                            weight_row.append(i)
                            weight_col.append(j)
                            weight_data.append(check_coomatrix_value(i, k, weight_row, weight_col, weight_data) + \
                                               check_coomatrix_value(k, j, weight_row, weight_col, weight_data))
                        else:
                            for n in range(len(weight_row)):
                                if weight_row[n] == i and weight_col[n] == j:
                                    weight_data[n] = check_coomatrix_value(i, k, weight_row, weight_col, weight_data) + \
                                                     check_coomatrix_value(k, j, weight_row, weight_col, weight_data)
        return max(weight_data)

    def cal_vector(self, d, p, q):
        """
        Get the column vector V of each node through matrix P and matrix Q.

        :param Point d: the node d
        :param np.array p: matrix P
        :param np.array q: matrix Q
        :return: column vector V

        """
        TR = []

        # D=S[*,d]
        for node in self.nodes:
            if not (node == d or check_node_absorbing(self.nodes, self.edges, node) == True):
                TR.append(node)

        m = zip(p.row, p.col)
        row = []
        col = []
        data = []
        for i in range(len(TR)):
            if (TR[i], d) in m:
                n = m.index((TR[i], d))
                row.append(TR[i])
                col.append(d)
                data.append(p.data[n])
            else:
                continue
        D = sp.coo_matrix((data, (row, col)), shape=(len(TR), 1), dtype=int)

        # V=D+Q·D+Q^2·D+...+Q^(t-1)·D
        v = sp.coo_matrix(([], ([], [])), shape=(len(TR), 1), dtype=int)
        for j in range(0, self.step_t()):
            v = SparseMatrixAdd(v, SparseMatrixMultiply(self.matrix_multiply(q, j), D))
        return v
