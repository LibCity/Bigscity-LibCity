import os
import numpy as np

from libcity.data.dataset import TrafficStatePointDataset

from collections import defaultdict


# This class represents an directed graph
# using adjacency list representation
class Graph:

    def __init__(self, vertices):
        # No. of vertices
        self.V = vertices

        # default dictionary to store graph
        self.graph = defaultdict(list)

        self.res = []

        # time is used to find discovery times
        self.Time = 0

        # Count is number of biconnected components
        self.count = 0

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    '''A recursive function that finds and prints strongly connected
    components using DFS traversal
    u --> The vertex to be visited next
    disc[] --> Stores discovery times of visited vertices
    low[] -- >> earliest visited vertex (the vertex with minimum
               discovery time) that can be reached from subtree
               rooted with current vertex
    st -- >> To store visited edges'''

    def BCCUtil(self, u, parent, low, disc, st):

        # Count of children in current node
        children = 0

        # Initialize discovery time and low value
        disc[u] = self.Time
        low[u] = self.Time
        self.Time += 1

        # Recur for all the vertices adjacent to this vertex
        for v in self.graph[u]:
            # If v is not visited yet, then make it a child of u
            # in DFS tree and recur for it
            if disc[v] == -1:
                parent[v] = u
                children += 1
                st.append((u, v))  # store the edge in stack
                # st.append([u, v])
                self.BCCUtil(v, parent, low, disc, st)

                # Check if the subtree rooted with v has a connection to
                # one of the ancestors of u
                # Case 1 -- per Strongly Connected Components Article
                low[u] = min(low[u], low[v])

                # If u is an articulation point, pop
                # all edges from stack till (u, v)
                if parent[u] == -1 and children > 1 or parent[u] != -1 and low[v] >= disc[u]:
                    self.count += 1  # increment count
                    w = -1
                    tmp = []
                    while w != (u, v):
                        w = st.pop()
                        tmp.append(w)
                    self.res.append(tmp)


            elif v != parent[u] and low[u] > disc[v]:
                '''Update low value of 'u' only of 'v' is still in stack
                (i.e. it's a back edge, not cross edge).
                Case 2
                -- per Strongly Connected Components Article'''

                low[u] = min(low[u], disc[v])

                st.append((u, v))
                # st.append((u, v))

    # The function to do DFS traversal.
    # It uses recursive BCCUtil()
    def BCC(self):

        # Initialize disc and low, and parent arrays
        disc = [-1] * (self.V)
        low = [-1] * (self.V)
        parent = [-1] * (self.V)
        st = []

        # Call the recursive helper function to
        # find articulation points
        # in DFS tree rooted with vertex 'i'
        for i in range(self.V):
            if disc[i] == -1:
                self.BCCUtil(i, parent, low, disc, st)

            # If stack is not empty, pop all edges from stack
            if st:
                self.count = self.count + 1
                tmp=[]
                while st:
                    w = st.pop()
                    tmp.append(w)
                self.res.append(tmp)


class HIESTDataset(TrafficStatePointDataset):
    def _load_rel(self):
        """
        加载.rel文件，格式[rel_id, type, origin_id, destination_id, properties(若干列)]

        Returns:
            np.ndarray: self.adj_mx, N*N的邻接矩阵
        """
        super()._load_rel()

        original = self.adj_mx
        g1 = Graph(len(original))  # 325

        for i in range(len(original)):
            for j in range(len(original)):
                if original[i][j] > 0.8:  # threshhold
                    g1.addEdge(i, j)
        g1.BCC()

        map = []
        for lists in g1.res:
            tmp = ()
            for sets in lists:
                # print(type(set))
                tmp += sets
            map.append(list(set(tmp)))

        indices = ()
        for i in map:
            indices += tuple(i)

        base = [i for i in range(len(original[0]))]
        assigned = set(indices)
        notMap = set(base) - assigned
        notMap = list(notMap)

        Mor = np.zeros((len(original), (len(map) + len(notMap))))
        for i, nodes in enumerate(map):
            for j in range(len(nodes)):
                Mor[nodes[j]][i] = 1

        for i, nodes in enumerate(notMap):
            Mor[nodes][i + len(map)] = 1

        Mor = Mor / Mor.sum(axis=0)
        self.Mor_mx = Mor
        self._logger.info('generate mapping matrix Mor_mx: {}'.format(self.Mor_mx.shape))
        self.regional_nodes = len(self.Mor_mx[0])

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"scaler": self.scaler, "adj_mx": self.adj_mx ,"ext_dim": self.ext_dim,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,"regional_nodes": self.regional_nodes,
                "Mor_mx": self.Mor_mx,"output_dim": self.output_dim, "num_batches": self.num_batches}