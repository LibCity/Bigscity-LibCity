from sortedcontainers import SortedList

from libcity.model.abstract_machine_learning_model import AbstractMachineLearningModel
from libcity.utils.transfer_probability import TransferProbability

class MPR(AbstractMachineLearningModel):
    def __init__(self, config, data_feature):
        """

        Args:
            config (ConfigParser): the dict of config
            data_feature (dict): the dict of data feature passed from Dataset.
        """
        super().__init__(config, data_feature)
        self.data_feature = data_feature
        self.nodes = self.data_feature.get('road_gps')
        self.road_num = self.data_feature.get('loc_num')
        self.traj_num = self.data_feature.get('traj_num')
        self.adj_mx = self.data_feature.get('adj_mx')

        self.edges = None
        self.transferpro_mx = None

    def predict(self, batch):
        """
        Args:
            batch (Batch): a batch of input (start, end)
            start node: batch[0]
            end node: batch[1]

        Returns:
            result (object) : predict result of this batch
        """
        # attribute L records the maximum ρ() value of the route from the
        # start node s to node ni
        L_dict = dict()
        for node in self.nodes:
            L_dict[node] = 0
        L_dict[batch[0]] = 1
        # create priority queue sorted by node's attribute L
        priority_queue = SortedList(key=lambda x: L_dict[x])
        priority_queue.add(batch[0])
        # save nodes that have been processed
        scanned_nodes = set()

        while priority_queue:
            # extract node u with maximum L value
            u = priority_queue.pop()

            # find destination node and return the most popular route
            parent_index = dict()
            if u == batch[1]:
                route = [batch[1]]
                node = batch[1]
                while node in parent_index.keys():
                    route.append(parent_index[node])
                    node = parent_index[node]
                return route
            # check node u's adjecent node
            adjacent_node_indexes = [self.transferpro_mx.col[index] for index in range(len(self.transferpro_mx.row))
                                     if self.transferpro_mx.row[index] == u]
            for v in adjacent_node_indexes:
                if v == batch[1]:
                    popularity = 1
                else:
                    vector = self.transferpro_mx.vector[batch[1]]
                    if v not in vector.row:
                        popularity = 0
                    else:
                        if v < batch[1]:
                            n = vector.row.index(v)
                            popularity = vector.data[n]
                        else:
                            n = vector.row.index(v - 1)
                            popularity = vector.data[n]
                new_L = L_dict(u) * popularity
                if L_dict[v] < new_L:
                    # modify node v's attribute L
                    priority_queue.discard(v)
                    L_dict[v] = new_L
                    parent_index[v] = u
                    if v not in scanned_nodes:
                        priority_queue.add(v)
            scanned_nodes.add(u)

    def fit(self, batch):
        """
        use train data to fit the machine learning model

        Args:
            batch (Batch): a batch of input. Generally speaking, the train data of machine learning method
            does not need to be divided into batches, just use Batch to pass in all the data directly.

        Returns:
            return None
        """
        trajectories = dict()
        traj_set = set()
        self.edges = {}
        for traj_id, trace in enumerate(batch):
            current_trace = batch[traj_id]
            trajectories[traj_id] = current_trace
            for step in range(len(current_trace) - 1):
                f_id = current_trace[step]
                t_id = current_trace[step + 1]
                if (f_id, t_id) not in traj_set:
                    traj_set.add((f_id, t_id))
                    self.edges[(f_id, t_id)] = [traj_id]
                else:
                    if traj_id not in self.edges[(f_id, t_id)]:
                        self.edges[(f_id, t_id)].append(traj_id)
                    else:
                        pass
        self.transferpro_mx = TransferProbability(self.nodes, self.edges, trajectories)
