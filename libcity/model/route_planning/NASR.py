from queue import PriorityQueue
import torch
import torch.nn as nn
import numpy as np
from geopy import distance
from libcity.model.abstract_model import AbstractModel
from libcity.model.road_representation.GAT import GATLayerImp3
from libcity.data.dataset.nasr_dataset import distance_to_bin


class Attention(nn.Module):
    """
    key: the last hidden state of current trace
    query: all hidden state of current trace
    """

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # hyper parameter
        self.hidden_size = hidden_size
        # linear
        self.out_linear = nn.Linear(in_features=hidden_size, out_features=1, bias=False)
        self.w1_linear = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.w2_linear = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)

    def forward(self, query, key):
        """
        attn_weight = w_out * tanh(w_1 * query + w_2 * key)
        Args:
            query (tensor): shape (batch_size, hidden_size)
            key (tensor): shape (batch_size, trace_len, hidden_size)
        Return:
            attn_hidden (tensor): shape (batch_size, hidden_size)
        """
        query_hidden = torch.relu(self.w1_linear(query).unsqueeze(1))  # shape: (batch_size, 1, hidden_size)
        key_hidden = torch.relu(self.w2_linear(key))  # shape: (batch_size, trace_len, hidden_size)
        attn_weight = torch.tanh(query_hidden + key_hidden)  # shape: (batch_size, trace_len, hidden_size)
        attn_weight = self.out_linear(attn_weight).squeeze(2)  # shape: (batch_size, trace_len)
        attn_weight = torch.softmax(attn_weight, dim=1).unsqueeze(1)  # shape: (batch_size, 1, trace_len)
        return torch.bmm(attn_weight, key).squeeze(1)


class FunctionG(nn.Module):
    """
    The function g network
    """

    def __init__(self, config, data_feature):
        super(FunctionG, self).__init__()
        # the hyper parameter
        self.hidden_size = config['hidden_size']
        self.gru_layer_num = config['gru_layer_num']
        self.uid_emb_size = config['uid_emb_size']
        self.loc_emb_size = config['loc_emb_size']
        self.weekday_emb_size = config['weekday_emb_size']
        self.hour_emb_size = config['hour_emb_size']
        self.device = config['device']
        # calculate emb_feature_size
        self.emb_feature_size = self.uid_emb_size + self.loc_emb_size + self.weekday_emb_size + self.hour_emb_size
        # Embedding layer
        self.uid_emb = nn.Embedding(num_embeddings=data_feature['uid_num'], embedding_dim=self.uid_emb_size)
        self.loc_emb = nn.Embedding(num_embeddings=data_feature['loc_num'], embedding_dim=self.loc_emb_size)
        self.weekday_emb = nn.Embedding(num_embeddings=7, embedding_dim=self.weekday_emb_size)
        self.hour_emb = nn.Embedding(num_embeddings=24, embedding_dim=self.hour_emb_size)
        # GRU layer
        self.gru = nn.GRU(input_size=self.emb_feature_size, hidden_size=self.hidden_size, num_layers=self.gru_layer_num,
                          batch_first=True)
        # intra-trajectory attention
        self.intra_trajectory_attn = Attention(hidden_size=self.hidden_size)
        # inter-trajectory attention
        self.inter_trajectory_attn = Attention(hidden_size=self.hidden_size)
        # score linear
        self.score_lr = nn.Linear(in_features=self.hidden_size, out_features=1, bias=False)

    def forward(self, uid, current_trace, history_trace, candidate_set, history_trace_hidden=None):
        """
        The FunctionG cannot parallel calculate, which means the batch_size may not work in this function.
        The FunctionG may cost a lot of time.

        input one trace per run.

        Args:
            uid (1): the uid of this trace.
            current_trace (list): list of current_trace tensor (step_len, 3),
                                  the feature dimension is [location id, weekday id, hour id]
                                  the len of list is trace_len - 1
                                  Examples: l0 -> ld, l1-> ld, ....
            history_trace (list): history_len list of history_trace. the list size will be
                                  (history_len, trace_len, 3). no padding.
            candidate_set (list): list of candidate set tensor (candidate_size). no padding.
            history_trace_hidden (tensor): already encoded history trace hidden. this param will be used in prediction.

        Returns:
            candidate_g_prob (list): list of each candidate prob in each step of a trace.
            moving_state (list): list of the hidden tensor of each candidate_set's moving state.
        """
        candidate_g_prob = []
        moving_state = []
        # calculate function g for each step in trace
        uid = uid.unsqueeze(0)  # (1)
        if history_trace_hidden is None:
            # encode history trace
            history_trace_hidden_list = []
            for history_j in range(len(history_trace)):
                # (1, trace_len, 3)
                history_trace_j = torch.LongTensor(history_trace[history_j]).to(self.device).unsqueeze(0)
                history_trace_j_hidden = self.encode_trace(uid, history_trace_j)  # (1, hidden_size)
                history_trace_hidden_list.append(history_trace_j_hidden)
            # form list to tensor
            # (1, history_len, hidden_size)
            history_trace_hidden = torch.cat(history_trace_hidden_list, dim=0).unsqueeze(0)
        for step_i in range(len(current_trace)):
            # calculate each candidate's hidden
            # append each candidate to current_trace
            current_trace_i = current_trace[step_i]  # (trace_len, 3)
            current_trace_i = torch.LongTensor(current_trace_i).to(self.device)
            candidate_set_i = candidate_set[step_i]  # (candidate_size)
            candidate_set_i = torch.LongTensor(candidate_set_i).to(self.device)
            # (candidate_size, trace_len, 3)
            current_trace_i = current_trace_i.unsqueeze(0).repeat(candidate_set_i.shape[0], 1, 1)
            candidate_set_i_time = current_trace_i[:, -1, 1:]  # (candidate_size, 2)
            # (candidate_size, 3)
            candidate_set_i = torch.cat([candidate_set_i.unsqueeze(1), candidate_set_i_time], dim=1)
            # (candidate_size, trace_len + 1, 3)
            candidate_trace_i = torch.cat([current_trace_i, candidate_set_i.unsqueeze(1)], dim=1)
            # (candidate_size, hidden_size)
            candidate_trace_hidden = self.encode_trace(uid.repeat(candidate_trace_i.shape[0]), candidate_trace_i)
            # inter-trajectory attn for each candidate trace
            # (candidate_size, hidden_size)
            inter_attn_candidate_trace = []
            for candidate_j in range(candidate_trace_hidden.shape[0]):
                inter_attn_candidate_j = self.inter_trajectory_attn(query=candidate_trace_hidden[
                                                                    candidate_j].unsqueeze(0),
                                                                    key=history_trace_hidden)
                inter_attn_candidate_trace.append(inter_attn_candidate_j)
            # cat as tensor
            # (candidate_size, hidden_size)
            inter_attn_candidate_trace = torch.cat(inter_attn_candidate_trace, dim=0)
            # (candidate_size)
            candidate_score = self.score_lr(inter_attn_candidate_trace).squeeze(1)
            candidate_g_prob.append(torch.softmax(candidate_score, dim=0))
            moving_state.append(inter_attn_candidate_trace)
        return candidate_g_prob, moving_state

    def encode_trace(self, uid, trace):
        """
        encode a trace with gru and intra-trajectory attention

        Args:
            uid (batch_size): uid tensor
            trace (batch_size, trace_len, 3): the tensor of a trace. no padding!

        Returns:
            trace_hidden_state: the attn hidden_state of trace, (batch_size, hidden_size)
        """
        # embed trace
        uid_emb = self.uid_emb(uid)  # (batch_size, uid_emb_size)
        trace_loc_emb = self.loc_emb(trace[:, :, 0])  # (batch_size, trace_len, loc_emb_size)
        trace_weekday_emb = self.weekday_emb(trace[:, :, 1])
        trace_hour_emb = self.hour_emb(trace[:, :, 2])
        # repeat uid_emb to trace_len
        trace_len = trace_loc_emb.shape[1]
        uid_emb = uid_emb.unsqueeze(1).repeat(1, trace_len, 1)  # (batch_size, trace_len, uid_emb_size)
        # cat all embedding
        # (batch_size, trace_len, emb_feature_size)
        trace_emb = torch.cat([uid_emb, trace_loc_emb, trace_weekday_emb, trace_hour_emb], dim=2)
        trace_hidden_state, hn = self.gru(trace_emb)
        intra_trace_hidden = self.intra_trajectory_attn(query=trace_hidden_state[:, -1, :], key=trace_hidden_state)
        return intra_trace_hidden

    def calculate_loss(self, batch):
        """
        calculate function g loss, which is min negative log of target prob

        Args:
            batch (Batch): a batch of input

        Returns:
            loss (tensor): the sum of negative log prob of target candidate
        """
        uid = batch['uid']  # (batch_size)
        current_trace = batch['current_trace']  # list
        history_trace = batch['history_trace']  # list
        candidate_set = batch['candidate_set']  # list
        target = batch['target']  # list
        target_g_prob = []
        for iter_i in range(uid.shape[0]):
            uid_i = uid[iter_i]
            candidate_g_prob, _ = self.forward(uid_i, current_trace[iter_i], history_trace[iter_i],
                                               candidate_set[iter_i])
            # select target's prob
            for target_i in range(len(target[iter_i])):
                target_index = target[iter_i][target_i]
                target_g_prob.append(candidate_g_prob[target_i][target_index].unsqueeze(0))
        # (batch_size * trace_len)
        target_g_prob = torch.cat(target_g_prob, dim=0)
        loss = torch.sum(-torch.log(target_g_prob))
        return loss


class ContextAwareGAT(GATLayerImp3):
    """
    make some modification on existing GAT models.
    introduce moving state and distance to attn score function
    """

    def __init__(self, config, data_feature):
        # model param
        self.gat_out_feature = config['gat_out_feature']
        self.num_of_heads = config['gat_num_of_head']
        self.device = config['device']
        self.dropout_p = config['dropout_p']
        self.hidden_size = config['hidden_size']
        self.distance_emb_size = config['distance_emb_size']
        # data feature
        self.node_features = data_feature['node_features']
        self.num_in_features = self.node_features.shape[1]
        self.adj_mx = data_feature['adj_mx']
        self.edge_index = torch.LongTensor([self.adj_mx.row.tolist(), self.adj_mx.col.tolist()]).to(self.device)
        self.distance_bins = data_feature['distance_bins']
        super(ContextAwareGAT, self).__init__(self.num_in_features, self.gat_out_feature, self.num_of_heads,
                                              self.device, dropout_prob=self.dropout_p)
        # moving state W and distance W
        self.moving_state_w = nn.Linear(in_features=self.hidden_size, out_features=self.num_of_heads, bias=False)
        self.distance_emb = nn.Embedding(num_embeddings=self.distance_bins, embedding_dim=self.distance_emb_size)
        self.distance_w = nn.Linear(in_features=self.distance_emb_size, out_features=self.num_of_heads, bias=False)

    def forward(self, data):
        """
        rewrite forward function.
        Args:
            data: moving_state of current trajectory (hidden_size)

        Returns:

        """
        #
        # Step 1: Linear Projection + regularization
        #
        in_nodes_features = torch.FloatTensor(self.node_features).to(self.device)
        edge_index = self.edge_index.clone()
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'
        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features)
        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well
        #
        # Step 2: Edge attention calculation
        #
        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
        # (1, NH)
        scores_moving_state = self.moving_state_w(data.unsqueeze(0))
        # get edge weight (distance) from adj_mx
        # (E, 1)
        edge_weight = torch.tensor(self.adj_mx.data).unsqueeze(1).to(self.device)
        # edge embedding
        # (E, distance_emb_size)
        edge_weight = self.distance_emb(edge_weight)
        # (E, NH)
        edge_score = self.distance_w(edge_weight).squeeze(1)
        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target,
                                                                                           nodes_features_proj,
                                                                                           edge_index)
        # (E, NH)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted + scores_moving_state + edge_score)
        # shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim],
                                                              num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)
        #
        # Step 3: Neighborhood aggregation
        #
        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index,
                                                      in_nodes_features, num_of_nodes)
        #
        # Step 4: Residual/skip connections, concat and bias
        #
        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return out_nodes_features


class FunctionH(nn.Module):
    """
    the h function of NASR.
    """

    def __init__(self, config, data_feature):
        super(FunctionH, self).__init__()
        # model param
        self.device = config['device']
        self.dropout_p = config['dropout_p']
        self.hidden_size = config['hidden_size']
        self.distance_emb_size = config['distance_emb_size']
        self.distance_bins = data_feature['distance_bins']
        self.gat = ContextAwareGAT(config, data_feature)
        self.distance_emb = nn.Embedding(num_embeddings=self.distance_bins, embedding_dim=self.distance_emb_size)
        self.input_size = self.gat.num_out_features * self.gat.num_of_heads * 2 + self.hidden_size
        self.input_size += self.distance_emb_size
        # two layer linear layer -- MLP
        self.fc1 = nn.Linear(in_features=self.input_size, out_features=self.hidden_size)
        self.fc2 = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, moving_state, li, ld, distance):
        """
        Because each candidate trace will need a different gat, the function h cannot be parallel.
        Args:
            moving_state (hidden_size):
            li (1): candidate location
            ld (1): destination
            distance (1): the discrete distance value

        Returns:
            fc2_out (1): the heuristics cost of function h
        """
        node_emb_features = self.gat(moving_state)
        li_emb = node_emb_features[li].squeeze(0)
        ld_emb = node_emb_features[ld].squeeze(0)
        distance_emb = self.distance_emb(distance).squeeze(0)
        input_feature = torch.cat([moving_state, li_emb, ld_emb, distance_emb], dim=0)
        fc1_out = torch.relu(self.fc1(input_feature))
        fc2_out = self.fc2(fc1_out)
        return fc2_out


class SearchNode(object):
    """
    The search node will save the context information of the search branch of the node
    """

    def __init__(self, trace, rid, date_time, log_prob):
        """
        Args:
            trace: the trace of the search node branch, (trace_len, 3)
            rid: the search node's rid
            date_time: The time object of this node is the number of minutes from zero.
                       This time will be used to calculate the next node's time.
            log_prob:  the log probability log(P(l_s -> l_i | l_s, l_d, t_s))
        """
        self.trace = trace
        self.rid = rid
        self.date_time = date_time
        self.log_prob = log_prob


class NASR(AbstractModel):
    """
    Empowering A∗ Search Algorithms with Neural Networks for Personalized Route Recommendation
    paper link: https://dl.acm.org/doi/pdf/10.1145/3292500.3330824
    """

    def __init__(self, config, data_feature):
        super(NASR, self).__init__(config, data_feature)
        self.time_diff_n = config['time_diff_n']
        self.discount_rate = config['discount_rate']
        self.road_gps = data_feature['road_gps']
        self.device = config['device']
        self.adjacent_list = data_feature['adjacent_list']
        self.function_g = FunctionG(config, data_feature)
        self.function_h = FunctionH(config, data_feature)

    def forward(self, uid, current_trace, history_hidden, candidate_set, candidate_distance, des):
        """

        Args:
            uid (1): the uid of this trace.
            current_trace (tensor): current_trace tensor (trace_len, 3),
                                  the feature dimension is [location id, weekday id, hour id]
            history_hidden (tensor): the encoded history hidden by function g.
            candidate_set (tensor): candidate set tensor (candidate_size). no padding.
            candidate_distance (tensor): the distance between candidate and destination.
            des (tensor): the destination rid.

        Returns:
            f_cost: each candidate's f_cost
        """
        # calculate g prob
        g_prob, moving_state = self.function_g.forward(uid=uid.squeeze(0), current_trace=[current_trace],
                                                       candidate_set=[candidate_set],
                                                       history_trace=[], history_trace_hidden=history_hidden)
        # This step needs to be done because of the function_g forward interface
        g_prob = g_prob[0]
        moving_state = moving_state[0]
        # calculate h cost for each
        h_cost = torch.zeros(moving_state.shape[0]).to(self.device)
        for i in range(moving_state.shape[0]):
            h_cost[i] = self.function_h.forward(moving_state=moving_state[i], li=candidate_set[i],
                                                ld=des, distance=candidate_distance[i])
        g_cost = -torch.log(g_prob)
        f_cost = g_cost + h_cost
        return f_cost

    def predict(self, batch):
        """
        Args:
            batch (Batch): a batch of queries. A query is composed of (l_s, weekday_s, hour_s, l_d, uid, history_trace)

        Returns:
            generate_trace (list): list of NASR recommend traces, only contains road list and uid. (uid, generate_trace)
        """
        # use A* search
        query = batch['query']
        generate_trace = []
        for i in range(len(query)):
            # parse query
            query_i = query[i]
            des_rid = query_i[3]
            uid = torch.LongTensor([query_i[4]]).to(self.device)
            open_set = PriorityQueue()  # unvisited node set
            close_set = set()  # visited node set
            rid2node = {}  # the dict key is rid, and value is corresponding search node
            des_center_gps = self.road_gps[des_rid]
            des = torch.LongTensor([des_rid]).to(self.device)
            # put l_s into open_set
            start_node = SearchNode(trace=[[query_i[0], query_i[1], query_i[2]]], rid=query_i[0],
                                    date_time=query_i[2] * 60, log_prob=0.0)
            open_set.put((start_node.log_prob, start_node))
            rid2node[query_i[0]] = start_node
            max_search_step = 500
            step = 0
            # encode history
            history_trace_i = query_i[5]
            history_trace_hidden_list = []
            for history_j in range(len(history_trace_i)):
                # (1, trace_len, 3)
                history_trace_j = torch.LongTensor(history_trace_i[history_j]).to(self.device).unsqueeze(0)
                history_trace_j_hidden = self.function_g.encode_trace(uid, history_trace_j)  # (1, hidden_size)
                history_trace_hidden_list.append(history_trace_j_hidden)
            # form list to tensor
            # (1, history_len, hidden_size)
            history_trace_hidden = torch.cat(history_trace_hidden_list, dim=0).unsqueeze(0)
            # find flag
            best_trace = []
            best_score = 0
            default_len = 15
            while not open_set.empty() and step < max_search_step:
                # get min f cost node to search
                cost, now_node = open_set.get()
                # put now node in close_set
                now_rid = now_node.rid
                if now_rid in close_set:
                    # now_node is an old node
                    # now_rid has been visited
                    continue
                close_set.add(now_rid)
                # del now_id from rid2node, because now node has been visited
                rid2node.pop(now_rid, None)
                if now_rid == des_rid:
                    # find destination
                    trace_i = now_node.trace
                    best_trace = [x[0] for x in trace_i]
                    # finish search for query_i
                    break
                else:
                    # search now's adjacent rid
                    if now_rid in self.adjacent_list:
                        candidate_set = self.adjacent_list[now_rid]
                        if len(candidate_set) != 0:
                            candidate_dis = []
                            for c in candidate_set:
                                candidate_gps = self.road_gps[c]
                                d = distance.distance((des_center_gps[1], des_center_gps[0]),
                                                      (candidate_gps[1], candidate_gps[0])).kilometers * 1000
                                candidate_dis.append(distance_to_bin(d))
                            trace = torch.LongTensor(now_node.trace).to(self.device)
                            candidate = torch.LongTensor(candidate_set).to(self.device)
                            dis = torch.LongTensor(candidate_dis).to(self.device)
                            # 根据模型计算 f 函数值和 g 函数值
                            with torch.no_grad():
                                candidate_f_cost = self.forward(uid=uid, current_trace=trace,
                                                                history_hidden=history_trace_hidden,
                                                                candidate_set=candidate, candidate_distance=dis,
                                                                des=des)
                            # update each candidate's node
                            # calculate next datetime
                            now_datetime = now_node.date_time
                            now_weekday = now_node.trace[-1][1]
                            # add a fix time step
                            next_datetime = now_datetime + 5
                            if next_datetime >= 1440:
                                # a new day
                                # weekday from 0 to 6
                                now_weekday = (now_weekday + 1) % 7
                                next_datetime = next_datetime % 1440
                            next_hourindex = next_datetime // 60
                            for index, c in enumerate(candidate_set):
                                candidate_log_prob = now_node.log_prob + candidate_f_cost[index].item()
                                candidate_score = np.exp(-candidate_log_prob)
                                if c not in rid2node and c not in close_set:
                                    # c has not been reached
                                    candidate_node = SearchNode(trace=now_node.trace + [[c, now_weekday,
                                                                                        next_hourindex]],
                                                                rid=c, date_time=next_datetime,
                                                                log_prob=candidate_log_prob)
                                    # put c in open_set
                                    open_set.put((candidate_log_prob, candidate_node))
                                    rid2node[c] = candidate_node
                                    if len(candidate_node.trace) == default_len and candidate_score > best_score:
                                        # give the default recommended trace
                                        best_trace = candidate_node.trace
                                        best_score = candidate_score
                                elif c in rid2node and rid2node[c].log_prob > candidate_log_prob:
                                    # update search node
                                    rid2node[c].log_prob = candidate_log_prob
                                    rid2node[c].trace = now_node.trace + [[c, now_weekday, next_hourindex]]
                                    rid2node[c].date_time = next_datetime
                                    # there seems no way to update c in open_set, so just put c in open_set.
                                    # the higher priority c will be searched first.
                                    # so this is still work.
                                    open_set.put((candidate_log_prob, rid2node[c]))
                                    if len(rid2node[c].trace) == default_len and candidate_score > best_score:
                                        # give the default recommended trace
                                        best_trace = rid2node[c].trace
                                        best_score = candidate_score
                step += 1
            generate_trace.append((uid.item(), best_trace))
        return generate_trace

    def calculate_loss(self, batch):
        """
        the dataset should ensure one trajectory one batch, otherwise the training
        process will be wrong.
        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: return training loss
        """
        loss_mode = batch['loss_mode']
        if loss_mode == 'g_loss':
            return self.function_g.calculate_loss(batch)
        elif loss_mode == 'h_loss':
            # first calculate g_prob
            uid = batch['uid']
            current_trace = batch['current_trace']
            history_trace = batch['history_trace']
            candidate_set = batch['candidate_set']
            target = batch['target']
            candidate_distance = batch['distance']  # list
            for iter_i in range(uid.shape[0]):
                # one trace one calculate
                uid_i = uid[iter_i]
                target_g_prob = []
                target_h_cost = []
                with torch.no_grad():
                    # when train h, g should be ground truth.
                    candidate_g_prob, moving_state = self.function_g.forward(uid_i, current_trace[iter_i],
                                                                             history_trace[iter_i],
                                                                             candidate_set[iter_i])
                # select target's prob and moving state
                des_index = target[iter_i][-1]
                ld = torch.LongTensor([candidate_set[iter_i][-1][des_index]]).to(self.device)
                for target_i in range(len(target[iter_i])):
                    # the last point do not need to learn
                    target_index = target[iter_i][target_i]
                    li = torch.LongTensor([candidate_set[iter_i][target_i][target_index]]).to(self.device)  # (1)
                    distance_li_ld = torch.LongTensor([candidate_distance[iter_i][target_i][target_index]]).to(
                        self.device)  # (1)
                    target_i_h_cost = self.function_h.forward(moving_state=moving_state[target_i][target_index], li=li,
                                                              ld=ld, distance=distance_li_ld)
                    target_g_prob.append(candidate_g_prob[target_i][target_index].unsqueeze(0))
                    target_h_cost.append(target_i_h_cost)
                # base on target_g_prob calculate function h's target
                target_h_cost = torch.cat(target_h_cost, dim=0)  # (trace_len)
                target_g_prob = torch.cat(target_g_prob, dim=0)  # (trace_len)
                target_g_cost = - torch.log(target_g_prob)
                # set n = min(time_diff_n, trace_len - 1)
                n = min(self.time_diff_n, target_h_cost.shape[0] - 1)
                pre_h_n = target_h_cost[:-n]
                after_h_n = target_h_cost[n:]
                discount_g = torch.zeros((pre_h_n.shape[0])).to(self.device)
                for i in range(pre_h_n.shape[0]):
                    c_lj = target_g_cost[i + 1].clone()
                    for j in range(i + 2, i + n):
                        c_lj += self.discount_rate ** (j - i - 1) * target_g_cost[j]
                    discount_g[i] = c_lj
                loss = torch.sum(torch.square(pre_h_n - self.discount_rate ** n * after_h_n - discount_g))
                return loss
        else:
            raise NotImplementedError('the loss mode {} is not supported by NASR'.format(loss_mode))
