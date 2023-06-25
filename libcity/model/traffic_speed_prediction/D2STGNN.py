import math
import scipy.sparse as sp
import numpy as np
from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def remove_nan_inf(tensor):
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor


def transition_matrix(adj):
    r"""
    Description:
    -----------
    Calculate the transition matrix `P` proposed in DCRNN and Graph WaveNet.
    P = D^{-1}A = A/rowsum(A)

    Parameters:
    -----------
    adj: np.ndarray
        Adjacent matrix A

    Returns:
    -----------
    P:np.matrix
        Renormalized message passing adj in `GCN`.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    # P = d_mat.dot(adj)
    P = d_mat.dot(adj).astype(np.float32).todense()
    return P


class EstimationGate(nn.Module):
    """The estimation gate module."""

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim, config=None):
        super().__init__()
        self.seq_length = config.get("input_window", 12)
        time_dim = 0
        if config.get("add_time_in_day", True):
            time_dim += time_emb_dim
        if config.get("add_day_in_week", True):
            time_dim += time_emb_dim
        self.fully_connected_layer_1 = nn.Linear(2 * node_emb_dim + time_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fully_connected_layer_2 = nn.Linear(hidden_dim, 1)

    def forward(self, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data):
        """Generate gate value in (0, 1) based on current node and time step embeddings to roughly estimating the proportion of the two hidden time series."""

        batch_size = history_data.shape[0]
        embedding_feat = [node_embedding_u.unsqueeze(0).unsqueeze(0).expand(batch_size, self.seq_length,  -1, -1), node_embedding_d.unsqueeze(0).unsqueeze(0).expand(batch_size, self.seq_length,  -1, -1)]
        if day_in_week_feat is not None:
            embedding_feat.insert(0, day_in_week_feat)
        if time_in_day_feat is not None:
            embedding_feat.insert(0, time_in_day_feat)
        estimation_gate_feat = torch.cat(embedding_feat, dim=-1)
        hidden = self.fully_connected_layer_1(estimation_gate_feat)
        hidden = self.activation(hidden)
        # activation
        estimation_gate = torch.sigmoid(self.fully_connected_layer_2(hidden))[:, -history_data.shape[1]:, :, :]
        history_data = history_data * estimation_gate
        return history_data


class STLocalizedConv(nn.Module):
    def __init__(self, hidden_dim, pre_defined_graph=None, use_pre=None, dy_graph=None, sta_graph=None, config=None):
        super().__init__()
        # gated temporal conv
        self.k_s = config.get('k_s', 2)
        self.k_t = config.get('k_t', 3)
        self.dropout = config.get('dropout', 0.1)
        self.device = config.get('device', torch.device('cpu'))
        self.hidden_dim = hidden_dim

        # graph conv
        self.pre_defined_graph = pre_defined_graph
        self.use_predefined_graph = use_pre
        self.use_dynamic_hidden_graph = dy_graph
        self.use_static__hidden_graph = sta_graph

        self.support_len = len(self.pre_defined_graph) + int(dy_graph) + int(sta_graph)
        self.num_matric = (int(use_pre) * len(self.pre_defined_graph) + len(self.pre_defined_graph) * int(dy_graph) + int(sta_graph)) * self.k_s + 1
        self.dropout = nn.Dropout(self.dropout)
        self.pre_defined_graph = self.get_graph(self.pre_defined_graph)

        self.fc_list_updt = nn.Linear(self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False)
        self.gcn_updt = nn.Linear(self.hidden_dim * self.num_matric, self.hidden_dim)

        # others
        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.activation = nn.ReLU()

    def gconv(self, support, X_k, X_0):
        out = [X_0]
        for graph in support:
            if len(graph.shape) == 2:  # staitic or predefined graph
                pass
            else:
                graph = graph.unsqueeze(1)
            H_k = torch.matmul(graph, X_k)
            out.append(H_k)
        out = torch.cat(out, dim=-1)
        out = self.gcn_updt(out)
        out = self.dropout(out)
        return out

    def get_graph(self, support):
        # Only used in static including static hidden graph and predefined graph, but not used for dynamic graph.
        graph_ordered = []
        mask = 1 - torch.eye(support[0].shape[0]).to(support[0].device)
        for graph in support:
            k_1_order = graph  # 1 order
            graph_ordered.append(k_1_order * mask)
            # e.g., order = 3, k=[2, 3]; order = 2, k=[2]
            for k in range(2, self.k_s+1):
                k_1_order = torch.matmul(graph, k_1_order)
                graph_ordered.append(k_1_order * mask)
        # get st localed graph
        st_local_graph = []
        for graph in graph_ordered:
            graph = graph.unsqueeze(-2).expand(-1, self.k_t, -1)
            graph = graph.reshape(
                graph.shape[0], graph.shape[1] * graph.shape[2])
            # [num_nodes, kernel_size x num_nodes]
            st_local_graph.append(graph)
        # [order, num_nodes, kernel_size x num_nodes]
        return st_local_graph

    def forward(self, X, dynamic_graph, static_graph):
        # X: [bs, seq, nodes, feat]
        # [bs, seq, num_nodes, ks, num_feat]
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        # seq_len is changing
        batch_size, seq_len, num_nodes, kernel_size, num_feat = X.shape

        # support
        support = []
        # predefined graph
        if self.use_predefined_graph:
            support = support + self.pre_defined_graph
        # dynamic graph
        if self.use_dynamic_hidden_graph:
            # k_order is caled in dynamic_graph_constructor component
            support = support + dynamic_graph
        # predefined graphs and static hidden graphs
        if self.use_static__hidden_graph:
            support = support + self.get_graph(static_graph)

        # parallelize
        X = X.reshape(batch_size, seq_len, num_nodes, kernel_size * num_feat)
        # batch_size, seq_len, num_nodes, kernel_size * hidden_dim
        out = self.fc_list_updt(X)
        out = self.activation(out)
        out = out.view(batch_size, seq_len, num_nodes, kernel_size, num_feat)
        X_0 = torch.mean(out, dim=-2)
        # batch_size, seq_len, kernel_size x num_nodes, hidden_dim
        X_k = out.transpose(-3, -2).reshape(batch_size,
                                            seq_len, kernel_size*num_nodes, num_feat)
        # Nx3N 3NxD -> NxD: batch_size, seq_len, num_nodes, hidden_dim
        hidden = self.gconv(support, X_k, X_0)
        return hidden


class DifForecast(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=None, config=None):
        super().__init__()
        self.k_t = config.get('k_t', 3)
        self.output_seq_len = config.get('output_window', 12)
        self.gap = config.get('gap', 3)
        self.forecast_fc = nn.Linear(hidden_dim, forecast_hidden_dim)

    def forward(self, gated_history_data, hidden_states_dif, localized_st_conv, dynamic_graph, static_graph):
        predict = []
        history = gated_history_data
        predict.append(hidden_states_dif[:, -1, :, :].unsqueeze(1))
        for _ in range(int(self.output_seq_len / self.gap)-1):
            _1 = predict[-self.k_t:]
            if len(_1) < self.k_t:
                sub = self.k_t - len(_1)
                _2  = history[:, -sub:, :, :]
                _1  = torch.cat([_2] + _1, dim=1)
            else:
                _1  = torch.cat(_1, dim=1)
            predict.append(localized_st_conv(_1, dynamic_graph, static_graph))
        predict = torch.cat(predict, dim=1)
        predict = self.forecast_fc(predict)
        return predict


class InhForecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, config=None):
        super().__init__()
        self.output_seq_len = config.get('output_window', 12)
        self.gap = config.get('gap', 3)
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        [batch_size, _, num_nodes, num_feat] = X.shape

        predict = [Z[-1, :, :].unsqueeze(0)]
        for _ in range(int(self.output_seq_len / self.gap)-1):
            # RNN
            _gru = rnn_layer.gru_cell(predict[-1][0], RNN_H[-1]).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru], dim=0)
            # Positional Encoding
            if pe is not None:
                RNN_H = pe(RNN_H)
            # Transformer
            _Z  = transformer_layer(_gru, K=RNN_H, V=RNN_H)
            predict.append(_Z)
        
        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, batch_size, num_nodes, num_feat)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)
        return predict


class ResidualDecomp(nn.Module):
    """Residual decomposition."""

    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.ReLU()

    def forward(self, x, y):
        u = x - self.ac(y)
        u = self.ln(u)
        return u


class DifBlock(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=256, config=None, data_feature=None):
        """Diffusion block

        Args:
            hidden_dim (int): hidden dimension.
            forecast_hidden_dim (int, optional): forecast branch hidden dimension. Defaults to 256.
            use_pre (bool, optional): if use predefined graph. Defaults to None.
            dy_graph (bool, optional): if use dynamic graph. Defaults to None.
            sta_graph (bool, optional): if use static graph (the adaptive graph). Defaults to None.
        """

        super().__init__()
        self.device = config.get('device', torch.device('cpu'))
        self.adj_mx = data_feature.get("adj_mx")
        self.pre_defined_graph = [
            torch.from_numpy(transition_matrix(self.adj_mx).T).to(self.device),
            torch.from_numpy(transition_matrix(self.adj_mx.T).T).to(self.device),
        ]
        use_pre = config.get("use_pre", False)
        dy_graph = config.get("dy_graph", True)
        sta_graph = config.get("sta_graph", True)

        # diffusion model
        self.localized_st_conv = STLocalizedConv(hidden_dim, pre_defined_graph=self.pre_defined_graph, use_pre=use_pre, dy_graph=dy_graph, sta_graph=sta_graph, config=config)

        # forecast
        self.forecast_branch = DifForecast(hidden_dim, forecast_hidden_dim=forecast_hidden_dim, config=config)
        # backcast
        self.backcast_branch = nn.Linear(hidden_dim, hidden_dim)
        # esidual decomposition
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        """Diffusion block, containing the diffusion model, forecast branch, backcast branch, and the residual decomposition link.

        Args:
            history_data (torch.Tensor): history data with shape [batch_size, seq_len, num_nodes, hidden_dim]
            gated_history_data (torch.Tensor): gated history data with shape [batch_size, seq_len, num_nodes, hidden_dim]
            dynamic_graph (list): dynamic graphs.
            static_graph (list): static graphs (the adaptive graph).

        Returns:
            torch.Tensor: the output after the decoupling mechanism (backcast branch and the residual link), which should be fed to the inherent model. 
                          Shape: [batch_size, seq_len', num_nodes, hidden_dim]. Kindly note that after the st conv, the sequence will be shorter.
            torch.Tensor: the output of the forecast branch, which will be used to make final prediction.
                          Shape: [batch_size, seq_len'', num_nodes, forecast_hidden_dim]. seq_len'' = future_len / gap. 
                          In order to reduce the error accumulation in the AR forecasting strategy, we let each hidden state generate the prediction of gap points, instead of a single point.
        """

        # diffusion model
        hidden_states_dif = self.localized_st_conv(gated_history_data, dynamic_graph, static_graph)
        # forecast branch: use the localized st conv to predict future hidden states.
        forecast_hidden = self.forecast_branch(gated_history_data, hidden_states_dif, self.localized_st_conv, dynamic_graph, static_graph)
        # backcast branch: use FC layer to do backcast
        backcast_seq = self.backcast_branch(hidden_states_dif)
        # residual decomposition: remove the learned knowledge from input data
        backcast_seq = backcast_seq
        history_data = history_data[:, -backcast_seq.shape[1]:, :, :]
        backcast_seq_res = self.residual_decompose(history_data, backcast_seq)

        return backcast_seq_res, forecast_hidden


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=None, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, X):
        X = X + self.pe[:X.size(0)]
        X = self.dropout(X)
        return X


class RNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        [batch_size, seq_len, num_nodes, hidden_dim] = X.shape
        X = X.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, hidden_dim)
        hx = torch.zeros_like(X[:, 0, :])
        output  = []
        for _ in range(X.shape[1]):
            hx  = self.gru_cell(X[:, _, :], hx)
            output.append(hx)
        output = torch.stack(output, dim=0)
        output = self.dropout(output)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.multi_head_self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, K, V):
        hidden_states_MSA = self.multi_head_self_attention(X, K, V)[0]
        hidden_states_MSA = self.dropout(hidden_states_MSA)
        return hidden_states_MSA


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True, forecast_hidden_dim=256, config=None):
        """Inherent block

        Args:
            hidden_dim (int): hidden dimension
            num_heads (int, optional): number of heads of MSA. Defaults to 4.
            bias (bool, optional): if use bias. Defaults to True.
            forecast_hidden_dim (int, optional): forecast branch hidden dimension. Defaults to 256.
        """
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim
        self.dropout = config.get('dropout', 0.1)

        # inherent model
        self.pos_encoder = PositionalEncoding(hidden_dim, self.dropout)
        self.rnn_layer = RNNLayer(hidden_dim, self.dropout)
        self.transformer_layer = TransformerLayer(hidden_dim, num_heads, self.dropout, bias)
        
        # forecast branch
        self.forecast_block = InhForecast(hidden_dim, forecast_hidden_dim, config=config)
        # backcast branch
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)
        # residual decomposition
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, hidden_inherent_signal):
        """Inherent block, containing the inherent model, forecast branch, backcast branch, and the residual decomposition link.

        Args:
            hidden_inherent_signal (torch.Tensor): hidden inherent signal with shape [batch_size, seq_len, num_nodes, num_feat].

        Returns:
            torch.Tensor: the output after the decoupling mechanism (backcast branch and the residual link), which should be fed to the next decouple layer. 
                          Shape: [batch_size, seq_len, num_nodes, hidden_dim]. 
            torch.Tensor: the output of the forecast branch, which will be used to make final prediction.
                          Shape: [batch_size, seq_len'', num_nodes, forecast_hidden_dim]. seq_len'' = future_len / gap. 
                          In order to reduce the error accumulation in the AR forecasting strategy, we let each hidden state generate the prediction of gap points, instead of a single point.
        """

        [batch_size, seq_len, num_nodes, num_feat] = hidden_inherent_signal.shape
        # inherent model
        ## rnn
        hidden_states_rnn = self.rnn_layer(hidden_inherent_signal)
        ## pe
        hidden_states_rnn = self.pos_encoder(hidden_states_rnn)                   
        ## MSA
        hidden_states_inh = self.transformer_layer(hidden_states_rnn, hidden_states_rnn, hidden_states_rnn)

        # forecast branch
        forecast_hidden = self.forecast_block(hidden_inherent_signal, hidden_states_rnn, hidden_states_inh, self.transformer_layer, self.rnn_layer, self.pos_encoder)

        # backcast branch
        hidden_states_inh = hidden_states_inh.reshape(seq_len, batch_size, num_nodes, num_feat)
        hidden_states_inh = hidden_states_inh.transpose(0, 1)
        backcast_seq = self.backcast_fc(hidden_states_inh)                                   
        backcast_seq_res= self.residual_decompose(hidden_inherent_signal, backcast_seq)                    

        return backcast_seq_res, forecast_hidden


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, config=None, data_feature=None):
        super().__init__()
        self.node_hidden = config.get('node_hidden', 10)
        self.time_emb_dim = config.get('time_emb_dim', 10)
        self.add_time_in_day = config.get('add_time_in_day', True)
        self.add_day_in_week = config.get('add_day_in_week', True)
        self.estimation_gate = EstimationGate(node_emb_dim=self.node_hidden, time_emb_dim=self.time_emb_dim, hidden_dim=64, config=config)
        self.dif_layer = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim, config=config, data_feature=data_feature)
        self.inh_layer = InhBlock(hidden_dim, forecast_hidden_dim=fk_dim, config=config)

    def forward(self, history_data: torch.Tensor, dynamic_graph: torch.Tensor, static_graph, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat):
        """decouple layer

        Args:
            history_data (torch.Tensor): input data with shape (B, L, N, D)
            dynamic_graph (list of torch.Tensor): dynamic graph adjacency matrix with shape (B, N, k_t * N)
            static_graph (ist of torch.Tensor): the self-adaptive transition matrix with shape (N, N)
            node_embedding_u (torch.Parameter): node embedding E_u
            node_embedding_d (torch.Parameter): node embedding E_d
            time_in_day_feat (torch.Parameter): time embedding T_D
            day_in_week_feat (torch.Parameter): time embedding T_W

        Returns:
            torch.Tensor: the un decoupled signal in this layer, i.e., the X^{l+1}, which should be feeded to the next layer. shape [B, L', N, D].
            torch.Tensor: the output of the forecast branch of Diffusion Block with shape (B, L'', N, D), where L''=output_seq_len / model_args['gap'] to avoid error accumulation in auto-regression.
            torch.Tensor: the output of the forecast branch of Inherent Block with shape (B, L'', N, D), where L''=output_seq_len / model_args['gap'] to avoid error accumulation in auto-regression.
        """

        gated_history_data = self.estimation_gate(node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data)
        dif_backcast_seq_res, dif_forecast_hidden = self.dif_layer(history_data=history_data, gated_history_data=gated_history_data, dynamic_graph=dynamic_graph, static_graph=static_graph)   
        inh_backcast_seq_res, inh_forecast_hidden = self.inh_layer(dif_backcast_seq_res)         
        return inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden


class DistanceFunction(nn.Module):
    def __init__(self, config):
        super().__init__()
        # attributes
        self.hidden_dim = config.get('num_hidden', 32)
        self.node_dim = config.get('node_hidden', 10)
        self.time_slot_emb_dim  = self.hidden_dim
        self.input_seq_len = config.get('input_window', 12)
        dropout = config.get('dropout', 0.1)
        self.time_emb_dim = config.get('time_emb_dim', 10)
        # Time Series Feature Extraction
        self.dropout    = nn.Dropout(dropout)
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ts_feat_dim= self.hidden_dim
        # Time Slot Embedding Extraction
        self.time_slot_embedding = nn.Linear(self.time_emb_dim, self.time_slot_emb_dim)
        # Distance Score
        time_dim = 0
        if config.get("add_time_in_day", True):
            time_dim += self.time_emb_dim
        if config.get("add_day_in_week", True):
            time_dim += self.time_emb_dim
        self.all_feat_dim = self.ts_feat_dim + self.node_dim + time_dim
        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.bn = nn.BatchNorm1d(self.hidden_dim*2)

    def reset_parameters(self):
        for q_vec in self.q_vecs:
            nn.init.xavier_normal_(q_vec.data)
        for bias in self.biases:
            nn.init.zeros_(bias.data)

    def forward(self, X, E_d, E_u, T_D, D_W):
        # last pooling
        if T_D is not None:
            T_D = T_D[:, -1, :, :]
        if D_W is not None:
            D_W = D_W[:, -1, :, :]
        # dynamic information
        X = X[:, :, :, 0].transpose(1, 2).contiguous()  # X->[batch_size, seq_len, num_nodes]->[batch_size, num_nodes, seq_len]
        [batch_size, num_nodes, seq_len] = X.shape
        X = X.view(batch_size * num_nodes, seq_len)
        dy_feat = self.fc_ts_emb2(self.dropout(self.bn(F.relu(self.fc_ts_emb1(X)))))  # [batchsize, num_nodes, hidden_dim]
        dy_feat = dy_feat.view(batch_size, num_nodes, -1)
        # node embedding
        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)
        # distance calculation
        if T_D  is not None and D_W is not None:
            X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)  # hidden state for calculating distance
            X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)  # hidden state for calculating distance
        elif D_W is not None:
            X1 = torch.cat([dy_feat, D_W, emb1], dim=-1)  # hidden state for calculating distance
            X2 = torch.cat([dy_feat, D_W, emb2], dim=-1)  # hidden state for calculating distance
        elif T_D is not None:
            X1 = torch.cat([dy_feat, T_D, emb1], dim=-1)  # hidden state for calculating distance
            X2 = torch.cat([dy_feat, T_D, emb2], dim=-1)  # hidden state for calculating distance
        else:
            X1 = torch.cat([dy_feat, emb1], dim=-1)  # hidden state for calculating distance
            X2 = torch.cat([dy_feat, emb2], dim=-1)  # hidden state for calculating distance
        X  = [X1, X2]
        adjacent_list = []
        for _ in X:
            Q = self.WQ(_)
            K = self.WK(_)
            QKT = torch.bmm(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
            W = torch.softmax(QKT, dim=-1)
            adjacent_list.append(W)
        return adjacent_list


class Mask(nn.Module):
    def __init__(self, config=None, data_feature=None):
        super().__init__()
        self.device = config.get('device', torch.device('cpu'))
        self.adj_mx = data_feature.get("adj_mx")
        self.mask = [
            torch.from_numpy(transition_matrix(self.adj_mx).T).to(self.device),
            torch.from_numpy(transition_matrix(self.adj_mx.T).T).to(self.device),
        ]
    
    def _mask(self, index, adj):
        mask = self.mask[index] + torch.ones_like(self.mask[index]) * 1e-7
        return mask.to(self.device) * adj

    def forward(self, adj):
        result = []
        for index, _ in enumerate(adj):
            result.append(self._mask(index, _))
        return result


class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def _norm(self, graph):
        degree  = torch.sum(graph, dim=2)
        degree  = remove_nan_inf(1 / degree)
        degree  = torch.diag_embed(degree)
        normed_graph = torch.bmm(degree, graph)
        return normed_graph

    def forward(self, adj):
        return [self._norm(_) for _ in adj]


class MultiOrder(nn.Module):
    def __init__(self, order=2, config=None):
        super().__init__()
        self.order = order
        self.device = config.get('device', torch.device('cpu'))

    def _multi_order(self, graph):
        graph_ordered = []
        k_1_order = graph  # 1 order
        mask = torch.eye(graph.shape[1]).to(self.device)
        mask = 1 - mask
        graph_ordered.append(k_1_order * mask)
        for k in range(2, self.order+1):  # e.g., order = 3, k=[2, 3]; order = 2, k=[2]
            k_1_order = torch.matmul(k_1_order, graph)
            graph_ordered.append(k_1_order * mask)
        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(_) for _ in adj]


class DynamicGraphConstructor(nn.Module):
    def __init__(self, config=None, data_feature=None):
        super().__init__()
        # model args
        self.k_s = config.get('k_s', 2)  # spatial order
        self.k_t = config.get('k_t', 3)  # temporal kernel size
        # hidden dimension of
        self.hidden_dim = config.get('num_hidden', 32)
        # trainable node embedding dimension
        self.node_dim = config.get('node_hidden', 10)

        self.distance_function = DistanceFunction(config=config)
        self.mask = Mask(config=config, data_feature=data_feature)
        self.normalizer = Normalizer()
        self.multi_order = MultiOrder(order=self.k_s, config=config)

    def st_localization(self, graph_ordered):
        st_local_graph = []
        for modality_i in graph_ordered:
            for k_order_graph in modality_i:
                k_order_graph = k_order_graph.unsqueeze(
                    -2).expand(-1, -1, self.k_t, -1)
                k_order_graph = k_order_graph.reshape(
                    k_order_graph.shape[0], k_order_graph.shape[1], k_order_graph.shape[2] * k_order_graph.shape[3])
                st_local_graph.append(k_order_graph)
        return st_local_graph

    def forward(self, history_data, node_embedding_d, node_embedding_u, time_in_day_feat, day_in_week_feat):
        """Dynamic graph learning module.

        Args:
            history_data (torch.Tensor): input data with shape (B, L, N, D)
            node_embedding_u (torch.Parameter): node embedding E_u
            node_embedding_d (torch.Parameter): node embedding E_d
            time_in_day_feat (torch.Parameter): time embedding T_D
            day_in_week_feat (torch.Parameter): time embedding T_W

        Returns:
            list: dynamic graphs
        """

        # distance calculation
        dist_mx = self.distance_function(history_data, node_embedding_d, node_embedding_u, time_in_day_feat, day_in_week_feat)
        # mask
        dist_mx = self.mask(dist_mx)
        # normalization
        dist_mx = self.normalizer(dist_mx)
        # multi order
        mul_mx = self.multi_order(dist_mx)
        # spatial temporal localization
        dynamic_graphs = self.st_localization(mul_mx)

        return dynamic_graphs


class D2STGNN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._scaler = self.data_feature.get('scaler') 
        self.num_nodes = data_feature.get('num_nodes')
        self.feature_dim = data_feature.get('feature_dim')
        self.output_dim = data_feature.get('output_dim')
        self.num_batches = data_feature.get('num_batches')

        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.add_time_in_day = config.get('add_time_in_day', False)
        self.add_day_in_week = config.get('add_day_in_week', False)
        if self.add_time_in_day and self.add_day_in_week:
            self.feature_dim -= 8
        elif self.add_time_in_day:
            self.feature_dim -= 1
        elif self.add_day_in_week:
            self.feature_dim -= 7

        self.hidden_dim = config.get('num_hidden', 32)
        self.node_dim = config.get('node_dim', 10)
        self.forecast_dim = config.get('forecast_dim', 256)
        self.output_hidden = config.get('output_hidden', 512)
        self.k_s = config.get('k_s', 2)
        self.k_t = config.get('k_t', 3)
        self.num_layers = config.get('num_layers', 5)
        self.time_emb_dim = config.get('time_emb_dim', 10)
        self.gap = config.get('gap', 3)

        self.use_pre = config.get('use_pre', False)
        self.dy_graph = config.get('dy_graph', True)
        self.sta_graph = config.get('sta_graph', True)

        self.use_curriculum_learning = config.get('use_curriculum_learning', False)
        self.cl_decay_epochs = config.get('cl_decay_epochs', 3)
        self.max_epoch = config.get('max_epoch', 100)
        if self.max_epoch < self.cl_decay_epochs * self.output_window:
            self._logger.warning('Parameter `step_size1` is too big with {} epochs and '
                                 'the model cannot be trained for all time steps.'.format(self.max_epoch))
        self.task_level = config.get('task_level', 0)

        # start embedding layer
        self.embedding = nn.Linear(self.feature_dim, self.hidden_dim)

        # time embedding
        self.T_i_D_emb = nn.Parameter(torch.empty(288, self.time_emb_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, self.time_emb_dim))

        # Decoupled Spatial Temporal Layer
        self.layers = nn.ModuleList([DecoupleLayer(self.hidden_dim, fk_dim=self.forecast_dim, config=config, data_feature=data_feature)])
        for _ in range(self.num_layers - 1):
            self.layers.append(DecoupleLayer(self.hidden_dim, fk_dim=self.forecast_dim, config=config, data_feature=data_feature))

        # dynamic and static hidden graph constructor
        if self.dy_graph:
            self.dynamic_graph_constructor = DynamicGraphConstructor(config=config, data_feature=data_feature)

        # node embeddings
        self.node_emb_u = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))

        # output layer
        self.out_fc_1 = nn.Linear(self.forecast_dim, self.output_hidden)
        self.out_fc_2 = nn.Linear(self.output_hidden, self.gap * self.output_dim)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def _graph_constructor(self, node_embedding_u, node_embedding_d, history_data, time_in_day_feat, day_in_week_feat):
        if self.sta_graph:
            static_graph = [F.softmax(F.relu(torch.mm(node_embedding_u, node_embedding_d.T)), dim=1)]
        else:
            static_graph = []
        if self.dy_graph:
            dynamic_graph = self.dynamic_graph_constructor(history_data, node_embedding_d, node_embedding_u, time_in_day_feat, day_in_week_feat)
        else:
            dynamic_graph = []
        return static_graph, dynamic_graph

    def _prepare_inputs(self, history_data):
        num_feat = self.feature_dim

        # node embeddings
        node_emb_u = self.node_emb_u  # [N, d]
        node_emb_d = self.node_emb_d  # [N, d]
        # time slot embedding
        time_in_day_feat, day_in_week_feat = None, None
        if self.add_time_in_day and self.add_day_in_week:
            time_in_day_feat = self.T_i_D_emb[(history_data[:, :, :, num_feat] * 288).type(torch.LongTensor)]  # [B, L, N, d]
            day_in_week_feat = self.D_i_W_emb[(history_data[:, :, :, num_feat + 1: num_feat + 8].argmax(dim=3)).type(torch.LongTensor)]  # [B, L, N, d]
        elif self.add_time_in_day:
            time_in_day_feat = self.T_i_D_emb[(history_data[:, :, :, num_feat] * 288).type(torch.LongTensor)]  # [B, L, N, d]
        elif self.add_day_in_week:
            day_in_week_feat = self.D_i_W_emb[(history_data[:, :, :, num_feat: num_feat + 7].argmax(dim=3)).type(torch.LongTensor)]  # [B, L, N, d]

        # traffic signals
        history_data = history_data[:, :, :, :num_feat]

        return history_data, node_emb_u, node_emb_d, time_in_day_feat, day_in_week_feat

    def forward(self, batch):
        """Feed forward of D2STGNN.

        Args:
            history_data (Tensor): history data with shape: [B, L, N, C]

        Returns:
            torch.Tensor: prediction data with shape: [B, N, L]
        """

        history_data = batch['X']
        # ==================== Prepare Input Data ==================== #
        history_data, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat = self._prepare_inputs(history_data)

        # ========================= Construct Graphs ========================== #
        static_graph, dynamic_graph = self._graph_constructor(node_embedding_u=node_embedding_u, node_embedding_d=node_embedding_d, history_data=history_data, time_in_day_feat=time_in_day_feat, day_in_week_feat=day_in_week_feat)

        # Start embedding layer
        history_data = self.embedding(history_data)

        dif_forecast_hidden_list = []
        inh_forecast_hidden_list = []

        inh_backcast_seq_res = history_data
        for _, layer in enumerate(self.layers):
            inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden = layer(inh_backcast_seq_res, dynamic_graph, static_graph, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat)
            dif_forecast_hidden_list.append(dif_forecast_hidden)
            inh_forecast_hidden_list.append(inh_forecast_hidden)

        # Output Layer
        dif_forecast_hidden = sum(dif_forecast_hidden_list)
        inh_forecast_hidden = sum(inh_forecast_hidden_list)
        forecast_hidden = dif_forecast_hidden + inh_forecast_hidden  # (B, T / gap, N, hiiden_dim)
        
        # regression layer
        forecast = self.out_fc_2(F.relu(self.out_fc_1(F.relu(forecast_hidden))))
        forecast = forecast.view(forecast.shape[0], forecast.shape[1], forecast.shape[2], self.gap, self.output_dim)  # (B, T / gap, N, gap, output_dim)
        forecast = forecast.transpose(1,2).contiguous().view(forecast.shape[0], forecast.shape[2], -1, self.output_dim)  # (B, N, T, output_dim)

        return forecast.transpose(1, 2)

    def calculate_loss(self, batch, batches_seen=None):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        if self.training:
            if batches_seen % (self.cl_decay_epochs * self.num_batches) == 0 and self.task_level < self.output_window:
                self.task_level += 1
                self._logger.info('Training: task_level increase from {} to {}'.format(
                    self.task_level-1, self.task_level))
                self._logger.info('Current batches_seen is {}'.format(batches_seen))
            if self.use_curriculum_learning:
                return loss.masked_mae_torch(y_predicted[:, :self.task_level, :, :],
                                             y_true[:, :self.task_level, :, :], 0)
            else:
                return loss.masked_mae_torch(y_predicted, y_true, 0)
        else:
            return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)
