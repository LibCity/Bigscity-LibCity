from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy as cp

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('nvlc,vw->nwlc', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, supports_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * supports_len + 1) * c_in
        self.mlp = nn.Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=-1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class QKVAttention(nn.Module):
    """
    Assume input has shape B, N, T, C or B, T, N, C
    Note: Attention map will be B, N, T, T or B, T, N, N
        - Could be utilized for both spatial and temporal modeling
        - Able to get additional kv-input (for Time-Enhanced Attention)
    """

    def __init__(self, in_dim, hidden_size, dropout, num_heads=4):
        super(QKVAttention, self).__init__()
        self.query = nn.Linear(in_dim, hidden_size, bias=False)
        self.key = nn.Linear(in_dim, hidden_size, bias=False)
        self.value = nn.Linear(in_dim, hidden_size, bias=False)
        self.num_heads = num_heads
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        assert hidden_size % num_heads == 0

    def forward(self, x, kv=None):
        if kv is None:
            kv = x
        query = self.query(x)
        key = self.key(kv)
        value = self.value(kv)
        num_heads = self.num_heads
        if num_heads > 1:
            query = torch.cat(torch.chunk(query, num_heads, dim=-1), dim=0)
            key = torch.cat(torch.chunk(key, num_heads, dim=-1), dim=0)
            value = torch.cat(torch.chunk(value, num_heads, dim=-1), dim=0)
        d = value.size(-1)
        energy = torch.matmul(query, key.transpose(-1, -2))
        energy = energy / (d ** 0.5)
        score = torch.softmax(energy, dim=-1)
        head_out = torch.matmul(score, value)
        out = torch.cat(torch.chunk(head_out, num_heads, dim=0), dim=-1)
        return self.dropout(self.proj(out))


class LayerNorm(nn.Module):
    # Assume input has shape B, N, T, C
    def __init__(self, normalized_shape, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(*normalized_shape))
        self.beta = nn.Parameter(torch.zeros(*normalized_shape))

    def forward(self, x):
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        # mean --> shape :(B, C, H, W) --> (B)
        # mean with keepdims --> shape: (B, C, H, W) --> (B, 1, 1, 1)
        mean = x.mean(dim=dims, keepdims=True)
        std = x.std(dim=dims, keepdims=True, unbiased=False)
        # x_norm = (B, C, H, W)
        x_norm = (x - mean) / (std + self.eps)
        out = x_norm * self.gamma + self.beta
        return out


class BatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5, track_running_stats=True):
        super(BatchNorm, self).__init__()
        self.momentum = momentum
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)

    def forward(self, x):
        dims = [i for i in range(x.dim() - 1)]
        mean = x.mean(dim=dims)
        var = x.var(dim=dims, correction=0)
        if (self.training) and (self.running_mean is not None):
            avg_factor = self.momentum
            moving_avg = lambda prev, cur: (1 - avg_factor) * prev + avg_factor * cur.detach()
            dims = [i for i in range(x.dim() - 1)]
            self.running_mean = moving_avg(self.running_mean, mean)
            self.running_var = moving_avg(self.running_var, var)
            mean, var = self.running_mean, self.running_var

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        out = x_norm * self.gamma + self.beta
        return out


class SkipConnection(nn.Module):
    """
    Helper Module to build skip connection
     - forward may get auxiliary input to handle multiple inputs (e.g., adjacency matrix or time-enhanced attention)
    """

    def __init__(self, module, norm):
        super(SkipConnection, self).__init__()
        self.module = module
        self.norm = norm

    def forward(self, x, aux=None):
        return self.norm(x + self.module(x, aux))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, in_dim, hidden_size, dropout, activation=nn.GELU()):
        super(PositionwiseFeedForward, self).__init__()
        self.act = activation
        self.l1 = nn.Linear(in_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, in_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, kv=None):
        return self.dropout(self.l2(self.act(self.l1(x))))


class SwitchPositionwiseFeedForward(nn.Module):
    """
    Switch Positionwise Feed Forward module for the normal mixture-of-experts model
     - Note: not used for the TESTAM
    """

    def __init__(self, in_dim, hidden_size, dropout, activation=nn.ReLU(), n_experts=4):
        super(SwitchPositionwiseFeedForward, self).__init__()
        self.n_experts = n_experts
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        expert = PositionwiseFeedForward(in_dim, hidden_size, dropout, activation)
        self.experts = nn.ModuleList([cp(expert) for _ in range(n_experts)])
        self.switch = nn.Linear(in_dim, n_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, kv=None):
        B, N, T, C = x.size()
        x = x.view(-1, C)

        route_prob = self.softmax(self.switch(x))
        route_prob_max, routes = torch.max(route_prob, dim=-1)

        # indices: (n_experts, B*T, N)
        indices = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]

        final_output = torch.zeros_like(x)

        for i in range(self.n_experts):
            expert_output = self.experts[i](x[indices[i]])
            final_output[indices[i]] = expert_output

        final_output = final_output * (route_prob_max).unsqueeze(dim=-1)
        final_output = final_output.view(B, N, T, C)

        return final_output


class TemporalInformationEmbedding(nn.Module):
    """
    We assume that input shape is B, T
        - Only contains temporal information with index
    Arguments:
        - vocab_size: total number of temporal features (e.g., 7 days)
        - freq_act: periodic activation function
        - n_freq: number of hidden elements for frequency components
            - if 0 or H, it only uses linear or frequency component, respectively
    """

    def __init__(self, hidden_size, vocab_size, freq_act=torch.sin, n_freq=1):
        super(TemporalInformationEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.freq_act = freq_act
        self.n_freq = n_freq

    def forward(self, x):
        x_emb = self.embedding(x)
        x_weight = self.linear(x_emb)
        if self.n_freq == 0:
            return x_weight
        if self.n_freq == x_emb.size(-1):
            return self.freq_act(x_weight)
        x_linear = x_weight[..., self.n_freq:]
        x_act = self.freq_act(x_weight[..., :self.n_freq])
        return torch.cat([x_linear, x_act], dim=-1)


class TemporalModel(nn.Module):
    """
    Input shape
        - x: B, T
            - Need modification to use the multiple temporal information with different indexing (e.g., dow and tod)
        - speed: B, N, T, in_dim = 1
            - Need modification to use them in different dataset
    Output shape B, N, T, O
        - In the traffic forecasting, O (outdim) is normally one
    Arguments:
        - vocab_size: total number of temporal features (e.g., 7 days)
            - Notes: in the trivial traffic forecasting problem, we have total 288 = 24 * 60 / 5 (5 min interval)
    """

    def __init__(self, hidden_size, num_nodes, layers, dropout, in_dim=1, vocab_size=288, activation=nn.ReLU()):
        super(TemporalModel, self).__init__()
        self.vocab_size = vocab_size
        self.act = activation
        self.embedding = TemporalInformationEmbedding(hidden_size, vocab_size=vocab_size)
        self.spd_proj = nn.Linear(in_dim, hidden_size)
        self.spd_cat = nn.Linear(hidden_size * 2, hidden_size)  # Cat speed information and TIM information

        module = QKVAttention(in_dim=hidden_size, hidden_size=hidden_size, dropout=dropout)
        ff = PositionwiseFeedForward(in_dim=hidden_size, hidden_size=4 * hidden_size, dropout=dropout)
        norm = LayerNorm(normalized_shape=(hidden_size,))

        self.node_features = nn.Parameter(torch.randn(num_nodes, hidden_size))

        self.attn_layers = nn.ModuleList()
        self.ff = nn.ModuleList()
        for _ in range(layers):
            self.attn_layers.append(SkipConnection(cp(module), cp(norm)))
            self.ff.append(SkipConnection(cp(ff), cp(norm)))

        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, x, speed=None):
        TIM = self.embedding(x)
        # For the traffic forecasting, we introduce learnable node features
        # The user may modify this node feature into meta-learning based representation, which enables the ability to adopt the model into different dataset
        x_nemb = torch.einsum('btc, nc -> bntc', TIM, self.node_features)
        if speed is None:
            speed = torch.zeros_like(x_nemb[..., 0])
        x_spd = self.spd_proj(speed.unsqueeze(dim=-1))
        x_nemb = self.spd_cat(torch.cat([x_spd, x_nemb], dim=-1))

        attns = []
        for i, (attn_layer, ff) in enumerate(zip(self.attn_layers, self.ff)):
            x_attn = attn_layer(x_nemb)
            x_nemb = ff(x_attn)
            attns.append(x_nemb)

        out = self.proj(self.act(x_nemb))

        return out, attns


class STModel(nn.Module):
    """
    Input shape B, N, T, in_dim
    Output shape B, N, T, out_dim
    Arguments:
        - spatial: Flag that determine when spatial attention will be performed
            - True --> spatial first and then temporal attention will be performed
    """

    def __init__(self, hidden_size, supports_len, num_nodes, dropout, layers, out_dim=1, in_dim=2, spatial=False,
                 activation=nn.ReLU()):
        super(STModel, self).__init__()
        self.spatial = spatial
        self.act = activation

        s_gcn = gcn(c_in=hidden_size, c_out=hidden_size, dropout=dropout, supports_len=supports_len, order=2)
        t_attn = QKVAttention(in_dim=hidden_size, hidden_size=hidden_size, dropout=dropout)
        ff = PositionwiseFeedForward(in_dim=hidden_size, hidden_size=4 * hidden_size, dropout=dropout)
        norm = LayerNorm(normalized_shape=(hidden_size,))

        self.start_linear = nn.Linear(in_dim, hidden_size)

        if out_dim == 1:
            self.proj = nn.Linear(hidden_size, hidden_size + out_dim)
        else:
            self.proj = nn.Linear(hidden_size, out_dim)
        self.out_dim = out_dim

        self.temporal_layers = nn.ModuleList()
        self.spatial_layers = nn.ModuleList()
        self.ed_layers = nn.ModuleList()
        self.ff = nn.ModuleList()

        for _ in range(layers):
            self.temporal_layers.append(SkipConnection(cp(t_attn), cp(norm)))
            self.spatial_layers.append(SkipConnection(cp(s_gcn), cp(norm)))
            self.ed_layers.append(SkipConnection(cp(t_attn), cp(norm)))
            self.ff.append(SkipConnection(cp(ff), cp(norm)))

    def forward(self, x, prev_hidden, supports):
        x = self.start_linear(x.permute(0, 2, 3, 1))
        x_start = x
        hiddens = []
        for i, (temporal_layer, spatial_layer, ed_layer, ff) in enumerate(
                zip(self.temporal_layers, self.spatial_layers, self.ed_layers, self.ff)):
            if not self.spatial:
                x1 = temporal_layer(x)  # B, N, T, C
                x_attn = spatial_layer(x1, supports)  # B, N, T, C
            else:
                x1 = spatial_layer(x, supports)
                x_attn = temporal_layer(x1)
            if prev_hidden is not None:
                x_attn = ed_layer(x_attn, prev_hidden[-1])
            x = ff(x_attn)
            hiddens.append(x)

        out = self.proj(self.act(x))

        return x_start - out[..., :-1], out[..., [-1]], hiddens


class AttentionModel(nn.Module):
    """
    Input shape B, N, T, in_dim
    Output shape B, N, T, out_dim

    """

    def __init__(self, hidden_size, layers, dropout, edproj=False, in_dim=2, out_dim=1, spatial=False,
                 activation=nn.ReLU()):
        super(AttentionModel, self).__init__()
        self.spatial = spatial
        self.act = activation

        base_model = SkipConnection(QKVAttention(hidden_size, hidden_size, dropout=dropout),
                                    LayerNorm(normalized_shape=(hidden_size,)))
        ff = SkipConnection(PositionwiseFeedForward(hidden_size, 4 * hidden_size, dropout=dropout),
                            LayerNorm(normalized_shape=(hidden_size,)))

        self.start_linear = nn.Linear(in_dim, hidden_size)

        self.spatial_layers = nn.ModuleList()
        self.temporal_layers = nn.ModuleList()
        self.ed_layers = nn.ModuleList()
        self.ff = nn.ModuleList()

        for i in range(layers):
            self.spatial_layers.append(cp(base_model))
            self.temporal_layers.append(cp(base_model))
            self.ed_layers.append(cp(base_model))
            self.ff.append(cp(ff))

        self.proj = nn.Linear(hidden_size, out_dim)

    def forward(self, x, prev_hidden=None):
        x = self.start_linear(x.permute(0, 2, 3, 1))

        for i, (s_layer, t_layer, ff) in enumerate(zip(self.spatial_layers, self.temporal_layers, self.ff)):
            if not self.spatial:
                x1 = t_layer(x)
                x_attn = s_layer(x1.transpose(1, 2))
            else:
                x1 = s_layer(x.transpose(1, 2))
                x_attn = t_layer(x1.transpose(1, 2)).transpose(1, 2)

            if prev_hidden is not None:
                x_attn = self.ed_layers[i](x_attn.transpose(1, 2), prev_hidden[-1])
                x_attn = x_attn.transpose(1, 2)
            x = ff(x_attn.transpose(1, 2))

        return self.proj(self.act(x)), x


class MemoryGate(nn.Module):
    """
    Input
     - input: B, N, T, in_dim, original input
     - hidden: hidden states from each expert, shape: E-length list of (B, N, T, C) tensors, where E is the number of experts
    Output
     - similarity score (i.e., routing probability before softmax function)
    Arguments
     - mem_hid, memory_size: hidden size and total number of memroy units
     - sim: similarity function to evaluate routing probability
     - nodewise: flag to determine routing level. Traffic forecasting could have a more fine-grained routing, because it has additional dimension for the roads
        - True: enables node-wise routing probability calculation, which is coarse-grained one
    """

    def __init__(self, hidden_size, num_nodes, mem_hid=32, input_dim=2, output_dim=1, memory_size=20,
                 sim=nn.CosineSimilarity(dim=-1), nodewise=False, ind_proj=True, attention_type='attention'):
        super(MemoryGate, self).__init__()
        self.attention_type = attention_type
        self.sim = sim
        self.nodewise = nodewise

        self.memory = nn.Parameter(torch.empty(memory_size, mem_hid))

        self.hid_query = nn.ParameterList([nn.Parameter(torch.empty(hidden_size, mem_hid)) for _ in range(3)])
        self.key = nn.ParameterList([nn.Parameter(torch.empty(hidden_size, mem_hid)) for _ in range(3)])
        self.value = nn.ParameterList([nn.Parameter(torch.empty(hidden_size, mem_hid)) for _ in range(3)])

        self.input_query = nn.Parameter(torch.empty(input_dim, mem_hid))

        self.We1 = nn.Parameter(torch.empty(num_nodes, memory_size))
        self.We2 = nn.Parameter(torch.empty(num_nodes, memory_size))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, input, hidden):
        if self.attention_type == 'attention':
            attention = self.attention
        else:
            attention = self.topk_attention
        B, N, T, _ = input.size()
        memories = self.query_mem(input)
        scores = []
        for i, h in enumerate(hidden):
            hidden_att = attention(h, i)
            scores.append(self.sim(memories, hidden_att))

        scores = torch.stack(scores, dim=-1)
        return scores

    def attention(self, x, i):
        B, N, T, _ = x.size()
        query = torch.matmul(x, self.hid_query[i])
        key = torch.matmul(x, self.key[i])
        value = torch.matmul(x, self.value[i])
        if self.nodewise:
            query = query.sum(dim=-2, keepdim=True)
        energy = torch.matmul(query, key.transpose(-1, -2))
        score = torch.softmax(energy, dim=-1)
        out = torch.matmul(score, value)
        return out.expand_as(value)

    def topk_attention(self, x, i, k=3):
        B, N, T, _ = x.size()
        query = torch.matmul(x, self.hid_query[i])
        key = torch.matmul(x, self.key[i])
        value = torch.matmul(x, self.value[i])
        if self.nodewise:
            query = query.sum(dim=-2, keepdim=True)
        energy = torch.matmul(query, key.transpose(-1, -2))
        values, indices = torch.topk(energy, k=k, dim=-1)
        score = energy.zero_().scatter_(-1, indices, torch.relu(values))
        out = torch.matmul(score, value)
        return out.expand_as(value)

    def query_mem(self, input):
        B, N, T, _ = input.size()
        mem = self.memory
        query = torch.matmul(input, self.input_query)
        energy = torch.matmul(query, mem.T)
        score = torch.softmax(energy, dim=-1)
        out = torch.matmul(score, mem)
        return out

    def reset_queries(self):
        with torch.no_grad():
            for p in self.hid_query:
                nn.init.xavier_uniform_(p)
            nn.init.xavier_uniform_(self.input_query)

    def reset_params(self):
        with torch.no_grad():
            for n, p in self.named_parameters():
                if n in "We1 We2 memory".split():
                    continue
                else:
                    nn.init.xavier_uniform_(p)


class AttnGate(nn.Module):
    def __init__(self, hidden_size, num_nodes, input_dim=2, sim=nn.CosineSimilarity(dim=-1)):
        super(AttnGate, self).__init__()
        self.in_key = nn.Linear(input_dim, hidden_size, bias=False)
        self.hid_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.in_value = nn.Linear(input_dim, hidden_size, bias=False)
        sim = lambda x, y: nn.PairwiseDistance()(x, y) * -1
        self.sim = sim
        self.proj = nn.Linear(hidden_size, 1)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, input, hidden):
        num_heads = 1
        key = self.in_key(input)
        value = self.in_value(input)
        if num_heads > 1:
            key = torch.cat(torch.chunk(key, num_heads, dim=-1), dim=0)
            value = torch.cat(torch.chunk(value, num_heads, dim=-1), dim=0)
        scores = []
        for h in hidden:
            query = self.hid_query(h)
            if num_heads > 1:
                head_query = torch.cat(torch.chunk(query, num_heads, dim=-1), dim=0)
                energy = torch.matmul(head_query, key.transpose(-1, -2)) / (head_query.size(-1) ** 0.5)
            else:
                energy = torch.matmul(query, key.transpose(-1, -2)) / (query.size(-1) ** 0.5)
            score = torch.softmax(energy, dim=-1)
            head_out = torch.matmul(score, value)
            out = torch.cat(torch.chunk(head_out, num_heads, dim=0), dim=-1)
            scores.append(self.sim(query, out))
        return torch.stack(scores, dim=-1)


class TESTAM(AbstractTrafficStateModel):

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))

        # data
        num_nodes = self.data_feature.get('num_nodes')
        self._scaler = self.data_feature.get('scaler')

        # model
        dropout = config.get("dropout", 0.3)
        prob_mul = config.get("prob_mul", False)
        in_dim = config.get("input_dim", 2)
        self.output_dim = config.get("output_dim", 1)
        hidden_size = config.get("hidden_size", 32)
        layers = config.get("layers", 3)
        self.is_quantile = config.get("is_quantile", False)
        self.quantile = config.get("quantile", 0.7)

        self.dropout = dropout
        self.prob_mul = prob_mul
        self.supports_len = 2

        self.identity_expert = TemporalModel(hidden_size, num_nodes, in_dim=in_dim - 1, layers=layers, dropout=dropout)
        self.adaptive_expert = STModel(hidden_size, self.supports_len, num_nodes, in_dim=in_dim, layers=layers,
                                       dropout=dropout)
        self.attention_expert = AttentionModel(hidden_size, in_dim=in_dim, layers=layers, dropout=dropout)

        self.gate_network = MemoryGate(hidden_size, num_nodes)

        for model in [self.identity_expert, self.adaptive_expert, self.attention_expert]:
            for n, p in model.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, input, gate_out=False):
        """
        input: B, in_dim, N, T
        o_identity shape B, N, T, 1
        """

        n1 = torch.matmul(self.gate_network.We1, self.gate_network.memory)
        n2 = torch.matmul(self.gate_network.We2, self.gate_network.memory)
        g1 = torch.softmax(torch.relu(torch.mm(n1, n2.T)), dim=-1)
        g2 = torch.softmax(torch.relu(torch.mm(n2, n1.T)), dim=-1)
        new_supports = [g1, g2]

        time_index = input[:, -1, 0]  # B, T
        cur_time_index = (time_index * 288).long()
        next_time_index = ((time_index * 288 + 12) % 288).long()
        o_identity, h_identity = self.identity_expert(cur_time_index, input[:, 0])
        _, h_future = self.identity_expert(next_time_index)

        _, o_adaptive, h_adaptive = self.adaptive_expert(input, h_future, new_supports)

        o_attention, h_attention = self.attention_expert(input, h_future)

        ind_out = torch.cat([o_identity, o_adaptive, o_attention], dim=-1)

        B, N, T, _ = o_identity.size()
        gate_in = [h_identity[-1], h_adaptive[-1], h_attention]
        gate = torch.softmax(self.gate_network(input.permute(0, 2, 3, 1), gate_in), dim=-1)
        out = torch.zeros_like(o_identity).view(-1, 1)

        outs = [o_identity, o_adaptive, o_attention]
        counts = []

        route_prob_max, routes = torch.max(gate, dim=-1)
        route_prob_max = route_prob_max.view(-1)
        routes = routes.view(-1)

        for i in range(len(outs)):
            cur_out = outs[i].view(-1, 1)
            indices = torch.eq(routes, i).nonzero(as_tuple=True)[0]
            out[indices] = cur_out[indices]
            counts.append(len(indices))
        if self.prob_mul:
            out = out * (route_prob_max).unsqueeze(dim=-1)

        out = out.view(B, N, T, 1)

        out = out.permute(0, 3, 1, 2)
        # out: BFNT
        # gate: BNTF
        # ind_out: BNTF
        if self.training or gate_out:
            return out, gate, ind_out
        else:
            # BFNT -> BTNF
            return out.transpose(1, 3)

    def predict(self, batch):
        return self.forward(batch['X'].transpose(1, 3))

    def calculate_loss(self, batch):
        real = batch['y']
        out = self.predict(batch)
        y_true = self._scaler.inverse_transform(real[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(out[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0.0)
