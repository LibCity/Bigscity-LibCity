import math
from logging import getLogger
from typing import Optional

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from infomap import Infomap
from torch import Tensor
from torch_geometric.utils import to_dense_adj, dense_to_sparse, degree

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def SinCosPosEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (
                    torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        print(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += .001
        else:
            x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


class Positional_Encoding(nn.Module):
    """
        general positional encoding layer
        return [len,d_model]
    """

    def __init__(self, pe_type, learn_pe, q_len, d_model, device=torch.device('cpu')):
        super(Positional_Encoding, self).__init__()
        # Positional encoding
        self.device = device
        self.pe_type = pe_type
        if pe_type == None:  # random pe , for measuring impact of pe
            W_pos = torch.empty((q_len, d_model))
            nn.init.uniform_(W_pos, -0.02, 0.02)
            learn_pe = False
        elif pe_type == 'zero':  # 1 dim random pe
            W_pos = torch.empty((q_len, 1))
            nn.init.uniform_(W_pos, -0.02, 0.02)
        elif pe_type == 'zeros':  # n dim random pe
            W_pos = torch.empty((q_len, d_model))
            nn.init.uniform_(W_pos, -0.02, 0.02)
        elif pe_type == 'normal' or pe_type == 'gauss':
            W_pos = torch.zeros((q_len, 1))
            torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
        elif pe_type == 'uniform':
            W_pos = torch.zeros((q_len, 1))
            nn.init.uniform_(W_pos, a=0.0, b=0.1)
        elif pe_type == 'lin1d':
            W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
        elif pe_type == 'exp1d':
            W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
        elif pe_type == 'lin2d':
            W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
        elif pe_type == 'exp2d':
            W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
        elif pe_type == 'sincos':
            W_pos = SinCosPosEncoding(q_len, d_model, normalize=True)
        elif self.__class__ is Positional_Encoding:
            raise ValueError(f"{pe_type} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
            'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
        else:
            W_pos = None
        if W_pos is not None:
            self.W_pos = nn.Parameter(W_pos, requires_grad=learn_pe).to(self.device)

    def forward(self):
        return self.W_pos


"""
    external encoding()
"""


class External_Encoding(nn.Module):
    '''
        External encoding
        output [batch, _, t_seq, embed_dim]
    '''

    def __init__(self, d_model, device):
        super().__init__()
        self.day_embedding = nn.Embedding(7, 64)
        self.time_embedding = nn.Embedding(24 * 12, 64)

    def forward(self, x: Tensor):
        '''
        Args:
            x: [b, #node, #len, 11]
        Output:
            x: [b, #node, #len, 3]
            ext: [b, #node, #len, 64]
        '''
        day_info = torch.argmax(x[..., -7:], dim=-1)
        time_info = (x[..., -8:-7] * 288).int().squeeze(-1)
        x = x[..., :-8]
        # day_ebd = self.day_embedding(day_info)
        time_ebd = self.time_embedding(time_info)
        return x, time_ebd


class S_Positional_Encoding(Positional_Encoding):
    def __init__(self, pe_type, learn_pe, node_num, d_model, dim_red_rate=0.5, device=torch.device('cpu')):
        super(S_Positional_Encoding, self).__init__(pe_type, learn_pe, node_num, d_model, device)
        self.pe_type = pe_type
        if pe_type == 'laplacian':
            self.pe_encoder = LaplacianPE(round(node_num * dim_red_rate), d_model, device=self.device)
        elif pe_type == 'centrality':
            self.pe_encoder = CentralityPE(node_num, d_model)
        else:
            raise ValueError(f"{pe_type} is not a valid spatial pe (positional encoder. Available types: 'laplacian','centrality','gauss'=='normal', \
            'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")

    def forward(self, adj_mx=None):
        if self.pe_type == 'laplacian':
            return self.pe_encoder(adj_mx)
        elif self.pe_type == 'centrality':
            return self.pe_encoder(adj_mx)
        else:
            return self.W_pos.to(self.device)


class LaplacianPE(nn.Module):  # from [Dwivedi and Bresson, 2020] code from PDformer
    def __init__(self, lape_dim, embed_dim, learn_pe=False, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.lape_dim = lape_dim
        self.learn_pe = learn_pe
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def _calculate_normalized_laplacian(self, adj):
        adj = sp.coo_matrix(adj)
        d = np.array(adj.sum(1))
        isolated_point_num = np.sum(np.where(d, 0, 1))
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        return normalized_laplacian, isolated_point_num

    def _calculate_random_walk_laplacian(self, adj):
        adj = sp.coo_matrix(adj)
        d = np.array(adj.sum(1))
        isolated_point_num = np.sum(np.where(d, 0, 1))
        d_inv = np.power(d, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        random_walk_mx = sp.eye(adj.shape[0]) - d_mat_inv.dot(adj).tocoo()
        return random_walk_mx, isolated_point_num

    def _cal_lape(self, dense_adj_mx):
        L, isolated_point_num = self._calculate_normalized_laplacian(dense_adj_mx)
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort()
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

        laplacian_pe: Tensor = torch.from_numpy(
            EigVec[:, isolated_point_num + 1: self.lape_dim + isolated_point_num + 1]
        ).float().to(self.device)
        laplacian_pe.require_grad = self.learn_pe
        return laplacian_pe

    def forward(self, adj_mx):
        lap_mx = self._cal_lape(adj_mx)
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx)
        return lap_pos_enc


class CentralityPE(nn.Module):  # from Graphormer
    """
        for link (unweight) graph
    """

    def __init__(self, num_node, embed_dim, device=torch.device('cpu'), ):
        super().__init__()
        self.device = device
        self.max_in_degree = num_node + 1
        self.max_out_degree = num_node + 1
        self.in_degree_encoder = nn.Embedding(self.max_in_degree, embed_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(self.max_out_degree, embed_dim, padding_idx=0)

    def forward(self, dense_adj_mx):
        (edge_index, _) = dense_to_sparse(torch.from_numpy(dense_adj_mx))
        outdegree = degree(edge_index[0]).to(self.device)
        indegree = degree(edge_index[1]).to(self.device)
        cen_pos_en = self.in_degree_encoder(indegree.long()) + self.out_degree_encoder(outdegree.long())
        return cen_pos_en


class Mask_Bias_Generator():
    """
        mask_heads_share : [True: shape (q,k,h)
                            False:  shape (q,k) or (t/n,q,k)]

    """

    def __init__(self, q_size, v_size):
        self.q_size = q_size
        self.v_size = v_size
        self._bias = None
        self._mask = None

    def get(self):
        return self._bias or self._mask


class Graph_Mask_Generator(Mask_Bias_Generator):
    """
        mask_heads_share: True, single graph, False, Multi-relation graph
        graph
    """

    def __init__(self, num_node, graph_data):
        super(Graph_Mask_Generator, self).__init__(num_node, num_node)
        self.num_node = num_node
        if len(graph_data) == len(graph_data[0]) and type(graph_data) == np.ndarray:  # dense
            dense_adj_mx = graph_data
        else:  # edge index
            dense_adj_mx = to_dense_adj(edge_index=graph_data, max_num_nodes=self.num_node)
        assert (
                dense_adj_mx.shape[0] == self.q_size and dense_adj_mx.shape[1] == self.v_size
        ), "Wrong adj matrix"
        dense_adj_mx = torch.from_numpy(graph_data)
        out0 = dense_adj_mx > 0
        out1 = torch.where(dense_adj_mx == torch.inf,
                           torch.tensor([False, ]).expand(dense_adj_mx.shape),
                           torch.tensor([True, ]).expand(dense_adj_mx.shape))
        self._mask = out0 * out1


class Infomap_Multi_Mask_Generator(Mask_Bias_Generator):
    # get masks shaped as [q x k x h]
    def __init__(self, num_node, graph_data):
        super(Infomap_Multi_Mask_Generator, self).__init__(num_node, num_node)
        self.im = Infomap(silent=True, num_trials=20)
        self.num_node = num_node
        if type(graph_data) is (nx.DiGraph or nx.Graph):
            self.G = graph_data
        else:
            self.G = nx.DiGraph(graph_data)  # dense_adj_mx
        self.im.add_networkx_graph(self.G)
        self._gen_mask()

    def _gen_mask(self):
        self.im.run()
        im = self.im
        num_levels = im.num_levels
        max_num_module = im.num_leaf_modules
        self.num_mask = num_levels - 1
        masks = list()
        for each_level in range(1, num_levels):
            itr = im.get_nodes(depth_level=each_level)
            # clu_tag = torch.zeros([self.num_node,],dtype=torch.int)
            # clu_tags = torch.zeros([max_num_module,self.num_node],dtype=torch.int)
            clu_tags = torch.full([max_num_module, self.num_node], -1, dtype=torch.int)
            # ind = torch.zeros([self.num_node,],dtype=torch.int)
            for each in itr:
                # 一个行复制，一个列复制，用两个矩阵where相等为True,overlap部分无法处理(mend)
                clu_tags[each.module_id - 1][each.node_id] = 1
            temp1 = clu_tags.unsqueeze(2).expand([max_num_module, self.num_node, self.num_node])
            temp2 = temp1.transpose(1, 2)
            out = torch.any((temp1 == temp2) * (temp1 != -1), dim=0)
            masks.append(out)
        masks = torch.stack(masks, dim=2)
        self._mask = masks


def get_static_multihead_mask(num_head, mask_generator_list: list, device=torch.device('cpu')):
    all_mask = list()
    for each_mg in mask_generator_list:
        temp_mask: Tensor = each_mg.get()
        if len(temp_mask.shape) == 2:
            temp_mask = temp_mask.unsqueeze(dim=-1)
        assert (
                len(temp_mask.shape) == 3
        ), "Unaccpetable static multihead mask"
        all_mask.append(temp_mask)
    all_mask = torch.cat(all_mask, dim=-1)
    assert (
            all_mask.shape[2] < num_head
    ), "Not enough multihead num"
    all_true_mask = torch.full(
        [all_mask.shape[0], all_mask.shape[1], num_head - all_mask.shape[2]], True)
    all_mask = torch.cat([all_mask, all_true_mask], dim=-1).contiguous().to(device)
    return all_mask


class MixhopConv(nn.Module):
    def __init__(self, gdep=3, alpha=0):
        super(MixhopConv, self).__init__()
        # self.mlp = nn.Linear((gdep+1)*c_in, c_out)
        self.gdep = gdep
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        adj = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h1 = torch.einsum('bntc,nm->bmtc', (h, adj))
            # h1 = torch.einsum('ncwl,vw->ncvl', (h, adj))
            # batch_size (bs) * node_num x time_seq_len x input_channel
            h = self.alpha * x + (1 - self.alpha) * h1
            out.append(h)
        ho = torch.cat(out, dim=-1)
        # ho = self.mlp(ho)
        return ho


class _ST_Attention(nn.Module):
    def __init__(self, type, embed_dim, num_heads, scale=None,
                 mask_flag=False, bias_flag=False, key_missing_mask_flag=False,
                 attention_dropout=0.1, output_attention=False, proj_bias=True):
        """
        Input shape:
            Q:       [batch_size (bs) x node_num x max_time_seq_len x embed_dim]
            K, V:    [batch_size (bs) x node_num x time_seq_len x embed_dim]
            mask:    [[t/n] x q_len x q_len x head_num] # dtype=torch.bool, [False] means masked/unseen attention
            bias/rencoding: [[t/n] x q_len x q_len x [head_num]]
            key_missing_mask_flag : [bs  x node_num x out_seq_len]

        Paramaters:
            miss_mask_flag: whether to mask missing value is ST data, refer to key_padding_mask
            scale={
                'lsa': learnable scale
                None: default
                else: given scale
            }
            attention_dropout: equals randomly attention mask

        Output shape:
            attention_weight/attention_score:  bnqkh or bqkth
            out:   as Q
        """
        self.mask_flag = mask_flag
        self.bias_flag = bias_flag
        self.key_missing_mask_flag = key_missing_mask_flag
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == embed_dim
        ), "Embedding dim needs to be divisible by num_heads"
        super(_ST_Attention, self).__init__()

        if scale == 'lsa':
            self.scale = nn.Parameter(torch.tensor(self.head_dim ** -0.5), requires_grad=True)
        else:
            self.scale = scale if scale is not None else 1. / math.sqrt(embed_dim)

        self.type = type
        self.output_attention = output_attention
        self.attn_dropout = nn.Dropout(attention_dropout)

        # full project O(d^2)
        self.values_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=proj_bias)
        self.keys_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=proj_bias)
        self.queries_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=proj_bias)

        self.att_dropout_layer = nn.Dropout(attention_dropout)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_dim)

    def forward(self, value: Tensor, key: Tensor, query: Tensor,
                attn_mask: Optional[Tensor] = None, attn_bias: Optional[Tensor] = None,
                key_missing_mask: Optional[Tensor] = None):

        batch_size, num_nodes, input_window, embed_dim = query.shape

        if self.mask_flag:
            assert (
                    attn_mask is not None
            ), "Require available mask!"
            attn_mask = ~attn_mask

        # full project
        value = self.values_proj(value)
        key = self.keys_proj(key)
        query = self.queries_proj(query)
        value = value.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)
        query = query.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)

        # Spatial attention
        if self.type == 'S':
            attention_score = torch.einsum("bqthd,bkthd->bqkth", [query, key])
            # masking & relative position enocding
            if self.mask_flag:
                attention_score = attention_score.permute(0, 3, 1, 2, 4)  # btqkh
                # masked_fill_ [True] means masked/unseen attention
                attention_score.masked_fill_(attn_mask, -1e10)
                attention_score = attention_score.permute(0, 2, 3, 1, 4)

            if self.bias_flag:
                attention_score = attention_score.permute(0, 3, 1, 2, 4)  # btqkh
                attention_score += attn_bias
                attention_score = attention_score.permute(0, 2, 3, 1, 4)

            if self.key_missing_mask_flag and key_missing_mask is not None:
                # bqkth bkt -> b1kt-> b1kt1
                attention_score.masked_fill_(key_missing_mask.unsqueeze(1).unsqueeze(-1), -1e10)

            attention_weight = torch.softmax(attention_score * self.scale, dim=2)
            attention_weight = self.attn_dropout(attention_weight)
            out = torch.einsum("bqkth,bkthd->bqthd", [attention_weight, value]).reshape(
                batch_size, num_nodes, input_window, self.num_heads * self.head_dim
            )
        elif self.type == 'T':
            attention_score = torch.einsum("bnqhd,bnkhd->bnqkh", [query, key])
            # masking & relative position enocding
            if self.mask_flag:
                attention_score.masked_fill_(attn_mask, -1e10)
            if self.bias_flag:
                attention_score += attn_bias
            if self.key_missing_mask_flag and key_missing_mask is not None:
                # bnqkh bnk -> bn1k1
                attention_score.masked_fill_(key_missing_mask.unsqueeze(2).unsqueeze(-1), -1e10)

            attention_weight = torch.softmax(attention_score * self.scale, dim=3)
            attention_weight = self.attn_dropout(attention_weight)
            out = torch.einsum("bnqkh,bnkhd->bnqhd", [attention_weight, value]).reshape(
                batch_size, num_nodes, input_window, self.num_heads * self.head_dim
            )
        # nan secure
        out = torch.where(torch.isnan(out), Tensor([0, ]).to(out.device), out)
        out = self.fc_out(out)
        if self.output_attention:
            return out, attention_score, attention_weight
        else:
            return out


class _ST_Transfomer(nn.Module):
    def __init__(self, type, embed_dim, num_heads, norm='BatchNorm', scale=None,
                 mask_flag=False, bias_flag=False, key_missing_mask_flag=False,
                 attention_dropout=0.1, proj_dropout=0.1,
                 ffn_forward_expansion=4, activation_fn=nn.ReLU, pre_norm=False, store_attn=False):
        super(_ST_Transfomer, self).__init__()
        self.norm = norm
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == embed_dim
        ), "Embedding dim needs to be divisible by num_heads"

        ## add absolute position encoding! # same shape as query

        ### add attention module
        self.attention = _ST_Attention(type=type, embed_dim=embed_dim, num_heads=num_heads, scale=scale,
                                       mask_flag=mask_flag, bias_flag=bias_flag,
                                       key_missing_mask_flag=key_missing_mask_flag,
                                       attention_dropout=attention_dropout, output_attention=store_attn)

        ### add normalized layer/ffn
        self.drop_attn = nn.Dropout(proj_dropout)
        self.norm_attn = Norm(self.norm, self.embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ffn_forward_expansion * embed_dim),
            nn.Dropout(proj_dropout),
            activation_fn(),
            nn.Linear(ffn_forward_expansion * embed_dim, embed_dim),
        )
        self.norm_ffn = Norm(self.norm, self.embed_dim)
        self.dropout_ffn = nn.Dropout(proj_dropout)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, value, key, query, attn_mask=None, attn_bias=None, key_missing_mask=None):
        # query = x + pe
        if self.store_attn:
            x1, attention_score, attention_weight = self.attention(value=value, key=key, query=query,
                                                                   attn_mask=attn_mask, attn_bias=attn_bias,
                                                                   key_missing_mask=key_missing_mask)
        else:
            x1 = self.attention(value=value, key=key, query=query,
                                attn_mask=attn_mask, attn_bias=attn_bias, key_missing_mask=key_missing_mask)
        x = query + self.drop_attn(x1)
        if not self.pre_norm:
            x = self.norm_attn(x)
        # self.dropout_layer(self.norm1(attention + query))
        if self.pre_norm:
            x = self.norm_ffn(x)
        x1 = self.feed_forward(x)
        ## Add & Norm
        x = x + self.dropout_ffn(x1)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            x = self.norm_ffn(x)
        if self.store_attn:
            return x, attention_score, attention_weight
        else:
            return x


class STBlock(nn.Module):
    """
        STencoder block with 1 Stransformer and 1 Ttransformer.
        Args:
            mode: different inner structure

    """

    def __init__(self, seq_len, node_num, embed_dim, num_heads, forward_mode=0,
                 norm='BatchNorm', scale=None,
                 global_nodePE=None, global_tseqPE=None,
                 smask_flag=False, sbias_flag=False,
                 tmask_flag=False, tbias_flag=False, key_missing_mask_flag=False,
                 attention_dropout=0.1, proj_dropout=0.1, activation_fn=nn.ReLU, pre_norm=False, sstore_attn=False):
        super(STBlock, self).__init__()
        self.smask_flag = smask_flag
        self.sbias_flag = sbias_flag
        self.tmask_flag = tmask_flag
        self.tbias_flag = tbias_flag
        self.key_missing_mask_flag = key_missing_mask_flag
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_forward_expansion = 4

        self.forward_mode = forward_mode
        # self.nodePE = global_nodePE if global_nodePE is not None else S_Positional_Encoding('laplacian', False, node_num, embed_dim)
        # self.tseqPE  = global_tseqPE if global_tseqPE is not None else Positional_Encoding('sincos', False, seq_len, embed_dim)

        self.STransformer = _ST_Transfomer(type='S', embed_dim=embed_dim, num_heads=num_heads, norm=norm, scale=scale,
                                           mask_flag=smask_flag, bias_flag=sbias_flag,
                                           key_missing_mask_flag=key_missing_mask_flag,
                                           attention_dropout=attention_dropout, proj_dropout=proj_dropout,
                                           activation_fn=activation_fn,
                                           ffn_forward_expansion=self.ffn_forward_expansion, pre_norm=pre_norm,
                                           store_attn=sstore_attn)
        self.TTransformer = _ST_Transfomer(type='T', embed_dim=embed_dim, num_heads=num_heads, norm=norm, scale=scale,
                                           mask_flag=tmask_flag, bias_flag=tbias_flag,
                                           key_missing_mask_flag=key_missing_mask_flag,
                                           attention_dropout=attention_dropout, proj_dropout=proj_dropout,
                                           activation_fn=activation_fn,
                                           ffn_forward_expansion=self.ffn_forward_expansion, pre_norm=pre_norm,
                                           store_attn=False)

        self.norm1 = Norm(norm, self.embed_dim)
        self.norm2 = Norm(norm, self.embed_dim)
        self.dropout_layer = nn.Dropout(proj_dropout)
        self.sstore_attn = sstore_attn

    def forward(self, x, dense_adj_mx, npe=None, tpe=None, sattn_mask=None, sattn_bias=None, tattn_mask=None,
                tattn_bias=None):  # bntc
        # npe = npe if npe is not None else self.nodePE(dense_adj_mx).reshape(1,-1,1,self.embed_dim).contiguous()
        # tpe = tpe if tpe is not None else self.tseqPE().reshape(1,1,-1,self.embed_dim).contiguous()
        npe = npe if npe is not None else 0
        tpe = tpe if tpe is not None else 0
        if self.forward_mode == 0:
            x1 = self.norm1(
                self.TTransformer(value=x, key=x, query=x + npe,
                                  attn_mask=tattn_mask, attn_bias=tattn_bias) + x)
            if self.sstore_attn:
                xtemp, attention_score, attention_weight = self.STransformer(value=x1, key=x1, query=x1 + tpe,
                                                                             attn_mask=sattn_mask, attn_bias=sattn_bias)
            else:
                xtemp = self.STransformer(value=x1, key=x1, query=x1 + tpe,
                                          attn_mask=sattn_mask, attn_bias=sattn_bias)
            out = self.dropout_layer(self.norm2(xtemp + x1))
        if self.sstore_attn:
            return out, attention_score, attention_weight
        else:
            return out

# tested
class patching_conv(nn.Module):
    """
    Input/Output shape:
        input: [batch_size (bs) * node_num x time_seq_len x input_channel]
        output: [batch_size (bs) * node_num x patch_num x embed_dim(out_channel*kernel_size)]
    """

    def __init__(self, in_channel, embed_dim, in_seq_len, kernel_sizes: list = [1, 2, 3, 6], stride=1,
                 activation_fn=nn.Tanh):
        super(patching_conv, self).__init__()
        self.kernel_num = len(kernel_sizes)
        assert (
                embed_dim % self.kernel_num == 0
        ), "Embedding dim needs to be divisible by kernel_size"
        self.kernel_size = kernel_sizes
        self.in_channel = in_channel
        self.out_channel = embed_dim // self.kernel_num
        self.embed_dim = embed_dim
        self.in_seq_len = in_seq_len
        self.out_seq_len = math.ceil(in_seq_len / stride)

        # pad seq for unified patch_num / len(shape)<=3
        self.paddings = nn.ModuleList([
            nn.ReplicationPad1d((round((ks - 1) / 2), (ks - 1) - round((ks - 1) / 2)))
            for ks in kernel_sizes
        ])
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=ks, stride=stride)
            for ks in kernel_sizes
        ])
        self.activation = activation_fn()

    def _t_patch_reshape(self, batch_size, node_num, x: Tensor, mode=0):
        if mode == 0:
            return x.view(batch_size * node_num, x.shape[2], x.shape[3])
        else:
            return x.reshape(batch_size, node_num, x.shape[2], x.shape[3])

    def forward(self, x: Tensor):
        batch_size, node_num, t_len, in_channel = x.shape
        x = x.view([-1, t_len, in_channel])
        x = x.permute(0, 2, 1)  # b n i t
        out = list()
        for i in range(self.kernel_num):
            xi = self.paddings[i](x)
            xi = self.convs[i](xi)  # b nn ed pn
            xi = xi.permute(0, 2, 1)
            out.append(xi)
        out = torch.cat(out, dim=-1)
        out = out.reshape([batch_size, node_num, -1, self.embed_dim]).contiguous()
        out = self.activation(out)
        return out


class patching_STconv(nn.Module):
    """
    adding k timeseq filter and a/k graph filter
    change to MixhopConv
    Input/Output shape:
        input: [batch_size (bs) x node_num x time_seq_len x input_channel]
        output: [batch_size (bs) x node_num x patch_num x embed_dim(out_channel*kernel_size)]
    """

    def __init__(self, in_channel, embed_dim, in_seq_len, kernel_sizes: list = [1, 2, 3, 6], stride=1,
                 gdep=3, alpha=0, norm='BatchNorm',
                 activation_fn=nn.Tanh, device=torch.device('cpu')):
        super(patching_STconv, self).__init__()
        self.device = device
        self.kernel_num = len(kernel_sizes)
        assert (
                embed_dim % (self.kernel_num * (gdep + 1)) == 0
        ), "Embedding dim needs to be divisible by kernel_size"
        self.kernel_size = kernel_sizes
        self.in_channel = in_channel
        self.out_channel = embed_dim // (self.kernel_num * (gdep + 1))
        self.embed_dim = embed_dim
        self.in_seq_len = in_seq_len
        self.out_seq_len = math.ceil(in_seq_len / stride)
        self.gdep = gdep
        self.alpha = alpha
        self.norm = norm

        # pad seq for unified patch_num / len(shape)<=3
        self.paddings = nn.ModuleList([
            nn.ReplicationPad1d((round((ks - 1) / 2), (ks - 1) - round((ks - 1) / 2)))
            for ks in kernel_sizes
        ])
        self.tconvs = nn.ModuleList([
            nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=ks, stride=stride)
            for ks in kernel_sizes
        ])
        self.gconv = MixhopConv(gdep=self.gdep, alpha=self.alpha)
        self.norm = Norm(self.norm, self.embed_dim)
        self.activation = activation_fn()

    def forward(self, x: Tensor, dense_adj_mx):
        # edge_index,edge_weight = dense_to_sparse(torch.from_numpy(dense_adj_mx))
        # edge_index,edge_weight = edge_index.to(self.device),edge_weight.to(self.device)
        batch_size, node_num, t_len, in_channel = x.shape
        x = x.view([-1, t_len, in_channel])
        x = x.permute(0, 2, 1)  # b*n i t
        out = list()
        for i in range(self.kernel_num):
            # timeseq pattern extraction
            xi: Tensor = self.paddings[i](x)
            xi = self.tconvs[i](xi)  # b*n ed pn
            xi = xi.permute(0, 2, 1)
            xi = xi.reshape([batch_size, node_num, -1, self.out_channel]).contiguous()
            # neighborhood pattern extraction
            out.append(xi)
        out = torch.cat(out, dim=-1)
        # out = out.permute(0,2,1,3).contiguous()
        # out = self.gconv(out,edge_index,edge_weight) # b t n c
        out = self.gconv(out, torch.from_numpy(dense_adj_mx).to(self.device))
        # out = out.permute(0,2,1,3)
        out = self.activation(out)
        return out


class Permution(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x:Tensor):
        if self.contiguous: return x.permute(*self.dims).contiguous()
        else: return x.permute(*self.dims)

def Norm(norm,embed_dim):
    if "batch" in norm.lower():
        return nn.Sequential(Permution(0,3,1,2), nn.BatchNorm2d(embed_dim), Permution(0,2,3,1))
    else:
        return nn.LayerNorm(embed_dim)


class depatching_conv(nn.Module):
    """
    depatch conv transpose layer with linear decoder
    """

    def __init__(self, embed_dim, unpatch_channel, out_channel, hid_seq_len, out_seq_len, kernal_size=None, stride=None,
                 activation_fn=nn.Tanh):
        super(depatching_conv, self).__init__()
        self.embed_dim = embed_dim
        self.unpatch_channel = unpatch_channel
        self.out_channel = out_channel
        self.in_len = hid_seq_len
        self.out_len = out_seq_len
        self.stride = stride or math.ceil(self.out_len / self.in_len)
        self.kernal = kernal_size or math.ceil(self.out_len / self.in_len)
        assert (
                self.kernal >= self.stride
        ), "Bad kernal size"
        # self.unpatch_seq_len = self.stride * (self.in_len+self.kernal-1)+self.stride-self.kernal
        self.unpatch_seq_len = self.stride * self.in_len

        self.padding = nn.ReplicationPad1d(
            (round((self.kernal - 1) / 2), (self.kernal - 1) - round((self.kernal - 1) / 2)))
        self.tconv = nn.ConvTranspose1d(in_channels=self.embed_dim, out_channels=self.unpatch_channel,
                                        kernel_size=self.kernal, stride=self.stride)
        self.seqlin = nn.Sequential(
            # in [b,n,patch_seq_len,embed_dim]
            # out [b,n,out_seq_len,b,n,out_dim]
            nn.Linear(self.unpatch_seq_len, self.out_len),
            Permution(0, 2, 1),
            nn.Linear(self.unpatch_channel, self.out_channel),
        )
        self.activation = activation_fn()

    def forward(self, x: Tensor):
        batch_size, node_num, t_len, in_channel = x.shape
        xt = x.reshape([-1, t_len, in_channel]).contiguous()
        xt = xt.permute(0, 2, 1)  # b*n c t
        xt = self.padding(xt)
        xt = self.tconv(xt)
        if round(((self.kernal - 1) * self.stride) / 2) - ((self.kernal - 1) * self.stride) == 0:
            xt = xt[..., round(((self.kernal - 1) * self.stride + (self.kernal - self.stride)) / 2):]
        else:
            xt = xt[..., round(((self.kernal - 1) * self.stride + (self.kernal - self.stride)) / 2): \
                         round(((self.kernal - 1) * self.stride + (self.kernal - self.stride)) / 2) - (
                                     (self.kernal - 1) * self.stride + (self.kernal - self.stride))]
        xt = self.activation(xt)
        xt = self.seqlin(xt)
        xt = xt.reshape([batch_size, node_num, -1, self.out_channel]).contiguous()

        return xt


class MultiSPANS(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._scaler = self.data_feature.get('scaler')
        self.adj_mx = data_feature.get('adj_mx')
        self.feature_dim = self.data_feature.get("feature_dim", 1)
        outfeat_dim = config.get('outfeat_dim', None)
        self.output_dim = outfeat_dim if outfeat_dim is not None else self.data_feature.get('output_dim', 1)
        self.num_nodes = self.data_feature.get("num_nodes", 1)
        self.load_external = config.get('load_external', False)
        if self.load_external:
            self.feature_dim -= 8
        self._logger = getLogger()

        self.device = config.get('device', torch.device('cpu'))
        self.embed_dim = config.get('embed_dim', 64)
        self.skip_conv_flag = config.get('skip_conv_flag', True)
        self.residual_conv_flag = config.get('residual_conv_flag', True)
        self.skip_dim = config.get('skip_dim', self.embed_dim)
        self.num_layers = config.get('num_layers', 3)
        self.num_heads = config.get('num_heads', 8)
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get('output_window', 12)

        self.gconv_hop_num = config.get('gconv_hop_num', 3)
        self.gconv_alpha = config.get('gconv_alpha', 0)

        self.conv_kernels = config.get('conv_kernels', [2, 3, 6, 12])
        self.conv_stride = config.get('conv_stride', 1)
        self.conv_if_gc = config.get('conv_if_gc', False)

        self.norm_type = config.get('norm_type', 'BatchNorm')

        self.att_scale = config.get('att_scale', None)
        self.att_dropout = config.get('att_dropout', 0.1)
        self.ffn_dropout = config.get('ffn_dropout', 0.1)
        self.Spe_type = config.get('Satt_pe_type', 'laplacian')
        self.Spe_learnable = config.get('Spe_learnable', False)
        self.Tpe_type = config.get('Tatt_pe_type', 'sincos')
        self.Tpe_learnable = config.get('Tpe_learnable', False)
        self.Smask_flag = config.get('Smask_flag', True)
        self.block_forward_mode = config.get('block_forward_mode', 0)
        self.sstore_attn = config.get('sstore_attn', False)
        # static parameters
        self.activition_fn = nn.ReLU

        if self.skip_conv_flag is False:
            self.skip_dim = self.embed_dim
        self.patchencoder = patching_STconv(
            in_channel=self.feature_dim, embed_dim=self.embed_dim,
            in_seq_len=self.input_window,
            gdep=self.gconv_hop_num, alpha=self.gconv_alpha,
            kernel_sizes=self.conv_kernels, stride=self.conv_stride, device=self.device
        ) if self.conv_if_gc else patching_conv(
            in_channel=self.feature_dim, embed_dim=self.embed_dim,
            in_seq_len=self.input_window, kernel_sizes=self.conv_kernels, stride=self.conv_stride
        )
        self.hid_seq_len = self.patchencoder.out_seq_len
        if self.Smask_flag:
            self.infomask = Infomap_Multi_Mask_Generator(self.num_nodes, self.adj_mx)
            self.graphmask = Graph_Mask_Generator(self.num_nodes, self.adj_mx)
        self.externalPEencoder = External_Encoding(d_model=self.embed_dim, device=self.device)
        self.nodePEencoder = S_Positional_Encoding(
            pe_type=self.Spe_type, learn_pe=self.Spe_learnable, node_num=self.num_nodes,
            d_model=self.embed_dim, device=self.device)
        self.tseqPEencoder = Positional_Encoding(
            pe_type=self.Tpe_type, learn_pe=self.Tpe_learnable, q_len=self.hid_seq_len,
            d_model=self.embed_dim, device=self.device)
        self.STencoders = nn.ModuleList(
            [STBlock(
                seq_len=self.hid_seq_len, node_num=self.num_nodes, embed_dim=self.embed_dim, num_heads=self.num_heads,
                forward_mode=self.block_forward_mode, norm=self.norm_type, scale=self.att_scale,
                global_nodePE=self.nodePEencoder, global_tseqPE=self.tseqPEencoder, smask_flag=self.Smask_flag,
                sbias_flag=False,
                tmask_flag=False, tbias_flag=False, key_missing_mask_flag=False,
                attention_dropout=self.att_dropout, proj_dropout=self.ffn_dropout, activation_fn=self.activition_fn,
                pre_norm=False, sstore_attn=self.sstore_attn
            ) for _ in range(self.num_layers)]
        )

        if self.skip_conv_flag:
            self.skip_convs = nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.embed_dim, out_channels=self.skip_dim, kernel_size=1,
                ) for _ in range(self.num_layers + 1)
            ])

        if self.residual_conv_flag:
            self.residual_convs = nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=1,
                ) for _ in range(self.num_layers)
            ])

        self.lineardecoder = depatching_conv(embed_dim=self.skip_dim, unpatch_channel=self.skip_dim // 2,
                                             out_channel=self.output_dim,
                                             hid_seq_len=self.hid_seq_len, out_seq_len=self.output_window)

        self.droput_layer = nn.Dropout(p=self.ffn_dropout)

    def forward(self, batch):
        dense_adj_mx = self.adj_mx
        if self.Smask_flag:
            multimask = get_static_multihead_mask(self.num_heads, [self.infomask, self.graphmask], device=self.device)
        else:
            multimask = None
        npe = self.nodePEencoder(dense_adj_mx).reshape(1, -1, 1, self.embed_dim).contiguous()
        tpe = self.tseqPEencoder().reshape(1, 1, -1, self.embed_dim).contiguous()
        x = batch['X'].permute(0, 2, 1, 3).contiguous()  # btnc -> bntc
        if self.load_external:
            x, epe = self.externalPEencoder(x)
            npe, tpe = npe + epe, tpe + epe
        if self.conv_if_gc:
            x = self.patchencoder(x, dense_adj_mx)
        else:
            x = self.patchencoder(x)  # [b,n,patch_seq_len,embed_dim]

        skip = self.skip_convs[-1](x.permute(0, 3, 2, 1)) if self.skip_conv_flag else x
        if self.sstore_attn:
            for i, block in enumerate(self.STencoders):
                h, attention_score, attention_weight = block(x, dense_adj_mx, npe, tpe,
                                                             sattn_mask=multimask)  # [b,n,patch_seq_len,embed_dim]
                skip = skip + self.skip_convs[i](h.permute(0, 3, 2, 1)) if self.skip_conv_flag else skip + h
                x = self.residual_convs[i](x.permute(0, 3, 2, 1)).permute(0, 3, 2,
                                                                          1) + h if self.residual_conv_flag else x + h
                if self.training is not True:
                    import time
                    t = time.localtime()
                    torch.save({'attention_score': attention_score, 'attention_weight': attention_weight},
                               "./attn_save/{}_att.pt".format(time.strftime("%d_%H_%M_%S", t)))

        else:
            for i, block in enumerate(self.STencoders):
                h = block(x, dense_adj_mx, npe, tpe, sattn_mask=multimask)  # [b,n,patch_seq_len,embed_dim]
                skip = skip + self.skip_convs[i](h.permute(0, 3, 2, 1)) if self.skip_conv_flag else skip + h
                x = self.residual_convs[i](x.permute(0, 3, 2, 1)).permute(0, 3, 2,
                                                                          1) + h if self.residual_conv_flag else x + h
        skip = skip.permute(0, 3, 2, 1) if self.skip_conv_flag else skip
        # out = torch.sum(torch.stack(skips))
        skip = self.droput_layer(skip)
        out = self.lineardecoder(skip).permute(0, 2, 1, 3).contiguous()
        return out

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true)

    def predict(self, batch):
        return self.forward(batch)