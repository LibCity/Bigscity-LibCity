import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from libtraffic.model import loss
from libtraffic.model.abstract_traffic_state_model import AbstractTrafficStateModel
from logging import getLogger


"""
模型估计只能跑自己的数据集，但是数据集没有开源
"""


class SpGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, ret_adj=False, pa_prop=False):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pa_prop = pa_prop
        self.w_key = nn.Linear(in_features, out_features, bias=True)
        self.w_value = nn.Linear(in_features, out_features, bias=True)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.cosinesimilarity = nn.CosineSimilarity(dim=-1, eps=1e-8)

    def edge_attention(self, edges):
        # edge UDF
        # dot-product attention
        att_sim = torch.sum(torch.mul(edges.src['h_key'], edges.dst['h_key']), dim=-1)
        #         att_sim = self.cosinesimilarity(edges.src['h_key'], edges.dst['h_key'])
        return {'att_sim': att_sim}

    def message_func(self, edges):
        # message UDF
        return {'h_value': edges.src['h_value'], 'att_sim': edges.data['att_sim']}

    def reduce_func(self, nodes):
        # reduce UDF
        alpha = F.softmax(nodes.mailbox['att_sim'], dim=1)  # (# of nodes, # of neibors)
        alpha = alpha.unsqueeze(-1)
        h_att = torch.sum(alpha * nodes.mailbox['h_value'], dim=1)
        return {'h_att': h_att}

    def forward(self, X_key, X_value, g):
        """
        :param X_key: X_key data of shape (batch_size(B), num_nodes(N), in_features_1).
        :param X_value: X_value dasta of shape (batch_size, num_nodes(N), in_features_2).
        :param g: sparse graph.
        :return: Output data of shape (batch_size, num_nodes(N), out_features).
        """
        B, N, in_features = X_key.size()
        h_key = self.w_key(X_key)  # (B,N,out_features)
        h_key = h_key.view(B * N, -1)  # (B*N,out_features)
        h_value = X_value if (self.pa_prop == True) else self.w_value(X_value)
        h_value = h_value.view(B * N, -1)
        g.ndata['h_key'] = h_key
        g.ndata['h_value'] = h_value
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        h_att = g.ndata.pop('h_att').view(B, N, -1)  # (B,N,out_features)
        h_conv = h_att if (self.pa_prop == True) else self.leakyrelu(h_att)
        return h_conv


class GAT(nn.Module):
    def __init__(self, in_feat, nhid=32, dropout=0, alpha=0.2, hopnum=2, pa_prop=False):
        """sparse GAT."""
        super(GAT, self).__init__()
        self.pa_prop = pa_prop
        self.dropout = nn.Dropout(dropout)
        if (pa_prop == True): hopnum = 1
        print('hopnum_gat:', hopnum)
        self.gat_stacks = nn.ModuleList()
        for i in range(hopnum):
            if (i > 0): in_feat = nhid
            att_layer = SpGraphAttentionLayer(in_feat, nhid, dropout=dropout, alpha=alpha, pa_prop=pa_prop)
            self.gat_stacks.append(att_layer)

    def forward(self, X_key, X_value, adj):
        out = X_key
        for att_layer in self.gat_stacks:
            if self.pa_prop:
                out = att_layer(out, X_value, adj)
            else:
                out = att_layer(out, out, adj)
        return out


class SCConv(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, latend_num, gcn_hop):
        super(SCConv, self).__init__()
        self.in_features = in_features
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.conv_block_after_pool = GCN(in_features=self.in_features, out_features=out_features, \
                                         dropout=dropout, alpha=alpha, hop=gcn_hop)
        self.w_classify = nn.Linear(self.in_features, latend_num, bias=True)

    def apply_bn(self, x):
        # Batch normalization of 3D tensor x
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        x = bn_module(x)
        return x

    def forward(self, X_lots, adj):
        """
        :param X_lots: Concat of the outputs of CxtConv and PA_approximation (batch_size, N, in_features).
        :param adj: adj_merge (N, N).
        :return: Output soft clustering representation for each parking lot of shape (batch_size, N, out_features).
        """
        B, N, in_features = X_lots.size()
        h_now = self.dropout(X_lots)  # (B, N, F)
        S = self.w_classify(h_now)  # (B, N, latend_num(K))
        S = F.softmax(S, dim=-1)  # (B, N, K)
        h_c = torch.bmm(S.permute(0, 2, 1), h_now)  # (B, K, F)
        h_c = self.apply_bn(h_c)
        adj = torch.bmm(torch.bmm(S.permute(0, 2, 1), adj), S)  # (B, K, K)
        # GCN
        h_latent = self.dropout(self.conv_block_after_pool(h_c, adj))  # (B, K, F)
        h_sc = torch.bmm(S, h_latent)
        return h_sc


class GCN(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, hop=1):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.hop = hop
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.w_lot = nn.ModuleList()
        for i in range(hop):
            in_features = self.in_features if (i == 0) else out_features
            self.w_lot.append(nn.Linear(in_features, out_features, bias=True))

    def forward(self, h_c, adj):
        # adj normalize
        adj_rowsum = torch.sum(adj, dim=-1, keepdim=True)
        adj = adj.div(torch.where(adj_rowsum > 1e-8, adj_rowsum, 1e-8 * torch.ones(1, 1).cuda()))  # row normalize
        # weight aggregate
        for i in range(self.hop):
            h_c = torch.bmm(adj, h_c)
            h_c = self.leakyrelu(self.w_lot[i](h_c))  # (B, N, F)
        return h_c


class FeatureEmb(nn.Module):
    def __init__(self):
        super(FeatureEmb, self).__init__()
        # time embedding
        # month,day,hour,minute,dayofweek
        self.time_emb = nn.ModuleList([nn.Embedding(feature_size, 4) for feature_size in [12, 31, 24, 4, 7]])
        for ele in self.time_emb:
            nn.init.xavier_uniform_(ele.weight.data, gain=math.sqrt(2.0))

    def forward(self, X, pa_onehot):
        B, N, T_in, F = X.size()  # (batch_size, N, T_in, F)
        X_time = torch.cat([emb(X[:, :, :, i + 4].long()) for i, emb in enumerate(self.time_emb)],
                           dim=-1)  # time F = 4*5 = 20
        X_cxt = X[..., 2:4]  # contextual features
        X_pa = X[..., :1].long()  # PA, 0,1,...,49
        pa_scatter = pa_onehot.clone()
        X_pa = pa_scatter.scatter_(-1, X_pa, 1.0)  # discretize to one-hot , F = 50
        return X_cxt, X_pa, X_time


class SHARE(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super(SHARE, self).__init__(config, data_feature)
        self.adj_mx = self.data_feature.get('adj_mx')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 2)
        self.output_dim = self.data_feature.get('output_dim', 2)
        self.ext_dim = self.data_feature.get('ext_dim', 1)
        self._scaler = self.data_feature.get('scaler')
        self._logger = getLogger()

        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        self.t_in = self.input_window
        self.t_out = self.output_window

        self.alpha = config.get('alpha', 0.2)
        self.hid_dim = config.get('hid_dim', 32)
        self.dropout = config.get('dropout', 0.5)
        self.hc_ratio = config.get('hc_ratio', 0.1)
        self.topk = config.get('topk', 10)
        self.disteps = config.get('disteps', 1000)
        self.beta = config.get('beta', 0.5)
        self.gat_hop = config.get('gat_hop', 2)
        self.batch_size = config.get('batch_size', 64)

        self.N = self.num_nodes
        self.train_num = self.N
        self.latend_num = int(self.N * self.hc_ratio + 0.5)
        # number of context features (here set 2 for test)
        self.nfeat = 2
        # Feature embedding
        self.feature_embedding = FeatureEmb()
        # FC layers
        self.output_fc = nn.Linear(self.hid_dim * 2, self.t_out, bias=True)
        self.w_pred = nn.Linear(self.hid_dim * 2, 50, bias=True)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # Spatial blocks
        # CxtConv
        self.CxtConv = GAT(in_feat=self.nfeat, nhid=self.hid_dim, dropout=self.dropout,
                           alpha=self.alpha, hopnum=self.gat_hop, pa_prop=False)
        # PropConv
        self.PropConv = GAT(in_feat=self.nfeat, nhid=self.hid_dim, dropout=self.dropout,
                            alpha=self.alpha, hopnum=1, pa_prop=True)
        # SCConv
        self.SCConv = SCConv(in_features=self.hid_dim + 50, out_features=self.hid_dim,
                             dropout=self.dropout, alpha=self.alpha,
                             latend_num=self.latend_num, gcn_hop=1)

        # GRU Cell
        self.GRU = nn.GRUCell(2 * self.hid_dim + 50 + 20, self.hid_dim * 2, bias=True)
        nn.init.xavier_uniform_(self.GRU.weight_ih, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.GRU.weight_hh, gain=math.sqrt(2.0))

        # Parameter initialization
        for ele in self.modules():
            if isinstance(ele, nn.Linear):
                nn.init.xavier_uniform_(ele.weight, gain=math.sqrt(2.0))

        adjs, h_t, pa_onehot = self.get_adj_ht_pa()
        self.adj = adjs[0]
        self.adj_label = adjs[1]
        self.adj_dense = adjs[2]
        self.h_t = h_t
        self.pa_onehot = pa_onehot

    def spadj_expand(self, adj, batch_size):
        adj = dgl.batch([adj] * batch_size)
        return adj

    def adj_process(self, adj):
        """
        return: sparse CxtConv and sparse PropConv adj
        """
        # sparse context graph adj (2,E)
        edge_1 = []
        edge_2 = []
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if i == j or adj[i, j] <= self.disteps:
                    edge_1.append(i)
                    edge_2.append(j)
        edge_adj = np.asarray([edge_1, edge_2], dtype=int)

        # sparse propagating adj (2,E)
        edge_1 = []
        edge_2 = []
        for i in range(adj.shape[0]):
            cnt = 0
            adj_row = adj[i, :self.train_num]
            adj_row = sorted(enumerate(adj_row), key=lambda x: x[1])  # [(idx,dis),...]
            for j, dis in adj_row:
                if i != j:
                    edge_1.append(i)
                    edge_2.append(j)
                    cnt += 1
                if cnt >= self.topk and dis > self.disteps:
                    break
        adj_label = np.asarray([edge_1, edge_2], dtype=int)
        return edge_adj, adj_label

    def get_adj_ht_pa(self):
        adj, adj_label = self.adj_process(self.adj_mx)
        adj = torch.from_numpy(adj).long()
        adj_label = torch.from_numpy(adj_label).long()
        # Merge 2 graph as scconv's adj
        adj_dense = torch.sparse_coo_tensor(
            adj, torch.ones((adj.shape[1])), torch.Size([self.N, self.N])).to_dense()
        adj_dense_label = torch.sparse_coo_tensor(adj_label, torch.ones((adj_label.shape[1])),
                                                  torch.Size([self.N, self.N])).to_dense()
        adj_dense = adj_dense + adj_dense_label
        adj_dense = torch.where(adj_dense < 1e-8, adj_dense, torch.ones(1, 1))
        adj_merge = adj_dense.to(device=self.device).repeat(self.batch_size, 1, 1)
        g_adj = dgl.DGLGraph()
        g_adj.add_nodes(self.N)
        g_adj.add_edges(adj[0], adj[1])
        # expand for batch training
        adj = self.spadj_expand(g_adj, self.batch_size)
        g_adj_label = dgl.DGLGraph()
        g_adj_label.add_nodes(self.N)
        g_adj_label.add_edges(adj_label[0], adj_label[1])
        adj_label = self.spadj_expand(g_adj_label, self.batch_size)
        adjs = (adj, adj_label, adj_merge)
        # to init GRU hidden state
        h_t = torch.zeros(self.batch_size, self.N, self.hid_dim * 2).to(device=self.device)
        # to discretize y for PA approximation
        pa_onehot = torch.zeros(self.batch_size, self.N, self.t_in, 50).to(device=self.device)
        return adjs, h_t, pa_onehot

    def forward(self, batch):
        X = batch['X']  # (B, input_window, N, feature_dim)
        X = X.permute(0, 2, 1, 3)  # (B, N, input_window, feature_dim)
        B, N, T, F_feat = X.size()
        X_cxt, X_pa, X_time = self.feature_embedding(X, self.pa_onehot)
        CE_loss = 0.0
        for i in range(T):
            y_t = F.softmax(self.w_pred(self.h_t), dim=-1)  # (B, N, p=50)
            if i == T - 1:
                CE_loss += F.binary_cross_entropy(y_t[:, :self.train_num, :].reshape(B * self.train_num, -1),
                                                  X_pa[:, :self.train_num, i, :].reshape(B * self.train_num, -1))
            # PropConv
            y_att = self.PropConv(X_cxt[:, :, i, :], X_pa[:, :, i, :], self.adj_label)  # (B, N, p=50)
            if i == T - 1:
                y_att[:, :self.train_num, :] = torch.where(y_att[:, :self.train_num, :] < 1.,
                                                           y_att[:, :self.train_num, :],
                                                           (1. - 1e-8) * torch.ones(1, 1).cuda())
                CE_loss += F.binary_cross_entropy(y_att[:, :self.train_num, :].reshape(B * self.train_num, -1),
                                                  X_pa[:, :self.train_num, i, :].reshape(B * self.train_num, -1))
            # PA approximation
            en_yt = torch.exp(torch.sum(y_t * torch.log(
                torch.where(y_t > 1e-8, y_t, 1e-8 * torch.ones(1, 1).to(self.device))), dim=-1, keepdim=True))
            en_yatt = torch.exp(torch.sum(y_att * torch.log(
                torch.where(y_att > 1e-8, y_att, 1e-8 * torch.ones(1, 1).cuda())), dim=-1, keepdim=True))
            en_yatt = torch.where(torch.sum(y_att, dim=-1, keepdim=True) > 1e-8,
                                  en_yatt, torch.zeros(1, 1).to(self.device))
            pseudo_y = (en_yt * y_t + en_yatt * y_att) / (en_yt + en_yatt)
            if not self.training:
                pseudo_y[:, :self.train_num, :] = X_pa[:, :self.train_num, i, :]
            # CxtConv
            h_cxt = self.CxtConv(X_cxt[:, :, i, :], None, self.adj)  # (B, N, tmp_hid)
            # SCConv
            h_sc = self.SCConv(torch.cat([h_cxt, pseudo_y], dim=-1), self.adj_dense)
            X_feat = torch.cat([h_cxt, pseudo_y, h_sc, X_time[..., i, :]], dim=-1)
            self.h_t = self.GRU(X_feat.view(-1, 2 * self.hid_dim + 50 + 20),
                                self.h_t.view(-1, self.hid_dim * 2))  # (B*N, 2*tmp_hid)
            self.h_t = self.h_t.view(B, N, -1)
        out = torch.sigmoid(self.output_fc(self.h_t))  # (B, N, T_out)
        return out, CE_loss

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted, ce_loss = self.forward(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        mse_loss = loss.masked_mse_torch(y_predicted, y_true)
        return ce_loss + mse_loss

    def predict(self, batch):
        return self.forward(batch)[0]
