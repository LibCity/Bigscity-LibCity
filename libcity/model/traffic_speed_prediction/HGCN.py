from logging import getLogger
import torch
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Conv2d, Parameter, BatchNorm1d
from libcity.model import loss
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32).todense()


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        A = A.transpose(-1, -2)
        x = torch.einsum('ncvl,vw->ncwl', x, A)
        return x.contiguous()


class multi_gcn_time(nn.Module):
    def __init__(self, c_in, c_out, Kt, dropout, support_len=3, order=2):
        super(multi_gcn_time, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear_time(c_in, c_out, Kt)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        count = 0
        for a in support:
            count += 1
            a = a.to(x.device)
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class TATT_1(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TATT_1, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)

        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn = BatchNorm1d(tem_size)
        self.c_in = c_in
        self.tem_size = tem_size

    def forward(self, seq):
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()  # b,l,n

        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze()  # b,c,n
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        logits = logits.permute(0, 2, 1).contiguous()
        logits = self.bn(logits).permute(0, 2, 1).contiguous()
        coefs = torch.softmax(logits, -1)
        return coefs


class linear_time(nn.Module):
    def __init__(self, c_in, c_out, Kt):
        super(linear_time, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class GCNPool(nn.Module):
    ''' #GCN      S-T Blocks'''

    def __init__(self, c_in, c_out, num_nodes, tem_size,
                 Kt, dropout, pool_nodes, support_len=3, order=2):
        super(GCNPool, self).__init__()
        self.time_conv = Conv2d(c_in, 2 * c_out, kernel_size=(1, Kt), padding=(0, 0),
                                stride=(1, 1), bias=True, dilation=2)

        self.multigcn = multi_gcn_time(c_out, 2 * c_out, Kt, dropout, support_len, order)

        self.num_nodes = num_nodes
        self.tem_size = tem_size
        self.TAT = TATT_1(c_out, num_nodes, tem_size)
        self.c_out = c_out
        # self.bn=LayerNorm([c_out,num_nodes,tem_size])
        self.bn = BatchNorm2d(c_out)

        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)

    def forward(self, x, support):
        residual = self.conv1(x)
        x = self.time_conv(x)

        x1, x2 = torch.split(x, [self.c_out, self.c_out], 1)
        x = torch.tanh(x1) * torch.sigmoid(x2)

        x = self.multigcn(x, support)
        x1, x2 = torch.split(x, [self.c_out, self.c_out], 1)
        x = torch.tanh(x1) * (torch.sigmoid(x2))
        # x=F.dropout(x,0.3,self.training)

        T_coef = self.TAT(x)
        T_coef = T_coef.transpose(-1, -2)
        x = torch.einsum('bcnl,blq->bcnq', x, T_coef)
        out = self.bn(x + residual[:, :, :, -x.size(3):])
        return out


class Transmit(nn.Module):
    '''#Transfer Blocks  交换层'''

    def __init__(self, c_in, tem_size, transmit, num_nodes, cluster_nodes):
        super(Transmit, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(tem_size, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(tem_size, c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(num_nodes, cluster_nodes), requires_grad=True)
        self.c_in = c_in
        self.transmit = transmit
        self.tem_size = tem_size

    def forward(self, seq, seq_cluster):
        c1 = seq
        f1 = self.conv1(c1).squeeze(1)  # b,n,l

        c2 = seq_cluster.permute(0, 3, 1, 2)  # b,c,n,l->b,l,n,c
        f2 = self.conv2(c2).squeeze(1)  # b,c,n
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        a = torch.mean(logits, 1, True)
        logits = logits - a
        logits = torch.sigmoid(logits)

        coefs = (logits) * self.transmit
        return coefs


class gate(nn.Module):
    def __init__(self, c_in):
        super(gate, self).__init__()
        self.conv1 = Conv2d(c_in, c_in // 2, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)

    def forward(self, seq, seq_cluster):
        # x=torch.cat((seq_cluster,seq),1)
        # gate=torch.sigmoid(self.conv1(x))
        out = torch.cat((seq, (seq_cluster)), 1)

        return out


class HGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self.device = config.get('device', torch.device('cpu'))

        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.transmit = self.data_feature.get('transmit').to(self.device)
        self.adjtype = config.get('adjtype', 'doubletransition')
        self.adj_mx = self.cal_adj(self.data_feature.get('adj_mx'), self.adjtype)
        self.adj_mx_cluster = self.cal_adj(self.data_feature.get('adj_mx_cluster'), self.adjtype)
        self.centers_ind_groups = self.data_feature.get('centers_ind_groups')

        self._logger = getLogger()

        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)

        self.cluster_nodes = config.get('cluster_nodes', 1)
        self.dropout = config.get('dropout', 0)
        self.channels = config.get('channels', 32)
        self.skip_channels = config.get('skip_channels', 32)
        self.end_channels = config.get('end_channels', 512)

        self.supports = [torch.tensor(i).to(self.device) for i in self.adj_mx]
        self.supports_cluster = [torch.tensor(i).to(self.device) for i in self.adj_mx_cluster]
        self.supports_len = torch.tensor(0, device=self.device)
        self.supports_len_cluster = torch.tensor(0, device=self.device)

        self.supports_len += len(self.supports)
        self.supports_len_cluster += len(self.supports_cluster)

        self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
                                    out_channels=self.channels,
                                    kernel_size=(1, 1))
        self.start_conv_cluster = nn.Conv2d(in_channels=self.feature_dim,
                                            out_channels=self.channels,
                                            kernel_size=(1, 1))

        self.h = Parameter(torch.zeros(self.num_nodes, self.num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.h_cluster = Parameter(torch.zeros(self.cluster_nodes, self.cluster_nodes), requires_grad=True)
        nn.init.uniform_(self.h_cluster, a=0, b=0.0001)
        self.supports_len += 1
        self.supports_len_cluster += 1
        self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10, self.num_nodes), requires_grad=True)
        self.nodevec1_c = nn.Parameter(torch.randn(self.cluster_nodes, 10), requires_grad=True)
        self.nodevec2_c = nn.Parameter(torch.randn(10, self.cluster_nodes), requires_grad=True)

        self.block1 = GCNPool(2 * self.channels, self.channels, self.num_nodes, self.input_window - 6, 3,
                              self.dropout, self.num_nodes,
                              self.supports_len)
        self.block2 = GCNPool(2 * self.channels, self.channels, self.num_nodes, self.input_window - 9, 2,
                              self.dropout, self.num_nodes,
                              self.supports_len)

        self.block_cluster1 = GCNPool(
            c_in=self.channels,
            c_out=self.channels,
            num_nodes=self.cluster_nodes,
            tem_size=self.input_window-6,
            Kt=3,
            dropout=self.dropout,
            pool_nodes=self.cluster_nodes,
            support_len=self.supports_len
        )
        self.block_cluster2 = GCNPool(
            c_in=self.channels,
            c_out=self.channels,
            num_nodes=self.cluster_nodes,
            tem_size=self.input_window - 9,
            Kt=2,
            dropout=self.dropout,
            pool_nodes=self.cluster_nodes,
            support_len=self.supports_len
        )

        self.skip_conv1 = Conv2d(2 * self.channels, self.skip_channels, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)
        self.skip_conv2 = Conv2d(2 * self.channels, self.skip_channels, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 3),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.output_window,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.bn = BatchNorm2d(self.feature_dim, affine=False)

        self.bn_cluster = BatchNorm2d(self.feature_dim, affine=False)
        self.gate1 = gate(2 * self.channels)
        self.gate2 = gate(2 * self.channels)
        self.gate3 = gate(2 * self.channels)

        self.transmit1 = Transmit(self.channels, self.input_window, self.transmit, self.num_nodes,
                                  self.cluster_nodes)
        self.transmit2 = Transmit(self.channels, self.input_window - 6, self.transmit, self.num_nodes,
                                  self.cluster_nodes)
        self.transmit3 = Transmit(self.channels, self.input_window - 9, self.transmit, self.num_nodes,
                                  self.cluster_nodes)
        self.linear = nn.Linear(1, self.output_dim, bias=True)

    def get_input_cluster(self, input):
        batch_size, input_length, feature_dim = input.shape[0], input.shape[1], input.shape[
            3]

        input_cluster = torch.zeros([batch_size, input_length, self.cluster_nodes, feature_dim], dtype=torch.float,
                                    device=self.device)

        for k in range(self.cluster_nodes):
            input_cluster[:, :, k, :] = input[:, :, self.centers_ind_groups[k][0], :] + \
                                        input[:, :, self.centers_ind_groups[k][0], :]
        return input_cluster

    def cal_adj(self, adj, adjtype):
        if adjtype == "scalap":
            adj_mx = [calculate_scaled_laplacian(adj)]
        elif adjtype == "normlap":
            adj_mx = [calculate_normalized_laplacian(adj).astype(np.float32).todense()]
        elif adjtype == "symnadj":
            adj_mx = [sym_adj(adj)]
        elif adjtype == "transition":
            adj_mx = [asym_adj(adj)]
        elif adjtype == "doubletransition":
            adj_mx = [asym_adj(adj), asym_adj(np.transpose(adj))]
        elif adjtype == "identity":
            adj_mx = [np.diag(np.ones(adj.shape[0])).astype(np.float32)]
        else:
            assert 0, "adj type not defined"
        return adj_mx

    def forward(self, batch):
        input = batch['X'].permute(0, 3, 2, 1)

        input_cluster = self.get_input_cluster(input)

        x = self.bn(input)
        x_cluster = self.bn_cluster(input_cluster)

        # nodes
        A = F.relu(torch.mm(self.nodevec1, self.nodevec2))
        d = 1 / (torch.sum(A, -1))
        D = torch.diag_embed(d)
        A = torch.matmul(D, A)

        new_supports = self.supports + [A]
        # region
        A_cluster = F.relu(torch.mm(self.nodevec1_c, self.nodevec2_c))
        d_c = 1 / (torch.sum(A_cluster, -1))
        D_c = torch.diag_embed(d_c)
        A_cluster = torch.matmul(D_c, A_cluster)

        new_supports_cluster = self.supports_cluster + [A_cluster]

        # network
        x = self.start_conv(x)
        x_cluster = self.start_conv_cluster(x_cluster)
        transmit1 = self.transmit1(x, x_cluster)

        x_1 = (torch.einsum('bmn,bcnl->bcml', transmit1, x_cluster))

        x = self.gate1(x, x_1)

        skip = torch.tensor(0, device=self.device)

        # 1
        x_cluster = self.block_cluster1(x_cluster, new_supports_cluster)
        x = self.block1(x, new_supports)
        transmit2 = self.transmit2(x, x_cluster)
        x_2 = (torch.einsum('bmn,bcnl->bcml', transmit2, x_cluster))

        x = self.gate2(x, x_2)

        s1 = self.skip_conv1(x)
        skip = s1 + skip

        # 2
        x_cluster = self.block_cluster2(x_cluster, new_supports_cluster)
        x = self.block2(x, new_supports)
        transmit3 = self.transmit3(x, x_cluster)
        x_3 = (torch.einsum('bmn,bcnl->bcml', transmit3, x_cluster))

        x = self.gate3(x, x_3)

        s2 = self.skip_conv2(x)
        skip = skip[:, :, :, -s2.size(3):]
        skip = s2 + skip

        # output
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.linear(x)
        x = self.end_conv_2(x)
        return x

    def calculate_loss(self, batch):
        y_true = batch['y'].to(self.device)

        output = self.predict(batch)
        y_predicted = output

        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        res = loss.masked_mae_torch(y_predicted, y_true, 0)

        return res

    def predict(self, batch):
        return self.forward(batch)
