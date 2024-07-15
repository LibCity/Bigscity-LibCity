import torch
import torch.nn as nn
from logging import getLogger

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import copy
import numpy as np
import torch


def sim_global(flow_data, sim_type='cos'):
    """Calculate the global similarity of traffic flow data.
    :param flow_data: tensor, original flow [n,l,v,c] or location embedding [n,v,c]
    :param type: str, type of similarity, attention or cosine. ['att', 'cos']
    :return sim: tensor, symmetric similarity, [v,v]
    """
    if len(flow_data.shape) == 4:
        n, l, v, c = flow_data.shape
        att_scaling = n * l * c
        cos_scaling = torch.norm(flow_data, p=2, dim=(0, 1, 3)) ** -1  # cal 2-norm of each node, dim N
        sim = torch.einsum('btnc, btmc->nm', flow_data, flow_data)
    elif len(flow_data.shape) == 3:
        n, v, c = flow_data.shape
        att_scaling = n * c
        cos_scaling = torch.norm(flow_data, p=2, dim=(0, 2)) ** -1  # cal 2-norm of each node, dim N
        sim = torch.einsum('bnc, bmc->nm', flow_data, flow_data)
    else:
        raise ValueError('sim_global only support shape length in [3, 4] but got {}.'.format(len(flow_data.shape)))

    if sim_type == 'cos':
        # cosine similarity
        scaling = torch.einsum('i, j->ij', cos_scaling, cos_scaling)
        sim = sim * scaling
    elif sim_type == 'att':
        # scaled dot product similarity
        scaling = float(att_scaling) ** -0.5
        sim = torch.softmax(sim * scaling, dim=-1)
    else:
        raise ValueError('sim_global only support sim_type in [att, cos].')

    return sim


def aug_topology(sim_mx, input_graph, percent=0.2):
    """Generate the data augumentation from topology (graph structure) perspective
        for undirected graph without self-loop.
    :param sim_mx: tensor, symmetric similarity, [v,v]
    :param input_graph: tensor, adjacency matrix without self-loop, [v,v]
    :return aug_graph: tensor, augmented adjacency matrix on cuda, [v,v]
    """
    ## edge dropping starts here
    drop_percent = percent / 2

    index_list = input_graph.nonzero()  # list of edges [row_idx, col_idx]

    edge_num = int(index_list.shape[0] / 2)  # treat one undirected edge as two edges
    edge_mask = (input_graph > 0).tril(diagonal=-1)
    add_drop_num = int(edge_num * drop_percent / 2)
    aug_graph = copy.deepcopy(input_graph)

    drop_prob = torch.softmax(sim_mx[edge_mask], dim=0)
    drop_prob = (1. - drop_prob).numpy()  # normalized similarity to get sampling probability
    drop_prob /= drop_prob.sum()
    drop_list = np.random.choice(edge_num, size=add_drop_num, p=drop_prob)
    drop_index = index_list[drop_list]

    zeros = torch.zeros_like(aug_graph[0, 0])
    aug_graph[drop_index[:, 0], drop_index[:, 1]] = zeros
    aug_graph[drop_index[:, 1], drop_index[:, 0]] = zeros

    ## edge adding starts here
    node_num = input_graph.shape[0]
    x, y = np.meshgrid(range(node_num), range(node_num), indexing='ij')
    mask = y < x
    x, y = x[mask], y[mask]

    add_prob = sim_mx[torch.ones(sim_mx.size(), dtype=bool).tril(diagonal=-1)]  # .numpy()
    add_prob = torch.softmax(add_prob, dim=0).numpy()
    add_list = np.random.choice(int((node_num * node_num - node_num) / 2),
                                size=add_drop_num, p=add_prob)

    ones = torch.ones_like(aug_graph[0, 0])
    aug_graph[x[add_list], y[add_list]] = ones
    aug_graph[y[add_list], x[add_list]] = ones

    return aug_graph


def aug_traffic(t_sim_mx, flow_data, percent=0.2):
    """Generate the data augumentation from traffic (node attribute) perspective.
    :param t_sim_mx: temporal similarity matrix after softmax, [l,n,v]
    :param flow_data: input flow data, [n,l,v,c]
    """
    l, n, v = t_sim_mx.shape
    mask_num = int(n * l * v * percent)
    aug_flow = copy.deepcopy(flow_data)

    mask_prob = (1. - t_sim_mx.permute(1, 0, 2).reshape(-1)).numpy()
    mask_prob /= mask_prob.sum()

    x, y, z = np.meshgrid(range(n), range(l), range(v), indexing='ij')
    mask_list = np.random.choice(n * l * v, size=mask_num, p=mask_prob)

    zeros = torch.zeros_like(aug_flow[0, 0, 0])
    aug_flow[
        x.reshape(-1)[mask_list],
        y.reshape(-1)[mask_list],
        z.reshape(-1)[mask_list]] = zeros

    return aug_flow


########################################
## Spatial Heterogeneity Modeling
########################################
class SpatialHeteroModel(nn.Module):
    '''Spatial heterogeneity modeling by using a soft-clustering paradigm.
    '''

    def __init__(self, c_in, nmb_prototype, batch_size, tau=0.5):
        super(SpatialHeteroModel, self).__init__()
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.prototypes = nn.Linear(c_in, nmb_prototype, bias=False)

        self.tau = tau
        self.d_model = c_in
        self.batch_size = batch_size

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, z1, z2):
        """Compute the contrastive loss of batched data.
        :param z1, z2 (tensor): shape nlvc
        :param loss: contrastive loss
        """
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = self.l2norm(w)
            self.prototypes.weight.copy_(w)

        # l2norm avoids nan of Q in sinkhorn
        zc1 = self.prototypes(self.l2norm(z1.reshape(-1, self.d_model)))  # nd -> nk, assignment q, embedding z
        zc2 = self.prototypes(self.l2norm(z2.reshape(-1, self.d_model)))  # nd -> nk
        with torch.no_grad():
            q1 = sinkhorn(zc1.detach())
            q2 = sinkhorn(zc2.detach())
        l1 = - torch.mean(torch.sum(q1 * F.log_softmax(zc2 / self.tau, dim=1), dim=1))
        l2 = - torch.mean(torch.sum(q2 * F.log_softmax(zc1 / self.tau, dim=1), dim=1))
        return l1 + l2


@torch.no_grad()
def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
    Q = torch.exp(out / epsilon).t()  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1]  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


########################################
## Temporal Heterogeneity Modeling
########################################
class TemporalHeteroModel(nn.Module):
    '''Temporal heterogeneity modeling in a contrastive manner.
    '''

    def __init__(self, c_in, batch_size, num_nodes, device):
        super(TemporalHeteroModel, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_nodes, c_in))  # representation weights
        self.W2 = nn.Parameter(torch.FloatTensor(num_nodes, c_in))
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))

        self.read = AvgReadout()
        self.disc = Discriminator(c_in)
        self.b_xent = nn.BCEWithLogitsLoss()

        lbl_rl = torch.ones(batch_size, num_nodes)
        lbl_fk = torch.zeros(batch_size, num_nodes)
        lbl = torch.cat((lbl_rl, lbl_fk), dim=1)
        if device == 'cuda' or device.type == 'cuda':
            self.lbl = lbl.cuda()

        self.n = batch_size

    def forward(self, z1, z2):
        '''
        :param z1, z2 (tensor): shape nlvc, i.e., (batch_size, seq_len, num_nodes, feat_dim)
        :return loss: loss of generative branch. nclv
        '''
        h = (z1 * self.W1 + z2 * self.W2).squeeze(1)  # nlvc->nvc
        s = self.read(h)  # s: summary of h, nc

        # select another region in batch
        idx = torch.randperm(self.n)
        shuf_h = h[idx]

        logits = self.disc(s, h, shuf_h)
        loss = self.b_xent(logits, self.lbl)
        return loss


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
        self.sigm = nn.Sigmoid()

    def forward(self, h):
        '''Apply an average on graph.
        :param h: hidden representation, (batch_size, num_nodes, feat_dim)
        :return s: summary, (batch_size, feat_dim)
        '''
        s = torch.mean(h, dim=1)
        s = self.sigm(s)
        return s


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.net = nn.Bilinear(n_h, n_h, 1)  # similar to score of CPC

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, summary, h_rl, h_fk):
        '''
        :param s: summary, (batch_size, feat_dim)
        :param h_rl: real hidden representation (w.r.t summary),
            (batch_size, num_nodes, feat_dim)
        :param h_fk: fake hidden representation
        :return logits: prediction scores, (batch_size, num_nodes, 2)
        '''
        s = torch.unsqueeze(summary, dim=1)
        s = s.expand_as(h_rl).contiguous()

        # score of real and fake, (batch_size, num_nodes)
        sc_rl = torch.squeeze(self.net(h_rl, s), dim=2)
        sc_fk = torch.squeeze(self.net(h_fk, s), dim=2)

        logits = torch.cat((sc_rl, sc_fk), dim=1)

        return logits


####################
## ST Encoder
####################
class STEncoder(nn.Module):
    def __init__(self, Kt, Ks, blocks, input_length, num_nodes, droprate=0.1):
        super(STEncoder, self).__init__()
        self.Ks = Ks
        c = blocks[0]
        self.tconv11 = TemporalConvLayer(Kt, c[0], c[1], "GLU")
        self.pooler = Pooler(input_length - (Kt - 1), c[1])

        self.sconv12 = SpatioConvLayer(Ks, c[1], c[1])
        self.tconv13 = TemporalConvLayer(Kt, c[1], c[2])
        self.ln1 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout1 = nn.Dropout(droprate)

        c = blocks[1]
        self.tconv21 = TemporalConvLayer(Kt, c[0], c[1], "GLU")

        self.sconv22 = SpatioConvLayer(Ks, c[1], c[1])
        self.tconv23 = TemporalConvLayer(Kt, c[1], c[2])
        self.ln2 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout2 = nn.Dropout(droprate)

        self.s_sim_mx = None
        self.t_sim_mx = None

        out_len = input_length - 2 * (Kt - 1) * len(blocks)
        self.out_conv = TemporalConvLayer(out_len, c[2], c[2], "GLU")
        self.ln3 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout3 = nn.Dropout(droprate)
        self.receptive_field = input_length + Kt - 1

    def forward(self, x0, graph):
        lap_mx = self._cal_laplacian(graph)
        Lk = self._cheb_polynomial(lap_mx, self.Ks)

        in_len = x0.size(1)  # x0, nlvc
        if in_len < self.receptive_field:
            x = F.pad(x0, (0, 0, 0, 0, self.receptive_field - in_len, 0))
        else:
            x = x0
        x = x.permute(0, 3, 1, 2)  # (batch_size, feature_dim, input_length, num_nodes), nclv

        ## ST block 1
        x = self.tconv11(x)  # nclv
        x, x_agg, self.t_sim_mx = self.pooler(x)
        self.s_sim_mx = sim_global(x_agg, sim_type='cos')

        x = self.sconv12(x, Lk)  # nclv
        x = self.tconv13(x)
        x = self.dropout1(self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

        ## ST block 2
        x = self.tconv21(x)
        x = self.sconv22(x, Lk)
        x = self.tconv23(x)
        x = self.dropout2(self.ln2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

        ## out block
        x = self.out_conv(x)  # ncl(=1)v
        x = self.dropout3(self.ln3(x.permute(0, 2, 3, 1)))  # nlvc
        return x  # nl(=1)vc

    def _cheb_polynomial(self, laplacian, K):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [v, v].
        :return: the multi order Chebyshev laplacian, [K, v, v].
        """
        N = laplacian.size(0)
        multi_order_laplacian = torch.zeros([K, N, N], device=laplacian.device, dtype=torch.float)
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k - 1]) - \
                                               multi_order_laplacian[k - 2]

        return multi_order_laplacian

    def _cal_laplacian(self, graph):
        """
        return the laplacian of the graph.

        :param graph: the graph structure **without** self loop, [v, v].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        I = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype)
        graph = graph + I  # add self-loop to prevent zero in D
        D = torch.diag(torch.sum(graph, dim=-1) ** (-0.5))
        L = I - torch.mm(torch.mm(D, graph), D)
        return L


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        '''Align the input and output.
        '''
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)  # filter=(1,1), similar to fc

    def forward(self, x):  # x: (n,c,l,v)
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x


class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        """
        :param x: (n,c,l,v)
        :return: (n,c,l-kt+1,v)
        """
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)


class SpatioConvLayer(nn.Module):
    def __init__(self, ks, c_in, c_out):
        super(SpatioConvLayer, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks))  # kernel: C_in*C_out*ks
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1))
        self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
        x_in = self.align(x)
        return torch.relu(x_gc + x_in)


class Pooler(nn.Module):
    '''Pooling the token representations of region time series into the region level.
    '''

    def __init__(self, n_query, d_model, agg='avg'):
        """
        :param n_query: number of query
        :param d_model: dimension of model
        """
        super(Pooler, self).__init__()

        ## attention matirx
        self.att = FCLayer(d_model, n_query)
        self.align = Align(d_model, d_model)
        self.softmax = nn.Softmax(dim=2)  # softmax on the seq_length dim, nclv

        self.d_model = d_model
        self.n_query = n_query
        if agg == 'avg':
            self.agg = nn.AvgPool2d(kernel_size=(n_query, 1), stride=1)
        elif agg == 'max':
            self.agg = nn.MaxPool2d(kernel_size=(n_query, 1), stride=1)
        else:
            raise ValueError('Pooler supports [avg, max]')

    def forward(self, x):
        """
        :param x: key sequence of region embeding, nclv
        :return x: hidden embedding used for conv, ncqv
        :return x_agg: region embedding for spatial similarity, nvc
        :return A: temporal attention, lnv
        """
        x_in = self.align(x)[:, :, -self.n_query:, :]  # ncqv
        # calculate the attention matrix A using key x
        A = self.att(x)  # x: nclv, A: nqlv
        A = F.softmax(A, dim=2)  # nqlv

        # calculate region embeding using attention matrix A
        x = torch.einsum('nclv,nqlv->ncqv', x, A)
        x_agg = self.agg(x).squeeze(2)  # ncqv->ncv
        x_agg = torch.einsum('ncv->nvc', x_agg)  # ncv->nvc

        # calculate the temporal simlarity (prob)
        A = torch.einsum('nqlv->lnqv', A)
        A = self.softmax(self.agg(A).squeeze(2))  # A: lnqv->lnv
        return torch.relu(x + x_in), x_agg.detach(), A.detach()


########################################
## An MLP predictor
########################################
class MLP(nn.Module):
    def __init__(self, c_in, c_out):
        super(MLP, self).__init__()
        self.fc1 = FCLayer(c_in, int(c_in // 2))
        self.fc2 = FCLayer(int(c_in // 2), c_out)

    def forward(self, x):
        x = torch.tanh(self.fc1(x.permute(0, 3, 1, 2)))  # nlvc->nclv
        x = self.fc2(x).permute(0, 2, 3, 1)  # nclv->nlvc
        return x


class FCLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(FCLayer, self).__init__()
        self.linear = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        return self.linear(x)


class STSSL(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.len_closeness = self.data_feature.get('len_closeness', 4)
        self.len_period = self.data_feature.get('len_period', 2)
        self.len_trend = self.data_feature.get('len_trend', 0)
        self.yita = config.get('yita', 0.6)
        self.percent = config.get('percent', 0.1)
        # spatial temporal encoder
        self.encoder = STEncoder(Kt=3, Ks=3,
                                 blocks=[[2, int(config.get('d_model', 64) // 2), config.get('d_model', 64)],
                                         [config.get('d_model', 64), int(config.get('d_model', 64) // 2),
                                          config.get('d_model', 64)]],
                                 input_length=self.len_closeness + self.len_period + self.len_trend,
                                 num_nodes=self.data_feature.get('num_nodes', 1), droprate=config.get('dropout', 0.2))

        # traffic flow prediction branch
        self.mlp = MLP(config.get('d_model', 64), config.get('d_output', 2))
        # temporal heterogenrity modeling branch
        self.thm = TemporalHeteroModel(config.get('d_model', 64), config.get('batch_size', 32),
                                       self.data_feature.get('num_nodes', 1),
                                       config.get('device', 'cuda'))
        # spatial heterogenrity modeling branch
        self.shm = SpatialHeteroModel(config.get('d_model', 64), config.get('nmb_prototype', 6),
                                      config.get('batch_size', 32), config.get('shm_temp', 0.5))
        self.mae = lambda x, y: loss.masked_mae_torch(x, y, 0, mask_val=5.001)

        self.graph = torch.from_numpy(self.data_feature.get('adj_mx'))
        self.graph = self.graph.cuda()
        self.scaler = self.data_feature.get('scaler')

        self.W = self.data_feature.get('len_row')
        self.H = self.data_feature.get('len_column')

    def forward(self, batch):
        x = batch['X']

        repr1 = self.encoder(x, self.graph)  # view1: n,l,v,c; graph: v,v

        s_sim_mx = self.fetch_spatial_sim()
        graph2 = aug_topology(s_sim_mx, self.graph, percent=self.percent * 2)

        t_sim_mx = self.fetch_temporal_sim()
        view2 = aug_traffic(t_sim_mx, x, percent=self.percent)

        repr2 = self.encoder(view2, graph2)
        return repr1, repr2

    def fetch_spatial_sim(self):
        """
        Fetch the region similarity matrix generated by region embedding.
        Note this can be called only when spatial_sim is True.
        :return sim_mx: tensor, similarity matrix, (v, v)
        """
        return self.encoder.s_sim_mx.cpu()

    def fetch_temporal_sim(self):
        return self.encoder.t_sim_mx.cpu()

    def _predict(self, z1, z2):
        '''Predicting future traffic flow.
        :param z1, z2 (tensor): shape nvc
        :return: nlvc, l=1, c=2
        '''
        return self.mlp(z1)

    def loss(self, z1, z2, y_true, scaler, loss_weights):
        l1 = self.pred_loss(z1, z2, y_true, scaler)
        sep_loss = [l1.item()]
        loss = loss_weights[0] * l1

        l2 = self.temporal_loss(z1, z2)
        sep_loss.append(l2.item())
        loss += loss_weights[1] * l2

        l3 = self.spatial_loss(z1, z2)
        sep_loss.append(l3.item())
        loss += loss_weights[2] * l3
        return loss, sep_loss

    def pred_loss(self, z1, z2, y_true, scaler):
        y_pred = scaler.inverse_transform(self._predict(z1, z2))
        y_true = scaler.inverse_transform(y_true)

        loss = self.yita * self.mae(y_pred[..., 0], y_true[..., 0]) + \
               (1 - self.yita) * self.mae(y_pred[..., 1], y_true[..., 1])
        return loss

    def temporal_loss(self, z1, z2):
        return self.thm(z1, z2)

    def spatial_loss(self, z1, z2):
        return self.shm(z1, z2)

    def predict(self, batch):
        z1, z2 = self.forward(batch)
        ret = self._predict(z1, z2)
        return ret

    def calculate_loss_with_weight(self, batch, loss_weights):
        repr1, repr2 = self.forward(batch)  # nvc
        y_true = batch['y']
        loss, sep_loss = self.loss(repr1, repr2, y_true, self.scaler, loss_weights)
        return loss, sep_loss

    def calculate_loss(self, batch):
        repr1, repr2 = self.forward(batch)  # nvc
        y_true = batch['y']
        loss, sep_loss = self.loss(repr1, repr2, y_true, self.scaler, np.ones(3))
        return loss
