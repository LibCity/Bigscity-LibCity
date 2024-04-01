import warnings
from logging import getLogger

import numpy as np
from scipy.sparse.linalg import eigs

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class h_swish(nn.Module):
    def __init__(self, inplace=False):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


class DYReLU(nn.Module):
    def __init__(self, inp, oup, norm_layer=nn.BatchNorm2d, reduction=4, lambda_a=1.0, K2=True, use_bias=True,
                 use_spatial=False,
                 init_a=[1.0, 0.0], init_b=[0.0, 0.0]):
        super(DYReLU, self).__init__()
        self.oup = oup
        self.lambda_a = lambda_a * 2
        self.K2 = K2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.use_bias = use_bias
        if K2:
            self.exp = 4 if use_bias else 2
        else:
            self.exp = 2 if use_bias else 1
        self.init_a = init_a
        self.init_b = init_b

        # determine squeeze
        if reduction == 4:
            squeeze = inp // reduction
        else:
            squeeze = _make_divisible(inp // reduction, 4)
        # print('reduction: {}, squeeze: {}/{}'.format(reduction, inp, squeeze))
        # print('init_a: {}, init_b: {}'.format(self.init_a, self.init_b))

        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze, oup * self.exp),
            h_sigmoid()
        )
        if use_spatial:
            self.spa = nn.Sequential(
                nn.Conv2d(inp, 1, kernel_size=1),
                norm_layer(1),
            )
        else:
            self.spa = None

    def forward(self, x):
        if isinstance(x, list):
            x_in = x[0]
            x_out = x[1]
        else:
            x_in = x
            x_out = x

        x_shape = len(x_in.size())
        if x_shape == 2:
            b, c = x_in.size()
            x = x.view(b, c, 1, 1)
            x_in = x
            x_out = x
        else:
            b, c, h = x_in.size()
            x = x.view(b, c, h, 1)
            x_in = x
            x_out = x

        b, c, h, w = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup * self.exp, 1, 1)
        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
            a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]

            b1 = b1 - 0.5 + self.init_b[0]
            b2 = b2 - 0.5 + self.init_b[1]
            out = torch.max(x_out * a1 + b1, x_out * a2 + b2)

        elif self.exp == 2:
            if self.use_bias:  # bias but not PL
                a1, b1 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
                b1 = b1 - 0.5 + self.init_b[0]
                out = x_out * a1 + b1

            else:
                a1, a2 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
                a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
                out = torch.max(x_out * a1, x_out * a2)

        elif self.exp == 1:
            a1 = y
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
            out = x_out * a1

        if self.spa:
            ys = self.spa(x_in).view(b, -1)
            ys = F.softmax(ys, dim=1).view(b, 1, h, w) * h * w
            ys = F.hardtanh(ys, 0, 3, inplace=True) / 3
            out = out * ys

        if x_shape == 2:
            out = out.view(b, c)
        else:
            out = out.view(b, c, -1)
        return out


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        x_sz = len(x.size())
        if x_sz == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        elif x_sz == 3:
            x = x.unsqueeze(-1)
        else:
            pass

        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        if x_sz == 2:
            out = out.squeeze(-1).squeeze(-1)
        elif x_sz == 3:
            out = out.squeeze(-1)
        else:
            pass
        return out


class AttLayer(nn.Module):
    def __init__(self, out_channels, use_bias=False, reduction=16):
        super(AttLayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // reduction, 1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, 1, 1)
        return x * y.expand_as(x)


class SigM(nn.Module):
    def __init__(self, in_channel, output_channel, reduction=1):
        super(SigM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.output_channel = output_channel
        self.h_sigmoid = h_sigmoid()
        if in_channel == output_channel:
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
            )
        else:
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv2d(in_channel, output_channel, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x_sz = len(x.size())
        if x_sz == 2:
            x = x.unsqueeze(-1)
        b, c, _, = x.size()
        y = self.fc(x).view(b, self.output_channel, 1)
        y = self.h_sigmoid(y)
        out = x * y.expand_as(x)
        if x_sz == 2:
            out = out.squeeze(-1)
        return out


class SELayer(nn.Module):
    def __init__(self, in_channel, output_channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, output_channel, bias=False),
            nn.Sigmoid()
        )

        self.output_channel = output_channel

    def forward(self, x):
        x_sz = len(x.size())
        if x_sz == 2:
            x = x.unsqueeze(-1)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.output_channel, 1)
        out = x * y.expand_as(x)
        if x_sz == 2:
            out = out.squeeze(-1)
        return out


class AVWGCN(nn.Module):
    def __init__(self, cheb_polynomials, L_tilde, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.cheb_polynomials = cheb_polynomials
        self.L_tilde = L_tilde
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

        # for existing graph convolution
        # self.init_gconv = nn.Conv1d(dim_in, dim_out, kernel_size=5, padding=0)
        self.init_gconv = nn.Linear(dim_in, dim_out)
        self.gconv = nn.Linear(dim_out * cheb_k, dim_out)
        self.dy_gate1 = AttLayer(dim_out)
        self.dy_gate2 = AttLayer(dim_out)

    def forward(self, x, node_embeddings, L_tilde_learned):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        b, n, _ = x.shape
        # 0) learned cheb_polynomials
        node_num = node_embeddings.shape[0]

        # L_tilde_learned = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        # L_tilde_learned = torch.matmul(L_tilde_learned, self.L_tilde) * L_tilde_learned

        support_set = [torch.eye(node_num).to(L_tilde_learned.device), L_tilde_learned]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * L_tilde_learned, support_set[-1]) - support_set[-2])

        # 1) convolution with learned graph convolution (implicit knowledge)
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv0 = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # b, N, dim_out

        # 2) convolution with existing graph (explicit knowledge)
        graph_supports = torch.stack(self.cheb_polynomials, dim=0)  # [k, n, m]
        x = self.init_gconv(x)
        x_g1 = torch.einsum("knm,bmc->bknc", graph_supports, x)
        x_g1 = x_g1.permute(0, 2, 1, 3).reshape(b, n, -1)  # B, N, cheb_k, dim_in
        x_gconv1 = self.gconv(x_g1)

        # 3) fusion of explit knowledge and implicit knowledge
        x_gconv = self.dy_gate1(F.leaky_relu(x_gconv0).transpose(1, 2)) + self.dy_gate2(
            F.leaky_relu(x_gconv1).transpose(1, 2))
        # x_gconv = F.leaky_relu(x_gconv0) + F.leaky_relu(x_gconv1)

        return x_gconv.transpose(1, 2)


class RGSLCell(nn.Module):
    def __init__(self, cheb_polynomials, L_tilde, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(RGSLCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(cheb_polynomials, L_tilde, dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(cheb_polynomials, L_tilde, dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings, learned_tilde):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings, learned_tilde))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings, learned_tilde))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    # y_soft = gumbels.softmax(dim)
    y_soft = gumbels

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class AVWDCRNN(nn.Module):
    def __init__(self, cheb_polynomials, L_tilde, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(RGSLCell(cheb_polynomials, L_tilde, node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(RGSLCell(cheb_polynomials, L_tilde, node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings, learned_tilde):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings, learned_tilde)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


def scaled_Laplacian(W):
    '''
    compute \tilde{L}
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)
    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}
    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)
    K: the maximum order of chebyshev polynomials
    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}
    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


class RGSL(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))

        # data
        self.num_node = self.data_feature.get('num_nodes')
        self.num_nodes = self.num_node
        self.adj_mx = self.data_feature.get('adj_mx')
        self._scaler = self.data_feature.get('scaler')

        # model
        self.horizon = config.get('output_window', 12)
        self.embed_dim = config.get('embed_dim', 8)
        self.cheb_k = config.get('cheb_k', 2)
        self.input_dim = config.get('input_dim', 1)
        self.hidden_dim = config.get('rnn_units', 64)
        self.output_dim = config.get('output_dim', 1)
        self.num_layers = config.get('num_layers', 2)

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)

        L_tilde = scaled_Laplacian(self.adj_mx)
        cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(self.device) for i in
                            cheb_polynomial(L_tilde, self.cheb_k)]
        L_tilde = torch.from_numpy(L_tilde).type(torch.FloatTensor).to(self.device)

        self.encoder = AVWDCRNN(cheb_polynomials, L_tilde, self.num_nodes, self.input_dim, self.hidden_dim, self.cheb_k,
                                self.embed_dim, self.num_layers)

        # predictor
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.adj = None
        self.tilde = None

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def scaled_laplacian(self, node_embeddings, is_eval=False):
        """
        Normalized graph Laplacian function.
        :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
        :return: np.matrix, [n_route, n_route].
        """
        # learned graph
        node_num = self.num_node
        learned_graph = torch.mm(node_embeddings, node_embeddings.transpose(0, 1))
        norm = torch.norm(node_embeddings, p=2, dim=1, keepdim=True)
        norm = torch.mm(norm, norm.transpose(0, 1))
        learned_graph = learned_graph / norm
        learned_graph = (learned_graph + 1) / 2.
        # learned_graph = F.sigmoid(learned_graph)
        learned_graph = torch.stack([learned_graph, 1 - learned_graph], dim=-1)

        # make the adj sparse
        if is_eval:
            adj = gumbel_softmax(learned_graph, tau=1, hard=True)
        else:
            adj = gumbel_softmax(learned_graph, tau=1, hard=True)
        adj = adj[:, :, 0].clone().reshape(node_num, -1)
        # mask = torch.eye(self.num_nodes, self.num_nodes).to(device).byte()
        mask = torch.eye(node_num, node_num).bool().cuda()
        adj.masked_fill_(mask, 0)

        # d ->  diagonal degree matrix
        W = adj
        n = W.shape[0]
        d = torch.sum(W, axis=1)
        ## L -> graph Laplacian
        L = -W
        L[range(len(L)), range(len(L))] = d
        try:
            lambda_max = (L.max() - L.min())
        except Exception as e:
            print("eig error!!: {}".format(e))
            lambda_max = 1.0

        # pesudo laplacian matrix, lambda_max = eigs(L.cpu().detach().numpy(), k=1, which='LR')[0][0].real
        tilde = (2 * L / lambda_max - torch.eye(n).cuda())
        self.adj = adj
        self.tilde = tilde
        return adj, tilde

    def forward(self, source):
        # source: B, T_1, N, D
        # target: B, T_2, N, D
        # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        if self.train:
            adj, learned_tilde = self.scaled_laplacian(self.node_embeddings, is_eval=False)
        else:
            adj, learned_tilde = self.scaled_laplacian(self.node_embeddings, is_eval=True)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings, learned_tilde)  # B, T, N, hidden
        output = output[:, -1:, :, :]  # B, 1, N, hidden

        # CNN based predictor
        output = self.end_conv((output))  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)  # B, T, N, C

        return output

    def predict(self, batch):
        return self.forward(batch['X'])

    def calculate_loss(self, batch):
        y_true = batch['y']  # ground-truth value
        y_predicted = self.predict(batch)  # prediction results
        # denormalization the value
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, null_val = 0.0)
