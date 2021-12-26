import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import numpy as np

from torch.autograd import Variable
from torch.nn import Parameter

from libcity.layer.Spatial.atten import SpatialAttentionLayer
from libcity.layer.utils import calculate_scaled_laplacian, calculate_cheb_poly, calculate_first_approx, \
    calculate_random_walk_matrix
from libcity.model.traffic_speed_prediction.STAGGCN import GATConv


class GAT(nn.Module):
    # 2 layer GAT
    def __init__(self, adj_mx, input_size, hidden_dim, output_size, num_heads):
        """
        两层GAT模型
        Args:
            adj_mx:
            input_size:
            hidden_dim:
            output_size:
            num_heads:
        """
        super(GAT, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.layer1 = GATConv(adj_mx, input_size, hidden_dim, num_heads)
        self.layer2 = GATConv(adj_mx, hidden_dim * num_heads, output_size, 1)
        self.adj_mx = adj_mx

    def forward(self, h):
        h = self.layer1(self.adj_mx, h)
        h = F.elu(h)
        h = self.layer2(self.adj_mx, h)
        return h


# 可训练邻接矩阵+非共享权重GCN卷积
class AVWGCN(nn.Module):
    def __init__(self, input_size, output_size, cheb_k, num_nodes, feat_dim):
        """
        快速版 可学习邻接矩阵GCN
        借鉴了MF算法，对W进行矩阵分解，降低复杂度
        Args:
            input_size:
            output_size:
            cheb_k:
            num_nodes:
            feat_dim:
        """
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.num_nodes = num_nodes
        self.weights_pool = nn.Parameter(torch.FloatTensor(feat_dim, cheb_k, input_size, output_size))
        self.bias_pool = nn.Parameter(torch.FloatTensor(feat_dim, output_size))
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, feat_dim), requires_grad=True)

    def forward(self, x):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = self.num_nodes
        supports = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(self.node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # b, N, dim_out

        return x_gconv


class LearnedGCN(nn.Module):
    def __init__(self, node_num, input_size, output_size, feat_dim):
        """
        邻接矩阵为可学习的矩阵的GCN
        Args:
            node_num:
            input_size:
            output_size:
            feat_dim: node embedding，大小决定了可学习邻接矩阵的表达能力，越大表达能力越强，复杂度越高
        """
        super(LearnedGCN, self).__init__()
        self.node_num = node_num
        self.input_size = input_size
        self.output_size = output_size

        self.source_embed = nn.Parameter(torch.Tensor(self.node_num, feat_dim))
        self.target_embed = nn.Parameter(torch.Tensor(feat_dim, self.node_num))
        self.linear = nn.Linear(self.input_size, self.output_size)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.source_embed.size(0))
        self.source_embed.data.uniform_(-stdv, stdv)
        self.target_embed.data.uniform_(-stdv, stdv)

    def forward(self, input):
        learned_matrix = F.softmax(F.relu(torch.mm(self.source_embed, self.target_embed)), dim=1)
        output = learned_matrix.matmul(input)
        output = self.linear(output)
        return output


class ChebPolyConv(nn.Module):
    def __init__(self, k, c_in, c_out, lk, device):
        super(ChebPolyConv, self).__init__()
        self.Lk = lk
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, k).to(device))  # kernel: C_in*C_out*k
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1).to(device))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        # Lk: (Ks, num_nodes, num_nodes)
        # x:  (batch_size, c_in, input_length, num_nodes)
        # x_c: (batch_size, c_in, input_length, Ks, num_nodes)
        # theta: (c_in, c_out, Ks)
        # x_gc: (batch_size, c_out, input_length, num_nodes)
        x_c = torch.einsum("knm,bitm->bitkn", self.Lk, x)  # delete num_nodes(n)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b  # delete Ks(k) c_in(i)
        # x_in = self.align(x)  # (batch_size, c_out, input_length, num_nodes)
        return torch.relu(x_gc)


class spectralGCN(nn.Module):
    def __init__(self, K, input_size, output_size, dropout, adj_mx, device):
        super(spectralGCN, self).__init__()
        self.K = K
        self.input_size = input_size
        self.output_size = output_size
        self.drop_prob = dropout
        self.device = device
        adj_mx = adj_mx  # ndarray
        if self.K != 1:
            laplacian_mx = calculate_scaled_laplacian(adj_mx)
            self.Lk = calculate_cheb_poly(laplacian_mx, self.K)
            self._logger.info('Chebyshev_polynomial_Lk shape: ' + str(self.Lk.shape))
            self.Lk = torch.FloatTensor(self.Lk).to(self.device)
        elif self.K == 1:
            self.Lk = calculate_first_approx(adj_mx)
            self._logger.info('First_approximation_Lk shape: ' + str(self.Lk.shape))
            self.Lk = torch.FloatTensor(self.Lk).to(self.device)
            self.K = 1  # 一阶近似保留到K0和K1，但是不是数组形式，只有一个n*n矩阵，所以是1（本质上是2）

        self.gcn = ChebPolyConv(self.K, self.input_size, self.output_size, self.Lk, self.device)

    def forward(self, x):
        return self.gcn(x)


class GCONV(nn.Module):
    def __init__(self, num_nodes, max_diffusion_step, supports, device, input_dim, hid_dim, output_dim, bias_start=0.0):
        super().__init__()
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._supports = supports
        self._device = device
        self._num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Ks
        self._output_dim = output_dim
        input_size = input_dim + hid_dim
        shape = (input_size * self._num_matrices, self._output_dim)
        self.weight = torch.nn.Parameter(torch.empty(*shape, device=self._device))
        self.biases = torch.nn.Parameter(torch.empty(self._output_dim, device=self._device))
        torch.nn.init.xavier_normal_(self.weight)
        torch.nn.init.constant_(self.biases, bias_start)

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, inputs):
        # 对X(t)做图卷积，并加偏置bias
        # Reshape input and state to (batch_size, num_nodes, input_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        # (batch_size, num_nodes, input_dim)
        input_size = inputs.size(2)  # =input_dim

        x = inputs
        # T0=I x0=T0*x=x
        x0 = x.permute(1, 2, 0)  # (num_nodes, input_dim, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)  # (1, num_nodes, input_dim * batch_size)

        # 3阶[T0,T1,T2]Chebyshev多项式近似g(theta)
        # 把图卷积公式中的~L替换成了随机游走拉普拉斯D^(-1)*W
        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                # T1=L x1=T1*x=L*x
                x1 = torch.sparse.mm(support, x0)  # supports: n*n; x0: n*(total_arg_size * batch_size)
                x = self._concat(x, x1)  # (2, num_nodes, total_arg_size * batch_size)
                for k in range(2, self._max_diffusion_step + 1):
                    # T2=2LT1-T0=2L^2-1 x2=T2*x=2L^2x-x=2L*x1-x0...
                    # T3=2LT2-T1=2L(2L^2-1)-L x3=2L*x2-x1...
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)  # (3, num_nodes, total_arg_size * batch_size)
                    x1, x0 = x2, x1  # 循环
        # x.shape (Ks, num_nodes, input_dim  * batch_size)
        # Ks = len(supports) * self._max_diffusion_step + 1

        x = torch.reshape(x, shape=[self._num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, num_matrices)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * self._num_matrices])

        x = torch.matmul(x, self.weight)  # (batch_size * self._num_nodes, self._output_dim)
        x += self.biases
        # Reshape res back to 2D: (batch_size * num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * self._output_dim])


# k阶扩散卷积
class DCNN(nn.Module):
    def __init__(self, input_dim, num_units, adj_mx, max_diffusion_step, num_nodes, device, nonlinearity='tanh',
                 filter_type="laplacian"):
        """

        Args:
            input_dim: 输入维度
            num_units: 输出维度
            adj_mx: 邻接矩阵
            max_diffusion_step: 扩散步数
            num_nodes: 节点个数
            device: GPU/CPU
            nonlinearity: 激活函数
            filter_type: "laplacian", "random_walk", "dual_random_walk"
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self.input_size = input_dim
        self.num_nodes = num_nodes
        self.output_size = num_units
        self._device = device
        self._max_diffusion_step = max_diffusion_step
        self._supports = []

        supports = []
        if filter_type == "laplacian":
            supports.append(calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
            supports.append(calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support, self._device))

        self._gconv = GCONV(self.num_nodes, self._max_diffusion_step, self._supports, self._device,
                            input_dim=input_dim, hid_dim=self.output_size, output_dim=self.output_size, bias_start=0.0)

    @staticmethod
    def _build_sparse_matrix(lap, device):
        lap = lap.tocoo()
        indices = np.column_stack((lap.row, lap.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        lap = torch.sparse_coo_tensor(indices.T, lap.data, lap.shape, device=device)
        return lap

    def forward(self, inputs):
        return self._activation(self._gconv(inputs))


#
class FilterLinear(nn.Module):
    def __init__(self, device, input_dim, output_dim, in_features, out_features, filter_square_matrix, bias=True):
        """
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        """
        super(FilterLinear, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features

        self.num_nodes = filter_square_matrix.shape[0]
        self.filter_square_matrix = Variable(filter_square_matrix.repeat(output_dim, input_dim).to(device),
                                             requires_grad=False)

        self.weight = Parameter(torch.Tensor(out_features, in_features).to(device))  # [out_features, in_features]
        if bias:
            self.bias = Parameter(torch.Tensor(out_features).to(device))  # [out_features]
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')'


# 直接乘adj的K次方
class K_hopGCN(nn.Module):
    def __init__(self, K, input_size, output_size, num_nodes, adj_mx, device):
        """
        用邻接矩阵的1-K次方组成的集合作为多级邻接矩阵
        Args:
            K:
            input_size:
            output_size:
            num_nodes:
            adj_mx:
            device:
        """
        super(K_hopGCN, self).__init__()
        self.num_nodes = num_nodes
        self.input_size = input_size
        self.in_features = self.input_size * self.num_nodes
        self.output_size = output_size
        self.out_features = self.output_size * self.num_nodes
        self.K = K
        self.device = device

        self.A_list = []  # Adjacency Matrix List
        adj_mx = adj_mx
        adj_mx[adj_mx > 1e-4] = 1
        adj_mx[adj_mx <= 1e-4] = 0

        adj = torch.FloatTensor(adj_mx).to(self.device)
        adj_temp = torch.eye(self.num_nodes, self.num_nodes, device=self.device)
        for i in range(self.K):
            adj_temp = torch.matmul(adj_temp, adj)
            self.A_list.append(adj_temp)

        # a length adjustable Module List for hosting all graph convolutions
        self.gc_list = nn.ModuleList([FilterLinear(self.device, self.input_size, self.output_size, self.in_features,
                                                   self.out_features, self.A_list[i], bias=False) for i in
                                      range(self.K)])

    def step(self, x):
        # [batch_size, in_features]
        gc = self.gc_list[0](x)  # [batch_size, out_features]
        for i in range(1, self.K):
            gc = torch.cat((gc, self.gc_list[i](x)), 1)  # [batch_size, out_features * K]

        return gc


class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, adj):
        x = torch.einsum('ncwl,vw->ncvl', (x, adj))
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class MixProp(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(MixProp, self).__init__()
        self.nconv = NConv()
        self.mlp = Linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


# mixhop gcn
class MixhopGCN(nn.Module):
    def __init__(self, input_size, output_size, adj_mx, dropout, graph_depth, alpha, device):
        """
        Args:
            input_size:
            output_size:
            adj_mx: 邻接矩阵
            dropout:
            graph_depth: 跳数
            alpha: 混合比重，越高自身特征影响越大
            device:
        """
        self.input_size = input_size
        self.output_size = output_size
        self.gdep = graph_depth
        self.adj_mx = adj_mx
        self.dropout = dropout
        self.alpha = alpha
        self.gcn1 = MixProp(self.input_size, self.output_size, self.gdep, self.dropout, self.alpha).to(device)
        self.gcn2 = MixProp(self.input_size, self.output_size, self.gdep, self.dropout, self.alpha).to(device)

    def forward(self, x):
        h_out = self.gcn1(x, self.adj) + self.gcn2(x, self.adj.transpose(1, 0))
        return h_out


class ChebConvWithSAt(nn.Module):
    """
    K-order chebyshev graph convolution with spatial attention ASTGCN
    """

    def __init__(self, k, cheb_polynomials, in_channels, out_channels, num_nodes, num_timesteps):
        """
        Args:
            k(int): K-order
            cheb_polynomials: cheb_polynomials
            in_channels(int): num of channels in the input sequence
            out_channels(int): num of channels in the output sequence
        """
        super(ChebConvWithSAt, self).__init__()
        self.K = k
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).
                                                    to(self.DEVICE)) for _ in range(k)])
        self.SAt = SpatialAttentionLayer(self.DEVICE, in_channels, num_nodes, num_timesteps)

    def forward(self, x):
        """
        Chebyshev graph convolution operation

        Args:
            x: (batch_size, T, N, F_in)
            spatial_attention: (batch_size, N, N)

        Returns:
            torch.tensor: (batch_size, T, N, F_out)
        """
        batch_size, num_of_timesteps, num_of_vertices, in_channels = x.shape

        spatial_attention = self.SAt(x)
        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, time_step, :, :]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):
                t_k = self.cheb_polynomials[k]  # (N,N)

                t_k_with_at = t_k.mul(spatial_attention)  # (N,N)*(B,N,N) = (B,N,N) .mul->element-wise的乘法

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = t_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (B, N, N)(B, N, F_in) = (B, N, F_in)

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(1))  # (b, 1,N, F_out)

        return F.relu(torch.cat(outputs, dim=1))  # (b, t, N, F_out)


# adaptive gated gcn
class AGGCN(nn.Module):
    def __init__(self, node_num=524, seq_len=12, graph_dim=16,
                 choice='sp', input_dim=1, output_dim=1, num_layer=3, feat_dim=12):

        super(AGGCN, self).__init__()

        self.node_num = node_num
        self.seq_len = seq_len
        self.graph_dim = graph_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.choice = choice

        self.GATList = nn.ModuleList()
        self.GATList.append(GATConv(self.input_dim * seq_len, self.output_dim * graph_dim, heads=3, concat=False))
        for i in range(1, num_layer - 1):
            self.GATList.append(
                GATConv(self.output_dim * graph_dim, self.output_dim * graph_dim, heads=3, concat=False))
        if choice == 'sp':
            self.GATList.append(
                GATConv(self.output_dim * graph_dim, self.output_dim * graph_dim, heads=1, concat=False))
        else:
            self.GATList.append(
                GATConv(self.output_dim * graph_dim, self.output_dim * graph_dim, heads=3, concat=False))

        self.LinearList = nn.ModuleList()
        self.LinearList.append(nn.Linear(self.input_dim * seq_len, self.output_dim * self.graph_dim))
        for i in range(1, num_layer):
            self.LinearList.append(nn.Linear(self.output_dim * self.graph_dim, self.output_dim * self.graph_dim))

        self.source_embedding = nn.Parameter(torch.Tensor(self.node_num, feat_dim))
        self.target_embedding = nn.Parameter(torch.Tensor(feat_dim, self.node_num))
        self.orgin_output = nn.Linear(in_features=self.input_dim * seq_len, out_features=self.output_dim * graph_dim)

        self.num_layer = num_layer

        nn.init.xavier_uniform_(self.sp_source_embed)
        nn.init.xavier_uniform_(self.sp_target_embed)

    def forward(self, inputs, edge_index):
        """

        Args:
            inputs: [batch_size, seq_len, num_nodes, input_dim]
            edge_index: spatial

        Returns:

        """
        batch_size = inputs.shape[0]
        # [batch_size, num_nodes, input_dim*seq_len]
        h_out = inputs.permute(0, 2, 3, 1).reshape(-1, self.input_dim * self.seq_len).contiguous()
        h_out = self.seq_linear(h_out) + h_out

        learned_matrix = F.softmax(F.relu(torch.mm(self.sp_source_embed, self.sp_target_embed)), dim=1)

        for i in range(self.num_layer):
            h_cur = self.GATList[i](h_out, edge_index)
            adp_input = torch.reshape(h_out, (-1, self.node_num, self.input_dim * self.seq_len))
            adp_out = self.LinearList[i](learned_matrix.matmul(F.dropout(adp_input, p=0.1)))
            adp_out = torch.reshape(adp_out, (-1, self.output_dim * self.graph_dim))

            if i == 0:
                origin = self.orgin_output(h_out)
                h_out = torch.tanh(h_out) * torch.sigmoid(adp_out) + origin * (1 - torch.sigmoid(adp_out))
            elif i == 1:
                h_out = F.leaky_relu(h_cur) * torch.sigmoid(adp_out) + h_out * (1 - torch.sigmoid(adp_out))
            else:
                h_out = F.relu(h_cur) * torch.sigmoid(adp_out) + h_out * (1 - torch.sigmoid(adp_out))

        output = torch.reshape(h_out, (batch_size, self.node_num, self.output_dim, self.graph_dim))
        return output
