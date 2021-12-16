import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import numpy as np

#fixme 添加注释
from libcity.layer.utils import calculate_scaled_laplacian, calculate_cheb_poly, calculate_first_approx, \
    calculate_random_walk_matrix, calculate_normalized_laplacian, sym_adj, asym_adj


class GAT(nn.Module):
    # 2 layer GAT
    def __init__(self, g, in_dim, hidden_dim , out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = GATConv(g , in_dim, hidden_dim, num_heads)
        self.layer2 = GATConv(g, hidden_dim*num_heads, out_dim, 1)
        self.g=g

    def forward(self,h):
        h = self.layer1(self.g,h)
        h = F.elu(h)
        h = self.layer2(self.g,h)
        return h

    
#可训练邻接矩阵+非共享权重GCN卷积
class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k,num_nodes,embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.num_nodes=num_nodes
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)

    def forward(self, x):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = self.num_nodes
#         print("node_num：" +str(node_num))
        supports = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
#         print("suppoorts")
#         print(supports.shape)
#         print(x.shape)
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(self.node_embeddings, self.bias_pool)                       # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     # b, N, dim_out
        
        return x_gconv
    
    
class LearnedGCN(nn.Module):
    def __init__(self, node_num, in_feature, out_feature,feat_dim):
        super(LearnedGCN, self).__init__()
        self.node_num = node_num
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.source_embed = nn.Parameter(torch.Tensor(self.node_num, feat_dim))
        self.target_embed = nn.Parameter(torch.Tensor(feat_dim, self.node_num))
        self.linear = nn.Linear(self.in_feature, self.out_feature)
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
    def __init__(self, ks, c_in, c_out, lk, device):
        super(ChebPolyConv, self).__init__()
        self.Lk = lk
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks).to(device))  # kernel: C_in*C_out*ks
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

class SpectrumGCN(nn.Module):
    def __init__(self,config):
        self.K = config.get('K', 3)
        self.input_size = config.get('input_size', 1)
        self.output_size = config.get('output_size', 1)
        self.drop_prob = config.get('dropout', 0)
        self.device = config.get('device')
        # self.num_layer=config.get('num_layer',1)
        adj_mx = config.get('adj_mx')  # ndarray
        if self.K != 1:
            laplacian_mx = calculate_scaled_laplacian(adj_mx)
            self.Lk = calculate_cheb_poly(laplacian_mx, self.K)
            self._logger.info('Chebyshev_polynomial_Lk shape: ' + str(self.Lk.shape))
            self.Lk = torch.FloatTensor(self.Lk).to(self.device)
        elif self.K == 1 :
            self.Lk = calculate_first_approx(adj_mx)
            self._logger.info('First_approximation_Lk shape: ' + str(self.Lk.shape))
            self.Lk = torch.FloatTensor(self.Lk).to(self.device)
            self.K = 1  # 一阶近似保留到K0和K1，但是不是数组形式，只有一个n*n矩阵，所以是1（本质上是2）

        self.gcn = ChebPolyConv(self.K,self.input_size,self.output_size,self.Lk,self.device)

    def forward(self,x):
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

    def forward(self, inputs, state):
        # 对X(t)和H(t-1)做图卷积，并加偏置bias
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        # (batch_size, num_nodes, total_arg_size(input_dim+state_dim))
        input_size = inputs.size(2)  # =total_arg_size

        x = inputs
        # T0=I x0=T0*x=x
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)  # (1, num_nodes, total_arg_size * batch_size)

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
        # x.shape (Ks, num_nodes, total_arg_size * batch_size)
        # Ks = len(supports) * self._max_diffusion_step + 1

        x = torch.reshape(x, shape=[self._num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, num_matrices)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * self._num_matrices])

        x = torch.matmul(x, self.weight)  # (batch_size * self._num_nodes, self._output_dim)
        x += self.biases
        # Reshape res back to 2D: (batch_size * num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * self._output_dim])

# k阶扩散卷积
class DCGRUCell(nn.Module):
    def __init__(self, input_dim, num_units, adj_mx, max_diffusion_step, num_nodes, device, nonlinearity='tanh',
                 filter_type="laplacian"):
        """

        Args:
            input_dim:
            num_units:
            adj_mx:
            max_diffusion_step:
            num_nodes:
            device:
            nonlinearity:
            filter_type: "laplacian", "random_walk", "dual_random_walk"
            use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
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

        self._gconv = GCONV(self._num_nodes, self._max_diffusion_step, self._supports, self._device,
                            input_dim=input_dim, hid_dim=self._num_units, output_dim=self._num_units, bias_start=0.0)

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
class TGCLSTM(nn.Module):
    def __init__(self, config, data_feature):
        super(TGCLSTM, self).__init__(config, data_feature)
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.input_dim = self.data_feature.get('feature_dim', 1)
        self.in_features = self.input_dim * self.num_nodes
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.out_features = self.output_dim * self.num_nodes

        self.K = config.get('K_hop_numbers', 3)
        self.back_length = config.get('back_length', 3)
        self.device = config.get('device', torch.device('cpu'))

        self.A_list = []  # Adjacency Matrix List
        adj_mx = config.get('adj')
        adj_mx[adj_mx > 1e-4] = 1
        adj_mx[adj_mx <= 1e-4] = 0

        adj = torch.FloatTensor(adj_mx).to(self.device)
        adj_temp = torch.eye(self.num_nodes, self.num_nodes, device=self.device)
        for i in range(self.K):
            adj_temp = torch.matmul(adj_temp, adj)
            if config.get('Clamp_A', True):
                # confine elements of A
                adj_temp = torch.clamp(adj_temp, max=1.)
            if self.dataset_class == "TGCLSTMDataset":
                self.A_list.append(
                    torch.mul(adj_temp, torch.Tensor(data_feature['FFR'][self.back_length])
                              .to(self.device)))
            else:
                self.A_list.append(adj_temp)

        # a length adjustable Module List for hosting all graph convolutions
        self.gc_list = nn.ModuleList([FilterLinear(self.device, self.input_dim, self.output_dim, self.in_features,
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
        self.mlp = Linear((gdep+1)*c_in, c_out)
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
            h = self.alpha*x + (1-self.alpha)*self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class MixhopGCN(nn.Module):
    def __init__(self, config):
        self.input_size = config.get('input_size')
        self.output_size = config.get('output_size')
        self.gdep = config.get('graph_depth')
        self.adj = config.get('adj')
        self.dropout = config.get('dropout')
        self.alpha = config.get('alpha')
        self.gcn1 = MixProp(self.input_size, self.output_size, self.gdep, self.dropout, self.alpha)
        self.gcn2 = MixProp(self.input_size, self.output_size, self.gdep, self.dropout, self.alpha)

    def forward(self, x):
        h_out = self.gcn1(x, self.adj) + self.gcn2(x, self.adj.transpose(1, 0))
        return h_out

