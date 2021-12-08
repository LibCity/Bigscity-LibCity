import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.sparse.linalg import eigs
from dgl.nn.pytorch import GATConv

#fixme 添加注释

class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(GraphConvolution, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.zeros((input_size, output_size), device=device, dtype=dtype),
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(output_size, device=device, dtype=dtype), requires_grad=True)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, A):
        x = torch.einsum("ijk, kl->ijl", [x, self.weight])
        x = torch.einsum("ij, kjl->kil", [A, x])
        x = x + self.bias

        return x
    
    
class GCN(nn.Module):
    # 2 layer GCN
    def __init__(self, input_size, hidden_size, output_size, device):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(input_size, hidden_size, device)
        self.gc2 = GraphConvolution(hidden_size, output_size, device)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x
    

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
    
class ChebConvWithSAt(nn.Module):
    """
    K-order chebyshev graph convolution
    Attention scale adj
    """

    def __init__(self, k, cheb_polynomials, in_channels, out_channels):
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

    def forward(self, x, spatial_attention):
        """
        Chebyshev graph convolution operation

        Args:
            x: (batch_size, N, F_in, T)
            spatial_attention: (batch_size, N, N)

        Returns:
            torch.tensor: (batch_size, N, F_out, T)
        """
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                t_k = self.cheb_polynomials[k]  # (N,N)

                t_k_with_at = t_k.mul(spatial_attention)   # (N,N)*(B,N,N) = (B,N,N) .mul->element-wise的乘法

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = t_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (B, N, N)(B, N, F_in) = (B, N, F_in)

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)

    

