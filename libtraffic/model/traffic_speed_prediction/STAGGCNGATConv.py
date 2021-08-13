import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


from torch_scatter import scatter


def remove_self_loops(edge_index: torch.Tensor):
    return edge_index[:, edge_index[0] != edge_index[1]]


def maybe_num_nodes(edge_index: torch.Tensor, num_nodes: Optional[int] = None):
    if num_nodes is not None:
        return num_nodes
    else:
        return int(edge_index.max()) + 1


def add_self_loops(edge_index: torch.Tensor, num_nodes: Optional[int] = None):
    return torch.cat((edge_index,
                      torch.arange(maybe_num_nodes(edge_index, num_nodes))
                           .repeat(2, 1)
                           .to(edge_index.device)), dim=1)


def softmax(x: torch.Tensor, index: torch.Tensor, num_nodes: Optional[int] = None, dim: int = 0):
    N = maybe_num_nodes(index, num_nodes)
    x_max = scatter(x, index, dim, dim_size=N, reduce='max').index_select(dim, index)
    out = (x - x_max).exp()
    out_sum = scatter(out, index, dim, dim_size=N, reduce='sum').index_select(dim, index)
    return out / out_sum


class GATConv(nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int,
                 heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True):
        super(GATConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_heads = heads

        self.negative_slope = negative_slope
        self.dropout = dropout

        self.bias = bias
        self.concat = concat
        self.add_self_loops = add_self_loops

        self.linear = nn.Linear(self.in_channels, self.attn_heads * self.out_channels, bias=False)
        self.attn_j = nn.Parameter(torch.Tensor(1, self.attn_heads, self.out_channels))
        self.attn_i = nn.Parameter(torch.Tensor(1, self.attn_heads, self.out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.attn_heads * self.out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.init_weights()

    def init_weights(self):
        self._glorot(self.linear.weight)
        self._glorot(self.attn_j)
        self._glorot(self.attn_i)
        self._zeros(self.bias)

    @staticmethod
    def _glorot(t: torch.Tensor):
        if t is None:
            return
        stdv = math.sqrt(6. / (t.size(-2) * t.size(-1)))
        t.data.uniform_(-stdv, stdv)

    @staticmethod
    def _zeros(t: torch.Tensor):
        if t is None:
            return
        t.data.fill_(0.)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        num_nodes = x.size(0)

        edge_index = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)

        edge_index_j, edge_index_i = edge_index

        # x: [num_nodes, num_features]
        # [num_edges, attn_heads, out_channels]
        x_j = self.linear(x).view(-1, self.attn_heads, self.out_channels)[edge_index_j]
        x_i = self.linear(x).view(-1, self.attn_heads, self.out_channels)[edge_index_i]

        # [num_edges, attn_heads]
        alpha_j = (x_j * self.attn_j).sum(dim=-1)[edge_index_j]
        alpha_i = (x_i * self.attn_i).sum(dim=-1)[edge_index_i]

        # message passing
        # [num_edges, attn_heads]
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, x_i.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # [num_edges, attn_heads, out_channels]
        message = x_j * alpha.unsqueeze(-1)

        out = scatter(message, edge_index_i, dim=0, reduce='add')

        if self.concat:
            out = out.view(-1, self.attn_heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out += self.bias

        return out
