import math

import torch.nn as nn
import torch
from Output.mlp import FC


class TransformAttention(nn.Module):
    def __init__(self, num_heads, dim, bn, bn_decay, device):
        super(TransformAttention, self).__init__()
        self.K = num_heads
        self.D = dim
        assert self.D % self.K == 0
        self.d = self.D / self.K
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.input_query_fc = FC(input_dims=self.D, units=self.D, activations=nn.ReLU,
                                 bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.input_key_fc = FC(input_dims=self.D, units=self.D, activations=nn.ReLU,
                               bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.input_value_fc = FC(input_dims=self.D, units=self.D, activations=nn.ReLU,
                                 bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.output_fc = FC(input_dims=self.D, units=[self.D, self.D], activations=[nn.ReLU, None],
                            bn=self.bn, bn_decay=self.bn_decay, device=self.device)

    def forward(self, x, ste1, ste2):
        """
        transform attention mechanism

        Args:
            x: shape (batch_size, input_length, num_nodes, D)
            ste_1: shape (batch_size, input_length, num_nodes, D)
            ste_2: shape (batch_size, output_length, num_nodes, D)

        Returns:
            tensor: shape (batch_size, output_length, num_nodes, D)

        """
        # query: (batch_size, output_length, num_nodes, D)
        # key:   (batch_size, input_length, num_nodes, D)
        # value: (batch_size, input_length, num_nodes, D)
        query = self.input_query_fc(ste2)
        key = self.input_key_fc(ste1)
        value = self.input_value_fc(x)
        # query: (K*batch_size, output_length, num_nodes, d/k)
        # key:   (K*batch_size, input_length, num_nodes, d/k)
        # value: (K*batch_size, input_length, num_nodes, d/k)
        query = torch.cat(torch.split(query, query.size(-1) // self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, key.size(-1) // self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, value.size(-1) // self.K, dim=-1), dim=0)
        # query: (K*batch_size, num_nodes, output_length, d/k)
        # key:   (K*batch_size, num_nodes, d/k, input_length)
        # value: (K*batch_size, num_nodes, input_length, d/k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2).transpose(2, 3)
        value = value.transpose(1, 2)

        attention = torch.matmul(query, key)
        attention /= self.d ** 0.5
        attention = torch.softmax(attention, dim=-1)  # (K*batch_size, num_nodes, output_length, input_length)

        x = torch.matmul(attention, value)
        x = x.transpose(1, 2)
        x = torch.cat(torch.split(x, x.size(0) // self.K, dim=0), dim=-1)
        x = self.output_fc(x)  # (batch_size, output_length, num_nodes, D)
        return x


# fixme
class RnnAttnOutput(nn.Module):
    """ Attention 注意力机制模块, 对 Rnn 中间层输出做加权平均. """

    def __init__(self, hidden_size,output_feats):
        """ 初始化.
        Args:
            hidden_size (int): 中间层输出向量的大小
        """

        super(RnnAttnOutput, self).__init__()
        self.sqrt_rec_size = 1. / math.sqrt(hidden_size)
        self.output_feats=output_feats
        # context vector
        self.u = nn.Linear(in_features=hidden_size, out_features=output_feats, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        """ 前向传播.
        Args:
            x (torch.tensor.Tensor): shape (batch, seq_len, hidden_size) or (batch,T,n,hidden_size)中间层输出序列
        Returns:
            (torch.tensor.Tensor): shape (batch, o,size) or (batch,o,n,hidden_size)
        """
        if len(x.shape) == 4:
            batch = x.shape[0]
            nodes = x.shape[2]
            x = x.view(-1, nodes, x.shape[3]).contiguous()
        w = self.u(x) * self.sqrt_rec_size
        w = w.permute(0, 2, 1)
        w = self.softmax(w)  # batch_size * o *seq_len
        c = torch.bmm(w, x)
        if len(x.shape) == 4:
            c = c.view(batch, self.output_feats,nodes, -1)
        return c
