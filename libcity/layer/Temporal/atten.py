import torch
import torch.nn as nn
import torch.nn.functional as F
from Output.mlp import FC


# LSTM-attention
from libcity.layer.utils import cal_attention


class TemporalAttentionLayer(nn.Module):
    def __init__(self, device, in_channels, num_of_vertices, num_of_timesteps):
        super(TemporalAttentionLayer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(device))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(device))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(device))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(device))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(device))

    def forward(self, x):
        """
        Args:
            x: (batch_size, N, F_in, T)

        Returns:
            torch.tensor: (B, T, N, F_out)
        """

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        e = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        e_normalized = F.softmax(e, dim=1)

        o = torch.matmul(e_normalized,x)

        return o

class TemporalAttention(nn.Module):
    def __init__(self, num_heads, input_size, bn, bn_decay, device, mask=True):
        super(TemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.input_size = input_size
        assert self.input_size % self.num_heads == 0
        self.d = self.input_size / self.num_heads
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.mask = mask
        self.input_query_fc = FC(input_dims=2 * self.input_size, units=self.input_size, activations=nn.ReLU,
                                 bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.input_key_fc = FC(input_dims=2 * self.input_size, units=self.input_size, activations=nn.ReLU,
                               bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.input_value_fc = FC(input_dims=2 * self.input_size, units=self.input_size, activations=nn.ReLU,
                                 bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.output_fc = FC(input_dims=self.input_size, units=[self.input_size, self.input_size], activations=[nn.ReLU, None],
                            bn=self.bn, bn_decay=self.bn_decay, device=self.device)

    def forward(self, x, ste):
        """
        temporal attention mechanism
        x:      (batch_size, num_step, num_nodes, D)
        ste:    (batch_size, num_step, num_nodes, D)
        return: (batch_size, num_step, num_nodes, D)
        """
        x = torch.cat((x, ste), dim=-1)
        # (batch_size, num_step, num_nodes, D)
        query = self.input_query_fc(x)
        key = self.input_key_fc(x)
        value = self.input_value_fc(x)
        # (K*batch_size, num_step, num_nodes, d/k)
        query = torch.cat(torch.split(query, query.size(-1) // self.num_heads, dim=-1), dim=0)
        key = torch.cat(torch.split(key, key.size(-1) // self.num_heads, dim=-1), dim=0)
        value = torch.cat(torch.split(value, value.size(-1) // self.num_heads, dim=-1), dim=0)
        # query: (K*batch_size, num_nodes, num_step, d/k)
        # key:   (K*batch_size, num_nodes, d/k, num_step)
        # value: (K*batch_size, num_nodes, num_step, d/k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2).transpose(2, 3)
        value = value.transpose(1, 2)

        attention = torch.matmul(query, key)
        attention /= self.d ** 0.5  # (K*batch_size, num_nodes, num_step, num_step)
        if self.mask:
            batch_size = x.size(0)
            num_step = x.size(1)
            num_nodes = x.size(2)
            mask = torch.ones((num_step, num_step), device=self.device)
            mask = torch.tril(mask)
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.repeat(self.num_heads * batch_size, num_nodes, 1, 1)
            mask = mask.bool().int()
            mask_rev = -(mask - 1)
            attention = mask * attention + mask_rev * torch.full(attention.shape, -2 ** 15 + 1, device=self.device)
        attention = torch.softmax(attention, dim=-1)

        x = torch.matmul(attention, value)
        x = x.transpose(1, 2)
        x = torch.cat(torch.split(x, x.size(0) // self.num_heads, dim=0), dim=-1)
        x = self.output_fc(x)  # (batch_size, output_length, num_nodes, D)
        return x

class MSA(nn.Module):
    """
    Multi-space attention
    """

    def __init__(self, d, n_h, self_att=True):
        """
        Args:
            d: d-dimension representations after B-layer CNN/FCN
            n_h: num of head
            self_att: whether use self attention
        """
        super(MSA, self).__init__()
        self.d = d
        self.n_h = n_h
        self.d_h = d / n_h
        self.self_att = self_att

        assert d % n_h == 0
        self.d_h = d // n_h

        if self_att:
            self.wx = nn.Linear(in_features=d, out_features=d * 3)
        else:
            self.wq = nn.Linear(in_features=d, out_features=d)
            self.wkv = nn.Linear(in_features=d, out_features=d * 2)

        self.wo = nn.Linear(in_features=d, out_features=d)

    def split_heads(self, x):
        """

        Args:
            x: shape == [batch_size, input_window, N, d]

        Returns:
                shape == [batch_size, input_window, n_h, N, d_h]
        """
        shape = x.shape
        x = torch.reshape(x, [shape[0], shape[1], shape[2], self.n_h, self.d_h])
        return x.permute([0, 1, 3, 2, 4])

    def forward(self, V, K, Q, M):

        # linear
        if self.self_att:
            wx_o = self.wx(Q)
            Q, K, V = torch.split(tensor=wx_o, split_size_or_sections=wx_o.shape[-1] // 3, dim=-1)
        else:
            Q = self.wq(Q)
            wkv_o = self.wkv(K)
            K, V = torch.split(tensor=wkv_o, split_size_or_sections=wkv_o.shape[-1] // 2, dim=-1)

        # split head
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        scaled_attention = cal_attention(Q=Q, K=K, V=V, M=M, n_h=self.n_h)

        scaled_attention = scaled_attention.permute([0, 1, 3, 2, 4])

        d_shape = scaled_attention.shape
        concat_attention = torch.reshape(
            scaled_attention, (d_shape[0], d_shape[1], d_shape[2], self.d))

        output = self.wo(concat_attention)
        return output


class MSA(nn.Module):
    """
    Multi-space attention
    """

    def __init__(self, d, n_h, self_att=True):
        """
        Args:
            d: d-dimension representations after B-layer CNN/FCN
            n_h: num of head
            self_att: whether use self attention
        """
        super(MSA, self).__init__()
        self.d = d
        self.n_h = n_h
        self.d_h = d / n_h
        self.self_att = self_att

        assert d % n_h == 0
        self.d_h = d // n_h

        if self_att:
            self.wx = nn.Linear(in_features=d, out_features=d * 3)
        else:
            self.wq = nn.Linear(in_features=d, out_features=d)
            self.wkv = nn.Linear(in_features=d, out_features=d * 2)

        self.wo = nn.Linear(in_features=d, out_features=d)

    def split_heads(self, x):
        """

        Args:
            x: shape == [batch_size, input_window, N, d]

        Returns:
                shape == [batch_size, input_window, n_h, N, d_h]
        """
        shape = x.shape
        x = torch.reshape(x, [shape[0], shape[1], shape[2], self.n_h, self.d_h])
        return x.permute([0, 1, 3, 2, 4])

    def forward(self, V, K, Q, M):

        # linear
        if self.self_att:
            wx_o = self.wx(Q)
            Q, K, V = torch.split(tensor=wx_o, split_size_or_sections=wx_o.shape[-1] // 3, dim=-1)
        else:
            Q = self.wq(Q)
            wkv_o = self.wkv(K)
            K, V = torch.split(tensor=wkv_o, split_size_or_sections=wkv_o.shape[-1] // 2, dim=-1)

        # split head
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        scaled_attention = cal_attention(Q=Q, K=K, V=V, M=M, n_h=self.n_h)

        scaled_attention = scaled_attention.permute([0, 1, 3, 2, 4])

        d_shape = scaled_attention.shape
        concat_attention = torch.reshape(
            scaled_attention, (d_shape[0], d_shape[1], d_shape[2], self.d))

        output = self.wo(concat_attention)
        return output
