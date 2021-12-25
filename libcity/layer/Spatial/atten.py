import torch
import torch.nn as nn
import torch.nn.functional as F
from Output.mlp import FC


class SpatialAttention(nn.Module):
    def __init__(self, num_heads, dim, bn, bn_decay, device):
        super(SpatialAttention, self).__init__()
        # k num_heads
        self.num_heads = num_heads
        self.D = dim
        assert self.D % self.num_heads == 0
        self.d = self.D / self.num_heads
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.input_query_fc = FC(input_dims=2 * self.D, units=self.D, activations=nn.ReLU,
                                 bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.input_key_fc = FC(input_dims=2 * self.D, units=self.D, activations=nn.ReLU,
                               bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.input_value_fc = FC(input_dims=2 * self.D, units=self.D, activations=nn.ReLU,
                                 bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.output_fc = FC(input_dims=self.D, units=[self.D, self.D], activations=[nn.ReLU, None],
                            bn=self.bn, bn_decay=self.bn_decay, device=self.device)

    def forward(self, x, ste):
        """
        spatial attention mechanism

        Args:
            x: shape (batch_size, num_step, num_nodes, D)
            ste: shape (batch_size, num_step, num_nodes, D)

        Returns:
            tensor: shape (batch_size, num_step, num_nodes, D)
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

        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= self.d ** 0.5
        attention = torch.softmax(attention, dim=-1)  # (K*batch_size, num_step, num_nodes, num_nodes)

        x = torch.matmul(attention, value)
        x = torch.cat(torch.split(x, x.size(0) // self.num_heads, dim=0), dim=-1)
        x = self.output_fc(x)  # (batch_size, num_steps, num_nodes, D)
        return x


class SpatialAttentionLayer(nn.Module):
    """
    compute spatial attention scores,chebploy graph attention
    """

    def __init__(self, device, in_channels, num_of_vertices, num_of_timesteps):
        super(SpatialAttentionLayer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(device))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(device))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(device))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(device))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(device))

    def forward(self, x):
        """
        Args:
            x(torch.tensor): (B,T ,N, F_in)

        Returns:
            torch.tensor: (B,N,N)
        """
        # x --> (b n f t)
        # x * W1 --> (B,N,F,T)(T)->(B,N,F)
        # x * W1 * W2 --> (B,N,F)(F,T)->(B,N,T)
        x=x.permute(0, 2, 3, 1)

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)
        # (W3 * x) ^ T --> (F)(B,N,F,T)->(B,N,T)-->(B,T,N)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)
        # x = lhs * rhs --> (B,N,T)(B,T,N) -> (B, N, N)
        product = torch.matmul(lhs, rhs)
        # S = Vs * sig(x + bias) --> (N,N)(B,N,N)->(B,N,N)
        s = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        # softmax (B,N,N)
        s_normalized = F.softmax(s, dim=1)
        return s_normalized
