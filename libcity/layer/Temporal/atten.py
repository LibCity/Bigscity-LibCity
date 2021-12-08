import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Output.mlp import FC

    
class TemporalAttention(nn.Module):
    def __init__(self, num_heads, dim, bn, bn_decay, device, mask=True):
        super(TemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.D = dim
        assert self.D%self.num_heads==0
        self.d = self.D / self.num_heads
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.mask = mask
        self.input_query_fc = FC(input_dims=2 * self.D, units=self.D, activations=nn.ReLU,
                                 bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.input_key_fc = FC(input_dims=2 * self.D, units=self.D, activations=nn.ReLU,
                               bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.input_value_fc = FC(input_dims=2 * self.D, units=self.D, activations=nn.ReLU,
                                 bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.output_fc = FC(input_dims=self.D, units=[self.D, self.D], activations=[nn.ReLU, None],
                            bn=self.bn, bn_decay=self.bn_decay, device=self.device)

    def forward(self, x, ste):
        '''
        temporal attention mechanism
        x:      (batch_size, num_step, num_nodes, D)
        ste:    (batch_size, num_step, num_nodes, D)
        return: (batch_size, num_step, num_nodes, D)
        '''
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


