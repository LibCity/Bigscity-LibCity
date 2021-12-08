import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Output.mlp import FC

class SpatialAttention(nn.Module):
    def __init__(self, num_heads, dim, bn, bn_decay, device):
        super(SpatialAttention, self).__init__()
        #k num_heads
        self.num_heads = num_heads
        self.D = dim
        
        assert self.D%self.num_heads==0
        
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
        '''
        spatial attention mechanism
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

        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= self.d ** 0.5
        attention = torch.softmax(attention, dim=-1)  # (K*batch_size, num_step, num_nodes, num_nodes)

        x = torch.matmul(attention, value)
        x = torch.cat(torch.split(x, x.size(0) // self.num_heads, dim=0), dim=-1)
        x = self.output_fc(x)  # (batch_size, num_steps, num_nodes, D)
        return x