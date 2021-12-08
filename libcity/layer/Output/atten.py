import torch.nn as nn
import torch
from Output.mlp import FC
import torch.nn.functional as F
from torch.autograd import Variable
from spectral import SpectralNorm
import numpy as np

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B C W H)
            returns :
                out : self attention value + input feature 
                attention: B N N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention
    
class TransformAttention(nn.Module):
    def __init__(self, num_heads, dim, bn, bn_decay, device):
        super(TransformAttention, self).__init__()
        self.K = num_heads
        self.D = dim
        
        assert self.D%self.K==0
        
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
        '''
        transform attention mechanism
        x:      (batch_size, input_length, num_nodes, D)
        ste_1:  (batch_size, input_length, num_nodes, D)
        ste_2:  (batch_size, output_length, num_nodes, D)
        return: (batch_size, output_length, num_nodes, D)
        '''
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