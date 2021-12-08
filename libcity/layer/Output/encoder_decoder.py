import scipy.sparse as sp
from scipy.sparse import linalg
import numpy as np
import torch
import torch.nn as nn
from Spatial.atten import SpatialAttention
from Temporal.atten import TemporalAttention
from utils import GatedFusion

# class BaseEncoderLayer(nn.Module):
#     def __init__(self,config):
#         self.num_nodes = self.data_feature.get('num_nodes', 1)
#     def forward(self,x):
#         pass
#     def cell();
#         继承实现
#         pass
    
# class BaseDecoderLayer(nn.Module):
#     def __init__(self,config):
#         pass
    
#     def forward(self,x):
#         pass
    
#     def cell():
#         继承实现
#         pass

class STAttBlock(nn.Module):
    def __init__(self, K, D, bn, bn_decay, device, mask=True):
        super(STAttBlock, self).__init__()
        self.K = K
        self.D = D
        self.d = self.D / self.K
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.mask = mask
        self.sp_att = SpatialAttention(num_heads=self.K, dim=self.D, bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.temp_att = TemporalAttention(num_heads=self.K, dim=self.D, bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.gated_fusion = GatedFusion(dim=self.D, bn=self.bn, bn_decay=self.bn_decay, device=self.device)

    def forward(self, x, ste):
        HS = self.sp_att(x, ste)
        HT = self.temp_att(x, ste)
        H = self.gated_fusion(HS, HT)
        return torch.add(x, H)

class GMANEncoderLayer(nn.Module):
    def __init__(self,K, D, device,bn, bn_decay,L):
        super(GMANEncoderLayer,self).__init__()
        # get data feature
        self.D = D 
        self.K = K
        self.device = device
        self.bn = bn
        self.bn_decay = bn_decay
        self.L = L
        self.encoder = nn.ModuleList()
        for _ in range(self.L):
            self.encoder.append(STAttBlock(K=self.K, D=self.D, bn=self.bn, bn_decay=self.bn_decay, device=self.device))

    def forward(self,x,ste_p):
        for encoder_layer in self.encoder:
            x = encoder_layer(x, ste_p)
        return x
    
class GMANDecoderLayer(nn.Module):
    def __init__(self,K, D, device,bn, bn_decay,L):
        super(GMANDecoderLayer,self).__init__()
        self.D = D 
        self.K = K
        self.device = device
        self.bn = bn
        self.bn_decay = bn_decay
        self.L = L
        self.decoder = nn.ModuleList()
        for _ in range(self.L):
            self.decoder.append(STAttBlock(K=self.K, D=self.D, bn=self.bn, bn_decay=self.bn_decay, device=self.device))
        
    def forward(self,x,ste_q):
        for decoder_layer in self.decoder:
            x = decoder_layer(x, ste_q)                
        return x
        
        





        