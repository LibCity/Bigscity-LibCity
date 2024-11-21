from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss

import math


class pointCoder(nn.Module):
    def __init__(self, input_size, patch_count, weights=(1., 1., 1.), tanh=True):
        super().__init__()
        self.input_size = input_size
        self.patch_count = patch_count
        self.weights = weights
        # self._generate_anchor()
        self.tanh = tanh

    def _generate_anchor(self, device="cpu"):
        anchors = []
        patch_stride_x = 2. / self.patch_count
        for i in range(self.patch_count):
            x = -1 + (0.5 + i) * patch_stride_x
            anchors.append([x])
        anchors = torch.as_tensor(anchors)
        self.anchor = torch.as_tensor(anchors, device=device)
        # self.register_buffer("anchor", anchors)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, pts, model_offset=None):
        assert model_offset is None
        self.boxes = self.decode(pts)
        return self.boxes

    def decode(self, rel_codes):
        # print ('xyxy decoding')
        boxes = self.anchor
        pixel = 1. / self.patch_count
        wx, wy = self.weights

        dx = F.tanh(rel_codes[:, :, 0] / wx) * pixel if self.tanh else rel_codes[:, :, 0] * pixel / wx
        dy = F.tanh(rel_codes[:, :, 1] / wy) * pixel if self.tanh else rel_codes[:, :, 1] * pixel / wy

        pred_boxes = torch.zeros_like(rel_codes)

        ref_x = boxes[:, 0].unsqueeze(0)
        ref_y = boxes[:, 1].unsqueeze(0)

        pred_boxes[:, :, 0] = dx + ref_x
        pred_boxes[:, :, 1] = dy + ref_y
        pred_boxes = pred_boxes.clamp_(min=-1., max=1.)

        return pred_boxes

    def get_offsets(self):
        return (self.boxes - self.anchor) * self.input_size


class pointwhCoder(pointCoder):
    def __init__(self, input_size, patch_count, weights=(1., 1., 1.), pts=1, tanh=True, wh_bias=None,
                 deform_range=0.25):
        super().__init__(input_size=input_size, patch_count=patch_count, weights=weights, tanh=tanh)
        self.patch_pixel = pts
        self.wh_bias = None
        if wh_bias is not None:
            self.wh_bias = nn.Parameter(torch.zeros(2) + wh_bias)
        self.deform_range = deform_range

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, boxes):
        self._generate_anchor(device=boxes.device)
        # print(boxes.shape)
        # print(self.wh_bias.shape)
        if self.wh_bias is not None:
            boxes[:, :, 1:] = boxes[:, :, 1:] + self.wh_bias
        self.boxes = self.decode(boxes)
        points = self.meshgrid(self.boxes)
        return points

    def decode(self, rel_codes):
        # print ('xyxy decoding')
        boxes = self.anchor
        pixel_x = 2. / self.patch_count  # patch_count=in_size//stride 这里应该用2除而不是1除 得到pixel_x是两个patch中点的原本距离
        wx, ww1, ww2 = self.weights

        dx = F.tanh(rel_codes[:, :, 0] / wx) * pixel_x / 4 if self.tanh else rel_codes[:, :,
                                                                             0] * pixel_x / wx  # 中心点不会偏移超过patch_len

        dw1 = F.relu(F.tanh(rel_codes[:, :,
                            1] / ww1)) * pixel_x * self.deform_range + pixel_x  # 中心点左边长度在[stride,stride+1/4*stride]，右边同理
        dw2 = F.relu(F.tanh(rel_codes[:, :, 2] / ww2)) * pixel_x * self.deform_range + pixel_x  #
        # dw =

        pred_boxes = torch.zeros((rel_codes.shape[0], rel_codes.shape[1], rel_codes.shape[2] - 1)).to(rel_codes.device)

        ref_x = boxes[:, 0].unsqueeze(0)

        pred_boxes[:, :, 0] = dx + ref_x - dw1
        pred_boxes[:, :, 1] = dx + ref_x + dw2
        pred_boxes = pred_boxes.clamp_(min=-1., max=1.)

        return pred_boxes

    def meshgrid(self, boxes):
        B = boxes.shape[0]
        xs = boxes
        xs = torch.nn.functional.interpolate(xs, size=self.patch_pixel, mode='linear', align_corners=True)
        results = xs
        results = results.reshape(B, self.patch_count, self.patch_pixel, 1)
        # print((1+results[0])/2*336)
        return results


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


# decomposition

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# pos_encoding

def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


SinCosPosEncoding = PositionalEncoding


def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (
                    torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += .001
        else:
            x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model))  # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos':
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)


def generate_pairs(n):
    pairs = []
    
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs.append([i, j])
    
    return np.array(pairs)


def cal_PSI(x, r):
    #[bs x nvars x patch_len x patch_num]
    x = x.permute(0,1,3,2)
    batch, n_vars, patch_num, patch_len = x.shape
    x = x.reshape(batch*n_vars, patch_num, patch_len)
    # Generate all possible pairs of patch_num indices within each batch
    pairs = generate_pairs(patch_num)
    # Calculate absolute differences between pairs of sequences
    abs_diffs = torch.abs(x[:, pairs[:, 0], :] - x[:, pairs[:, 1], :])
    # Find the maximum absolute difference for each pair of sequences
    max_abs_diffs = torch.max(abs_diffs, dim=-1).values 
    max_abs_diffs = max_abs_diffs.reshape(-1,patch_num,patch_num-1)
    # Count the number of pairs with max absolute difference less than r
    c = torch.log(1+torch.mean((max_abs_diffs < r).float(),dim=-1))
    psi = torch.mean(c,dim=-1)
    return psi

def cal_PaEn(lfp,lep,r,lambda_):
    psi_lfp = cal_PSI(lfp,r)
    psi_lep = cal_PSI(lep,r)
    psi_diff = psi_lfp - psi_lep
    lep = lep.permute(0,1,3,2)
    batch, n_vars, patch_num, patch_len = lep.shape    
    lep = lep.reshape(batch*n_vars, patch_num, patch_len)
    sum_x = torch.sum(lep, dim=[-2,-1])
    PaEN_loss = torch.mean(sum_x*psi_diff)*lambda_ # update parameters with REINFORCE
    return PaEN_loss


class HDMixer_backbone(nn.Module):
    def __init__(self,configs, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int,
                 n_layers:int=3, d_model=128, n_heads=16, dropout:float=0., act:str="gelu",
                 pe:str='zeros', learn_pe:bool=True, head_dropout = 0, padding_patch = None,
                 head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        self.deform_patch = configs.deform_patch
        if self.deform_patch:
            self.patch_len = patch_len
            self.stride = stride
            self.patch_num = patch_num = context_window//self.stride
            self.patch_shift_linear = nn.Linear(context_window, self.patch_num*3)
            self.box_coder = pointwhCoder(input_size=context_window, patch_count=self.patch_num,
                                          weights=(1.,1.,1.), pts=self.patch_len, tanh=True,
                                          wh_bias=torch.tensor(5./3.).sqrt().log(),deform_range=configs.deform_range)
            self.lambda_ = configs.lambda_
            self.r = configs.r
        else:
                # Patching
            self.patch_len = patch_len
            self.stride = stride
            self.padding_patch = padding_patch
            patch_num = int((context_window - patch_len)/stride + 1) # patch的数量是(336-16)/8 + 1 向下取整
            if padding_patch == 'end': # can be modified to general case
                # 使用了 PyTorch 中的 nn.ReplicationPad1d 模块，用于对一维张量进行复制填充操作
                self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
                patch_num += 1
        # Backbone 
        self.backbone = Encoder(configs, c_in, patch_num=patch_num, patch_len=patch_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, dropout=dropout, act=act,
                                pe=pe, learn_pe=learn_pe)

        # Head
        self.head_nf = d_model * patch_num 
        self.n_vars = c_in
        self.head_type = head_type
        self.individual = individual
        
        self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        batch_size = z.shape[0]
        seq_len = z.shape[-1]
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)

        x_lfp = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        x_lfp = x_lfp.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        if self.deform_patch:
            anchor_shift = self.patch_shift_linear(z).view(batch_size*self.n_vars,self.patch_num,3)
            sampling_location_1d = self.box_coder(anchor_shift) # B*C, self.patch_num,self.patch_len, 1
            add1d = torch.ones(size=(batch_size*self.n_vars,self.patch_num,self.patch_len, 1)).float().to(sampling_location_1d.device)
            sampling_location_2d = torch.cat([sampling_location_1d,add1d],dim=-1)
            z = z.reshape(batch_size*self.n_vars,1,1,seq_len )
            patch = F.grid_sample(z, sampling_location_2d, mode='bilinear', padding_mode='border', align_corners=False).squeeze(1)  # B*C, self.patch_num,self.patch_len
            x_lep = patch.reshape(batch_size,self.n_vars,self.patch_num,self.patch_len).permute(0,1,3,2) #[bs x nvars x patch_len x patch_num]
            PaEN_Loss = cal_PaEn(x_lfp,x_lep,self.r,self.lambda_)
        else:
                if self.padding_patch == 'end':
                    z = self.padding_patch_layer(z)
                z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
                z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
                patch = z
        # model
        z = self.backbone(x_lep)                                                                # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)                                                                    # z: [bs x nvars x target_window] 
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z,PaEN_Loss


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class Encoder(nn.Module):  #i means channel-independent
    def __init__(self,configs, c_in, patch_num, patch_len,
                 n_layers=3, d_model=128, n_heads=16, dropout=0., act="gelu", pe='zeros', learn_pe=True):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len # 

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = HDMixerBlock(configs, q_len, d_model, n_heads, activation=act, n_layers=n_layers)

        
    def forward(self, x) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(x)  # z: [bs x nvars x patch_num x d_model]
        return z


class HDMixerBlock(nn.Module):
    def __init__(self, configs, q_len, d_model, dropout=0., activation='gelu', n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([HDMixerLayer(configs,q_len, d_model, dropout=dropout,
                                                  activation=activation) for i in range(n_layers)])

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return output


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, M, D, N = x.shape
        x = x.reshape(B * M,D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        return x


class HDMixerLayer(nn.Module):
    def __init__(self, configs, q_len, d_model, dropout=0., bias=True, activation="gelu"):
        super().__init__()

        c_in = configs.enc_in
        # Add & Norm
        # [bs x nvars x patch_num x d_model]
        # Position-wise Feed-Forward
        self.mix_time = configs.mix_time
        self.mix_variable = configs.mix_variable
        self.mix_channel  = configs.mix_channel
        self.patch_mixer = nn.Sequential(
                        LayerNorm(d_model),
                        nn.Linear(d_model, d_model*2, bias=bias),
                        get_activation_fn(activation),
                        nn.Dropout(dropout),
                        nn.Linear(d_model*2, d_model, bias=bias),
                        nn.Dropout(dropout),
                        )
        self.time_mixer = nn.Sequential(
                                Transpose(2,3), LayerNorm(q_len),
                                nn.Linear(q_len, q_len*2, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(q_len*2, q_len, bias=bias),
                                nn.Dropout(dropout),
                                Transpose(2,3)
                                )
        # [bs x nvars  x d_model  x patch_num] ->  [bs x nvars x patch_num x d_model]

        # [bs x nvars x patch_num x d_model]
        self.variable_mixer = nn.Sequential(
                                Transpose(1,3), LayerNorm(c_in),
                                nn.Linear(c_in, c_in*2, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(c_in*2, c_in, bias=bias),
                                nn.Dropout(dropout),
                                Transpose(1,3)
                                )

    def forward(self, src:Tensor) -> Tensor:
        # [bs x nvars x patch_num x d_model]
        #print(src.shape)
        if self.mix_channel:
            u = self.patch_mixer(src) + src
        else:
            u = src
        if self.mix_time:
            v = self.time_mixer(u) + src
        else:
            v=u
        if self.mix_variable:
            w = self.variable_mixer(v) + src
        else:
            w = v
        out = w
        return out


class HDMixer(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.device = config.get('device', torch.device('cpu'))
        self._scaler = self.data_feature.get('scaler')

        # load parameters
        self.c_in = self.data_feature.get('num_nodes', 1)
        self.seq_len = config.get('input_window', 12)
        self.context_window = config.get('input_window', 12)
        self.target_window = config.get('output_window', 12)

        self.n_layers = config.get('n_layers', 2)
        self.n_heads = config.get('n_heads', 4)
        self.d_model = config.get('d_model', 16)
        self.d_ff = config.get('d_ff', 32)
        self.dropout = config.get('dropout', 0.8)
        self.head_dropout = config.get('fc_dropout', 0.0)

        self.individual = config.get('individual', 0)

        self.patch_len = config.get('patch_len', 2)
        self.stride = config.get('stride', 1)
        self.padding_patch = config.get('padding_patch', 'end')

        self.revin = config.get('revin', 1)
        self.affine = config.get('affine', 0)
        self.subtract_last = config.get('subtract_last', 0)

        self.decomposition = config.get('decomposition', 0)
        self.act = config.get('act', 'gelu')
        self.pe = config.get('pe', 'zeros')
        self.learn_pe = config.get('learn_pe', True)
        self.head_type = config.get('head_type', 'flatten')

        self.model = HDMixer_backbone(config, c_in=self.c_in, context_window=self.context_window,
                                      target_window=self.target_window, patch_len=self.patch_len, stride=self.stride,
                                      n_layers=self.n_layers, d_model=self.d_model,
                                      n_heads=self.n_heads,
                                      dropout=self.dropout, act=self.act,
                                      pe=self.pe, learn_pe=self.learn_pe,
                                      head_dropout=self.head_dropout,
                                      padding_patch=self.padding_patch,
                                      head_type=self.head_type, individual=self.individual,
                                      revin=self.revin, affine=self.affine,
                                      subtract_last=self.subtract_last)

    def forward(self, x):  # x: [Batch, Input length, Channel]

        x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
        x, PaEN_Loss = self.model(x)
        x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        return x, PaEN_Loss

    def predict(self, batch):
        outs = []
        # spatial graph
        x = batch['X']
        y = self.forward(x)
        return y

    def calculate_loss(self, batch):
        y_true = batch['y']  # ground-truth value
        y_predicted = self.predict(batch)  # prediction results
        # denormalization the value
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.huber_loss(y_predicted, y_true, 1.0)
