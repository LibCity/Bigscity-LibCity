from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=2, window_size=1, qkv_bias=False, qk_scale=None, dropout=0., causal=True,
                 device=None):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        self.mask = torch.tril(torch.ones(window_size, window_size)).to(device)  # mask for causality

    def forward(self, x):
        B_prev, T_prev, C_prev = x.shape
        if self.window_size > 0:
            x = x.reshape(-1, self.window_size, C_prev)  # create local windows
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # merge key padding and attention masks
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [b, heads, T, T]

        if self.causal:
            attn = attn.masked_fill_(self.mask == 0, float("-inf"))

        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self.window_size > 0:  # reshape to the original size
            x = x.reshape(B_prev, T_prev, C_prev)
        return x


class CT_MSA(nn.Module):
    # Causal Temporal MSA
    def __init__(self,
                 dim,  # hidden dim
                 depth,  # the number of MSA in CT-MSA
                 heads,  # the number of heads
                 window_size,  # the size of local window
                 mlp_dim,  # mlp dimension
                 num_time,  # the number of time slot
                 dropout=0.,  # dropout rate
                 device=None):  # device, e.g., cuda
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_time, dim))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                TemporalAttention(dim=dim,
                                  heads=heads,
                                  window_size=window_size,
                                  dropout=dropout,
                                  device=device),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        # x: [b, c, n, t]
        b, c, n, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * n, t, c)  # [b*n, t, c]
        x = x + self.pos_embedding  # [b*n, t, c]
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.reshape(b, n, t, c).permute(0, 3, 1, 2)
        return x


class SimST(AbstractTrafficStateModel):

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))
        device = self.device

        # data
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes')

        # model
        node_num = self.num_nodes
        node_dim = config.get("node_dim", 20)
        in_dim = config.get("in_dim", 5)
        out_dim = config.get("output_window", 12)
        init_dim = config.get("init_dim", 64)
        end_dim = config.get("end_dim", 512)
        layer = config.get("layer", 2)
        dropout = config.get("dropout", 0.1)
        self.output_dim = config.get("output_dim", 1)

        self.node_dim = node_dim

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=init_dim,
                                    kernel_size=(1, 1))

        self.encoder = CT_MSA(dim=init_dim, depth=layer, heads=2, window_size=out_dim, mlp_dim=init_dim * 2,
                              num_time=out_dim, dropout=dropout, device=device)

        cnt = 1
        if node_dim != 0:
            cnt = 2
            self.node_embed = nn.Parameter(torch.randn(node_num, node_dim), requires_grad=True)
            self.node_linear = nn.Linear(node_dim, init_dim)

        self.end_conv_1 = nn.Conv2d(in_channels=init_dim * cnt,
                                    out_channels=end_dim,
                                    kernel_size=(1, 1), bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_dim,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1), bias=True)

    def forward(self, input, idx=None):  # input (bs, hid, node, time)

        x = self.start_conv(input)

        out = self.encoder(x)  # out (bs, hid, node, time)
        x = out[..., -1:]

        if self.node_dim != 0:
            x_ = self.node_linear(self.node_embed)

            if x.shape[0] > 64:
                x_idx = x_[idx].unsqueeze(dim=-1).unsqueeze(dim=-1)
            else:
                x_idx = x_.transpose(0, 1).unsqueeze(dim=0).unsqueeze(dim=-1).expand(x.shape[0], -1, -1, -1)

            x = torch.cat((x, x_idx), dim=1)

        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x

    def predict(self, batch):
        x = batch['X'].transpose(1, 3)
        node_idx = batch['node_idx']
        return self.forward(x, node_idx)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, null_val=0.0)
