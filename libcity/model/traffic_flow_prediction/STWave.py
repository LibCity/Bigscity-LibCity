import math
from logging import getLogger

import pywt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from libcity.model import loss

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class TemEmbedding(nn.Module):
    def __init__(self, D, a_dim):
        super(TemEmbedding, self).__init__()
        self.ff_te = FeedForward([a_dim + 7, D, D])

    def forward(self, TE, T=288):
        '''
        TE: [B,T,2]
        return: [B,T,N,D]
        '''
        # temporal embedding
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7).to(TE.device)  # [B,T,7]
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T).to(TE.device)  # [B,T,288]
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % T, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1)  # [B,T,295]
        TE = TE.unsqueeze(dim=2)  # [B,T,1,295]
        TE = self.ff_te(TE)  # [B,T,1,F]

        return TE  # [B,T,N,F]


class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i + 1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L - 1:
                x = F.relu(x)
        if self.res_ln:
            x += inputs
            x = self.ln(x)
        return x


class Sparse_Spatial_Attention(nn.Module):
    def __init__(self, heads, dims, samples, localadj):
        super(Sparse_Spatial_Attention, self).__init__()
        features = heads * dims
        self.h = heads
        self.d = dims
        self.s = samples
        self.la = localadj

        self.qfc = FeedForward([features, features])
        self.kfc = FeedForward([features, features])
        self.vfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])

        self.ln = nn.LayerNorm(features)
        self.ff = FeedForward([features, features, features], True)
        self.proj = nn.Linear(self.la.shape[1], 1)

    def forward(self, x, spa_eigvalue, spa_eigvec, tem_eigvalue, tem_eigvec):
        '''
        x: [B,T,N,D]
        spa_eigvalue, tem_eigvalue: [D]
        spa_eigvec, tem_eigvec: [N,D]
        return: [B,T,N,D]
        '''
        x_ = x + torch.matmul(spa_eigvec, torch.diag_embed(spa_eigvalue)) + torch.matmul(tem_eigvec,
                                                                                         torch.diag_embed(tem_eigvalue))

        Q = self.qfc(x_)
        K = self.kfc(x_)
        V = self.vfc(x_)

        Q = torch.cat(torch.split(Q, self.d, -1), 0)
        K = torch.cat(torch.split(K, self.d, -1), 0)
        V = torch.cat(torch.split(V, self.d, -1), 0)

        B, T, N, D = K.shape
        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, T, N, N, D)
        K_sample = K_expand[:, :, torch.arange(N).unsqueeze(1), self.la, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        Sampled_Nodes = int(self.s * math.log(N, 2))
        M = self.proj(Q_K_sample).squeeze(-1)
        M_top = M.topk(Sampled_Nodes, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(T)[None, :, None],
                   M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        Q_K /= (self.d ** 0.5)

        attn = torch.softmax(Q_K, dim=-1)

        # copy operation
        cp = attn.argmax(dim=-2, keepdim=True).transpose(-2, -1)

        value = torch.matmul(attn, V).unsqueeze(-3).expand(B, T, N, M_top.shape[-1], V.shape[-1])[
                torch.arange(B)[:, None, None, None],
                torch.arange(T)[None, :, None, None],
                torch.arange(N)[None, None, :, None], cp, :].squeeze(-2)

        value = torch.cat(torch.split(value, value.shape[0] // self.h, 0), -1)
        value = self.ofc(value)

        value = self.ln(value)

        return self.ff(value)


class TemporalAttention(nn.Module):
    def __init__(self, heads, dims):
        super(TemporalAttention, self).__init__()
        features = heads * dims
        self.h = heads
        self.d = dims

        self.qfc = FeedForward([features, features])
        self.kfc = FeedForward([features, features])
        self.vfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])

        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = FeedForward([features, features, features], True)

    def forward(self, x, te, Mask=True):
        '''
        x: [B,T,N,F]
        te: [B,T,N,F]
        return: [B,T,N,F]
        '''
        x += te

        query = self.qfc(x)  # [B,T,N,F]
        key = self.kfc(x)  # [B,T,N,F]
        value = self.vfc(x)  # [B,T,N,F]

        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0, 2, 1, 3)  # [k*B,T,N,d]
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0, 2, 3, 1)  # [k*B,N,d,T]
        value = torch.cat(torch.split(value, self.d, -1), 0).permute(0, 2, 1, 3)  # [k*B,N,T,d]

        attention = torch.matmul(query, key)  # [k*B,N,T,T]
        attention /= (self.d ** 0.5)  # scaled

        if Mask:
            batch_size = x.shape[0]
            num_steps = x.shape[1]
            num_vertexs = x.shape[2]
            mask = torch.ones(num_steps, num_steps).to(x.device)  # [T,T]
            mask = torch.tril(mask)  # [T,T]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)  # [1,1,T,T]
            mask = mask.repeat(self.h * batch_size, num_vertexs, 1, 1)  # [k*B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1) * torch.ones_like(attention).to(x.device)  # [k*B,N,T,T]
            attention = torch.where(mask, attention, zero_vec)

        attention = F.softmax(attention, -1)  # [k*B,N,T,T]

        value = torch.matmul(attention, value)  # [k*B,N,T,d]

        value = torch.cat(torch.split(value, value.shape[0] // self.h, 0), -1).permute(0, 2, 1, 3)  # [B,T,N,F]
        value = self.ofc(value)
        value += x

        value = self.ln(value)

        return self.ff(value)


class TemporalConvNet(nn.Module):
    def __init__(self, features, kernel_size=2, dropout=0.2, levels=1):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(levels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(features, features, (1, kernel_size), dilation=(1, dilation_size),
                                  padding=(0, padding))
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]
        self.tcn = nn.Sequential(*layers)

    def forward(self, xh):
        xh = self.tcn(xh.transpose(1, 3)).transpose(1, 3)
        return xh


class Dual_Enconder(nn.Module):
    def __init__(self, heads, dims, samples, localadj, spawave, temwave):
        super(Dual_Enconder, self).__init__()
        self.temporal_conv = TemporalConvNet(heads * dims)
        self.temporal_att = TemporalAttention(heads, dims)

        self.spatial_att_l = Sparse_Spatial_Attention(heads, dims, samples, localadj)
        self.spatial_att_h = Sparse_Spatial_Attention(heads, dims, samples, localadj)

        spa_eigvalue = torch.from_numpy(spawave[0].astype(np.float32))
        self.spa_eigvalue = nn.Parameter(spa_eigvalue, requires_grad=True)
        self.spa_eigvec = torch.from_numpy(spawave[1].astype(np.float32))

        tem_eigvalue = torch.from_numpy(temwave[0].astype(np.float32))
        self.tem_eigvalue = nn.Parameter(tem_eigvalue, requires_grad=True)
        self.tem_eigvec = torch.from_numpy(temwave[1].astype(np.float32))

    def forward(self, xl, xh, te):
        '''
        xl: [B,T,N,F]
        xh: [B,T,N,F]
        te: [B,T,N,F]
        return: [B,T,N,F]
        '''
        xl = self.temporal_att(xl, te)  # [B,T,N,F]
        xh = self.temporal_conv(xh)  # [B,T,N,F]

        spa_statesl = self.spatial_att_l(xl, self.spa_eigvalue, self.spa_eigvec.to(xl.device), self.tem_eigvalue,
                                         self.tem_eigvec.to(xl.device))  # [B,T,N,F]
        spa_statesh = self.spatial_att_h(xh, self.spa_eigvalue, self.spa_eigvec.to(xl.device), self.tem_eigvalue,
                                         self.tem_eigvec.to(xl.device))  # [B,T,N,F]
        xl = spa_statesl + xl
        xh = spa_statesh + xh

        return xl, xh


class Adaptive_Fusion(nn.Module):
    def __init__(self, heads, dims):
        super(Adaptive_Fusion, self).__init__()
        features = heads * dims
        self.h = heads
        self.d = dims

        self.qlfc = FeedForward([features, features])
        self.khfc = FeedForward([features, features])
        self.vhfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])

        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = FeedForward([features, features, features], True)

    def forward(self, xl, xh, te, Mask=True):
        '''
        xl: [B,T,N,F]
        xh: [B,T,N,F]
        te: [B,T,N,F]
        return: [B,T,N,F]
        '''
        xl += te
        xh += te

        query = self.qlfc(xl)  # [B,T,N,F]
        keyh = torch.relu(self.khfc(xh))  # [B,T,N,F]
        valueh = torch.relu(self.vhfc(xh))  # [B,T,N,F]

        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0, 2, 1, 3)  # [k*B,N,T,d]
        keyh = torch.cat(torch.split(keyh, self.d, -1), 0).permute(0, 2, 3, 1)  # [k*B,N,d,T]
        valueh = torch.cat(torch.split(valueh, self.d, -1), 0).permute(0, 2, 1, 3)  # [k*B,N,T,d]

        attentionh = torch.matmul(query, keyh)  # [k*B,N,T,T]

        if Mask:
            batch_size = xl.shape[0]
            num_steps = xl.shape[1]
            num_vertexs = xl.shape[2]
            mask = torch.ones(num_steps, num_steps).to(xl.device)  # [T,T]
            mask = torch.tril(mask)  # [T,T]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)  # [1,1,T,T]
            mask = mask.repeat(self.h * batch_size, num_vertexs, 1, 1)  # [k*B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1) * torch.ones_like(attentionh).to(xl.device)  # [k*B,N,T,T]
            attentionh = torch.where(mask, attentionh, zero_vec)

        attentionh /= (self.d ** 0.5)  # scaled
        attentionh = F.softmax(attentionh, -1)  # [k*B,N,T,T]

        value = torch.matmul(attentionh, valueh)  # [k*B,N,T,d]

        value = torch.cat(torch.split(value, value.shape[0] // self.h, 0), -1).permute(0, 2, 1, 3)  # [B,T,N,F]
        value = self.ofc(value)
        value = value + xl

        value = self.ln(value)

        return self.ff(value)


class STWave(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))

        # data
        self._scaler = self.data_feature.get('scaler')
        localadj = self.data_feature.get("localadj")
        spawave = self.data_feature.get("spawave")
        temwave = self.data_feature.get("temwave")
        self.output_dim = self.data_feature.get("output_dim")
        self.feature_dim = self.data_feature.get("feature_dim")

        # model
        heads = config.get("heads", 8)
        dims = config.get("dims", 16)
        layers = config.get("layers", 2)
        samples = config.get("samples", 1)
        input_len = config.get("input_window", 12)
        output_len = config.get("output_window", 12)
        self.wave = config.get("wave", "coif1")
        self.level = config.get("level", 1)
        self.vocab_size = 24 * 60 * 60 // config.get("time_intervals", 300)

        features = heads * dims
        I = torch.arange(localadj.shape[0]).unsqueeze(-1)
        localadj = torch.cat([I, torch.from_numpy(localadj)], -1)
        self.input_len = input_len

        self.dual_enc = nn.ModuleList(
            [Dual_Enconder(heads, dims, samples, localadj, spawave, temwave) for i in range(layers)])
        self.adp_f = Adaptive_Fusion(heads, dims)

        self.pre_l = nn.Conv2d(input_len, output_len, (1, 1))
        self.pre_h = nn.Conv2d(input_len, output_len, (1, 1))
        self.pre = nn.Conv2d(input_len, output_len, (1, 1))

        self.start_emb_l = FeedForward([self.output_dim, features, features])
        self.start_emb_h = FeedForward([self.output_dim, features, features])
        self.end_emb = FeedForward([features, features, self.output_dim])
        self.end_emb_l = FeedForward([features, features, self.output_dim])
        self.te_emb = TemEmbedding(features, self.vocab_size)

    def forward(self, XL, XH, TE):
        xl, xh = self.start_emb_l(XL), self.start_emb_h(XH)
        te = self.te_emb(TE, self.vocab_size)

        for enc in self.dual_enc:
            xl, xh = enc(xl, xh, te[:, :self.input_len, :, :])
        hat_y_l = self.pre_l(xl)
        hat_y_h = self.pre_h(xh)
        hat_y = self.adp_f(hat_y_l, hat_y_h,
                           te[:, self.input_len:, :, :])
        hat_y, hat_y_l = self.end_emb(hat_y), self.end_emb_l(hat_y_l)
        return hat_y, hat_y_l

    def disentangle(self, x, w, j):
        x = x.cpu()
        x = x.numpy()

        x = x.transpose(0, 3, 2, 1)  # [S,D,N,T]
        coef = pywt.wavedec(x, w, level=j)
        coefl = [coef[0]]
        for i in range(len(coef) - 1):
            coefl.append(None)
        coefh = [None]
        for i in range(len(coef) - 1):
            coefh.append(coef[i + 1])
        xl = pywt.waverec(coefl, w).transpose(0, 3, 2, 1)
        xh = pywt.waverec(coefh, w).transpose(0, 3, 2, 1)
        xl = torch.from_numpy(xl).to(self.device)
        xh = torch.from_numpy(xh).to(self.device)
        return xl, xh

    def prepare_param(self, batch):
        x = batch['X']  # BTNF
        y = batch['y']
        x, xte = x[:, :, :, :self.output_dim], x[:, :, :, self.output_dim:]
        yte = y[:, :, :, self.output_dim:]
        xl, xh = self.disentangle(x, self.wave, self.level)
        te = torch.cat([xte, yte], dim=1)[:, :, 0, :]
        return xl, xh, te

    def loss_predict(self, batch):
        xl, xh, te = self.prepare_param(batch)
        return self.forward(xl, xh, te)

    def predict(self, batch):
        xl, xh, te = self.prepare_param(batch)
        hat_y, hat_y_l = self.forward(xl, xh, te)
        return hat_y

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted, hat_y_l = self.loss_predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        # calc y l
        YL, _ = self.disentangle(y_true, self.wave, self.level)
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        hat_y_l = self._scaler.inverse_transform(hat_y_l[..., :self.output_dim])
        return (loss.masked_mae_torch(y_predicted, y_true, null_val=0.0) +
                loss.masked_mae_torch(hat_y_l, YL, null_val=0.0))
