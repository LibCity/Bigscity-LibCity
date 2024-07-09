from logging import getLogger

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class STPGConv(nn.Module):
    def __init__(self, C, d, V, t_size, device):
        super(STPGConv, self).__init__()

        self.C = C
        self.d = d
        self.V = V
        self.t_size = t_size

        self.x_proj = nn.Linear(self.C, self.C * 2).to(device)
        self.se_proj = nn.Linear(self.d, self.C * 2, bias=False).to(device)
        self.te_proj = nn.Linear(self.d, self.C * 2, bias=False).to(device)
        self.ln = nn.LayerNorm(self.C * 2).to(device)

    def forward(self, x, S, sape, tape):
        # x:B,t,V,C
        # S:B,V,tV
        # sape: V,d
        # tape: B,1,1,d

        # aggregation
        # B,t,V,C -> B,tV,C
        x = x.reshape((-1, self.t_size * self.V, self.C))  #
        # B,(V,tV x tV,C) -> B,V,C
        x = torch.bmm(S, x)
        # B,V,C -> B,V,2C
        x = self.x_proj(x)

        # STPGAU
        # V,d x d,2C -> V,2C
        SE = self.se_proj(sape)
        # B,1,1,d -> B,1,1,d -> B,1,d
        TE = tape.reshape((-1, 1, self.d))
        # B,1,d x d,2C -> B,1,2C
        TE = self.te_proj(TE)
        x += SE  # x +=  SE
        x += TE
        x = self.ln(x)
        lhs, rhs = torch.chunk(x, chunks=2, dim=-1)
        return lhs * F.sigmoid(rhs)


class Gaussian_component(nn.Module):
    def __init__(self, d, device, ):
        super(Gaussian_component, self).__init__()
        self.d = d
        self.device = device
        self.mu = nn.Parameter(torch.randn(1, self.d)).to(self.device)
        self.inv_sigma = nn.Parameter(torch.randn(1, self.d)).to(self.device)

    def forward(self, emb):
        """
        emb:
        return
        """
        e = -0.5 * torch.pow(emb - self.mu.expand_as(emb), 2)
        e = e * torch.pow(self.inv_sigma, 2).expand_as(emb)
        e = torch.sum(e, dim=-1, keepdim=True)
        return e


class STPRI(nn.Module):
    """Spatial-Temporal Position-aware Relation Inference"""

    def __init__(self, d, V, t_size, device):
        super(STPRI, self).__init__()

        self.d = d
        self.V = V
        self.t_size = t_size
        self.device = device

        self.gc_lst = []
        for i in range(6):
            self.gc_lst.append(Gaussian_component(self.d, self.device))

    def forward(self, sape, tape_i, tape_j, srpe, trpe):
        """
        sape:V,d
        tape_i: B, 1, 1, d
        tape_j: B, t, 1, d
        srpe:V,V,d
        trpe:t,1,d
        """

        B = tape_j.shape[0]
        # V,d -> V,1
        sapei = self.gc_lst[0](sape)

        # V,d -> V,1 -> 1,V
        sapej = self.gc_lst[1](sape)
        sapej = sapej.transpose(1, 0)

        # V,1 + 1,V -> V,V
        gaussian = sapei.expand(self.V, self.V) + sapej.expand(self.V, self.V)

        # B,1,1,d -> B,1,1,1
        tapei = self.gc_lst[2](tape_i)

        # B,1,1,1 + V,V -> B,1,V,V
        gaussian = gaussian + tapei

        # B,t,1,d -> B,t,1,1
        tapej = self.gc_lst[3](tape_j)

        #  B,1,V,V + B,t,1,1  -> B,t,V,V
        gaussian = gaussian.expand(B, self.t_size, self.V, self.V) + tapej.expand(B, self.t_size, self.V, self.V)

        # V,V,d -> V,V,1 -> V,V
        srpe = self.gc_lst[4](srpe).squeeze()

        # B,t,V,V + V,V -> B,t,V,V
        gaussian = gaussian + srpe

        # t,1,d -> t,1,1
        trpe = self.gc_lst[5](trpe)

        # B,t,V,V + t,1,1 -> B,t,V,V
        gaussian += trpe.unsqueeze(0).expand_as(gaussian)

        # B,t,V,V -> B,tV,V -> B,V,tV
        gaussian = gaussian.reshape(-1, self.t_size * self.V, self.V)
        gaussian = gaussian.transpose(2, 1)

        return torch.exp(gaussian)


class GLU(nn.Module):
    def __init__(self, dim, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.linear = nn.Conv2d(in_channels=dim, out_channels=dim * 2, kernel_size=(1, 1))

    def forward(self, x):
        # B,C,V,T
        x = self.linear(x)
        lhs, rhs = torch.chunk(x, chunks=2, dim=1)
        return lhs * F.sigmoid(rhs)


class GFS(nn.Module):
    """gated feature selection module"""

    def __init__(self, T, C, device):
        super(GFS, self).__init__()
        self.fc = nn.Conv2d(in_channels=T, out_channels=C, kernel_size=(1, C)).to(device)
        self.glu = GLU(C).to(device)

    def forward(self, x):
        """
        x: B,T,V,C
        return B,C,V,1
        """

        x = self.fc(x)  # B,T,V,C -> B,C,V,1
        x = self.glu(x)  # B,C,V,1
        return x


class OutputLayer(nn.Module):
    def __init__(self, V, num_features, num_prediction, C, L, **kwargs):
        super(OutputLayer, self).__init__(**kwargs)
        self.V = V
        self.D = num_features
        self.P = num_prediction
        self.C = C
        self.fc = nn.Conv2d(in_channels=self.C * (L + 1), out_channels=self.P * self.D, kernel_size=(1, 1))

    def forward(self, x):
        # x:B,C',V,1 -> B,PD,V,1 -> B,P,V,D
        x = self.fc(x)
        if self.D > 1:
            x = x.reshape((-1, self.P, self.D, self.V))
            x = x.permute((0, 1, 3, 2))
        return x


class InputLayer(nn.Module):
    def __init__(self, D, C, **kwargs):
        super(InputLayer, self).__init__(**kwargs)
        self.fc = nn.Linear(in_features=D, out_features=C)

    def forward(self, x):
        x = self.fc(x)
        return x
        # x:B,T,V,D -> B,T,V,C


class STPGCNs(nn.Module):
    """Spatial-Temporal Position-aware Graph Convolution"""

    def __init__(self, L, d, C, V, T, t_size, D, device, num_features, num_prediction, **kwargs):
        super(STPGCNs, self).__init__(**kwargs)

        self.D = D
        self.L = L
        self.d = d
        self.C = C
        self.V = V
        self.T = T
        self.t_size = t_size
        self.device = device

        self.input_layer = InputLayer(self.D, self.C)
        self.fs = GFS(self.T, self.C, self.device)

        self.ri_lst = []
        self.gc_lst = []
        self.fs_lst = []
        for i in range(self.L):
            self.ri_lst.append(STPRI(self.d, self.V, self.t_size, self.device))

            self.gc_lst.append(STPGConv(self.C, d, V, t_size, device))

            self.fs_lst.append(GFS(self.T, self.C, self.device))

        self.glu = GLU(self.C * 4)
        self.output_layer = OutputLayer(self.V, num_features, num_prediction, self.C, self.L)

    def forward(self, x, sape, tape, srpe, trpe, zeros_x, zeros_tape, range_mask):
        """
        x:B,T,V,D
        sape:V,d
        tape:B,T,1,d
        srpe:V,V,d
        trpe:t,1,d
        zeros_x:B,beta,V,D
        zeros_tape:B,beta,1,d
        range_mask:V,tV
        """

        # x:B,T,V,D -> B,T,V,C
        x = self.input_layer(x)

        # padding: B,T+beta,1,d
        tape_pad = torch.cat((zeros_tape, tape), dim=1)

        skip = [self.fs(x)]
        for i in range(self.L):
            # padding: B,T+beta,V,C
            x = torch.cat((zeros_x, x), dim=1)

            xs = []
            for t in range(self.T):
                # B,t,V,C
                xj = x[:, t:t + self.t_size, :, :]

                # B,1,1,d
                tape_i = tape[:, t:t + 1, :, :]

                # B,t,1,d
                tape_j = tape_pad[:, t:t + self.t_size, :, :]

                # Inferring spatial-temporal relations
                S = self.ri_lst[i](sape, tape_i, tape_j, srpe, trpe)
                S = S * range_mask

                # STPGConv
                xs.append(self.gc_lst[i](xj, S, sape, tape_i))

            x = torch.stack(xs, dim=1)

            # B,T,V,C->B,C',V,1
            skip.append(self.fs_lst[i](x))

        # B,(L+1)*C,V,1
        x = torch.cat(skip, dim=1)

        # B,(L+1)*C,V,1 ->  B,(L+1)*C,V,1
        x = self.glu(x)

        # B,C,V,1 -> B,PF,V,1 -> B,P,V,D
        x = self.output_layer(x)
        return x


class SAPE(nn.Module):
    """Spatial Absolute-Position Embedding"""

    def __init__(self, V, d, **kwargs):
        super(SAPE, self).__init__(**kwargs)

        self.sape = nn.Embedding(V, d)

    def forward(self):
        # return self.sape.data()
        return self.sape.weight


class TAPE(nn.Module):
    """Temporal Absolute-Position Embedding"""

    def __init__(self, week_len, day_len, d, **kwargs):
        super(TAPE, self).__init__(**kwargs)

        self.dow_emb = nn.Embedding(week_len, d)
        self.tod_emb = nn.Embedding(day_len, d)

    def forward(self, pos_w, pos_d):
        # B,T,i -> B,T,1,C
        dow = self.dow_emb(pos_w).unsqueeze(2)
        tod = self.tod_emb(pos_d).unsqueeze(2)
        return dow + tod


class SRPE(nn.Module):
    """Spatial Relative-Position Embedding"""

    def __init__(self, spatial_distance, device, alpha, d, **kwargs):
        super(SRPE, self).__init__(**kwargs)

        self.SDist = torch.from_numpy(spatial_distance).to(device).long().squeeze(-1)
        self.srpe = nn.Embedding(alpha + 1, d)

    def forward(self):
        return self.srpe(self.SDist)


class TRPE(nn.Module):
    """Temporal Relative-Position Embedding"""

    def __init__(self, t_size, device, d, **kwargs):
        super(TRPE, self).__init__(**kwargs)

        self.TDist = torch.from_numpy(np.expand_dims(range(t_size), -1)).to(device).long()
        self.trpe = nn.Embedding(t_size, d)

    def forward(self):
        # return self.trpe.data()[self.TDist]
        return self.trpe(self.TDist)


class GeneratePad(nn.Module):
    def __init__(self, device, C, V, d, beta, **kwargs):
        super(GeneratePad, self).__init__(**kwargs)
        self.device = device
        self.C = C
        self.V = V
        self.d = d
        self.pad_size = beta

    def forward(self, x):
        B = x.shape[0]
        return torch.zeros((B, self.pad_size, self.V, self.C), device=self.device), torch.zeros(
            (B, self.pad_size, 1, self.d), device=self.device)


class STPGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))

        # data
        self._scaler = self.data_feature.get('scaler')
        self.output_dim = self.data_feature.get('output_dim')
        self.range_mask = self.data_feature.get("range_mask")
        self.spatial_distance = self.data_feature.get("spatial_distance")
        self.V = self.data_feature.get("num_nodes")
        self.num_features = self.data_feature.get("output_dim")

        # model config
        self.D = self.data_feature.get("feature_dim")
        self.T = config.get("input_window", 12)
        self.num_prediction = config.get("output_window", 12)
        self.C = config.get("C", 64)
        self.L = config.get("L", 3)
        self.d = config.get("d", 8)
        self.alpha = config.get("alpha", 4)
        self.beta = config.get("beta", 2)
        self.t_size = self.beta + 1
        self.week_len = 7
        self.day_len = self.data_feature.get("points_per_hour") * 24
        self.range_mask = torch.Tensor(self.range_mask).to(self.device)

        self.PAD = GeneratePad(self.device, self.C, self.V, self.d, self.beta)
        self.SAPE = SAPE(self.V, self.d).to(self.device)
        self.TAPE = TAPE(self.week_len, self.day_len, self.d).to(self.device)
        self.SRPE = SRPE(self.spatial_distance, self.device, self.alpha, self.d).to(self.device)
        self.TRPE = TRPE(self.t_size, self.device, self.d).to(self.device)
        self.net = STPGCNs(self.L, self.d, self.C, self.V, self.T, self.t_size, self.D, self.device, self.num_features,
                           self.num_prediction).to(self.device)

    def forward(self, x, pos_w, pos_d):
        # x:B,T,V,D
        # pos_w:B,t,1,1
        # pos_d:B,t,1,1
        sape = self.SAPE()
        tape = self.TAPE(pos_w, pos_d)
        srpe = self.SRPE()
        trpe = self.TRPE()
        zeros_x, zeros_tape = self.PAD(x)
        x = self.net(x, sape, tape, srpe, trpe, zeros_x, zeros_tape, self.range_mask)
        return x

    def predict(self, batch):
        return self.forward(batch['X'], batch['pos_w'], batch['pos_d'])

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_loss(y_predicted, y_true)
