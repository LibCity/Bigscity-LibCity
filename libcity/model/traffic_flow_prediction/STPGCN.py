from logging import getLogger

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class STPGConv(nn.Module):
    def __init__(self, config):
        super(STPGConv, self).__init__()

        self.C = config.C
        self.d = config.d
        self.V = config.V
        self.t_size = config.t_size

        self.x_proj = nn.Linear(self.C, self.C * 2).to(config.device)
        self.se_proj = nn.Linear(self.d, self.C * 2, bias=False).to(config.device)
        self.te_proj = nn.Linear(self.d, self.C * 2, bias=False).to(config.device)
        self.ln = nn.LayerNorm(self.C * 2).to(config.device)

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
    def __init__(self, config):
        super(Gaussian_component, self).__init__()
        self.d = config.d
        self.device = config.device
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

    def __init__(self, config):
        super(STPRI, self).__init__()

        self.d = config.d
        self.V = config.V
        self.t_size = config.t_size

        self.gc_lst = []
        for i in range(6):
            self.gc_lst.append(Gaussian_component(config))

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

    def __init__(self, config):
        super(GFS, self).__init__()
        self.fc = nn.Conv2d(in_channels=config.T, out_channels=config.C, kernel_size=(1, config.C)).to(config.device)
        self.glu = GLU(config.C).to(config.device)

    def forward(self, x):
        """
        x: B,T,V,C
        return B,C,V,1
        """

        x = self.fc(x)  # B,T,V,C -> B,C,V,1
        x = self.glu(x)  # B,C,V,1
        return x


class OutputLayer(nn.Module):
    def __init__(self, config, **kwargs):
        super(OutputLayer, self).__init__(**kwargs)
        self.V = config.V
        self.D = config.num_features
        self.P = config.num_prediction
        self.C = config.C
        self.fc = nn.Conv2d(in_channels=self.C * (config.L + 1), out_channels=self.P * self.D, kernel_size=(1, 1))

    def forward(self, x):
        # x:B,C',V,1 -> B,PD,V,1 -> B,P,V,D
        x = self.fc(x)
        if self.D > 1:
            x = x.reshape((0, self.P, self.D, self.V))
            x = x.transpose((0, 1, 3, 2))
        return x


class InputLayer(nn.Module):
    def __init__(self, config, **kwargs):
        super(InputLayer, self).__init__(**kwargs)
        self.fc = nn.Linear(in_features=config.D, out_features=config.C)

    def forward(self, x):
        x = self.fc(x)
        return x
        # x:B,T,V,D -> B,T,V,C


class STPGCNs(nn.Module):
    """Spatial-Temporal Position-aware Graph Convolution"""

    def __init__(self, config, **kwargs):
        super(STPGCNs, self).__init__(**kwargs)

        self.config = config
        self.L = config.L
        self.d = config.d
        self.C = config.C
        self.V = config.V
        self.T = config.T
        self.t_size = config.t_size

        self.input_layer = InputLayer(self.config)
        self.fs = GFS(self.config)

        self.ri_lst = []
        self.gc_lst = []
        self.fs_lst = []
        for i in range(self.L):
            self.ri_lst.append(STPRI(self.config))

            self.gc_lst.append(STPGConv(self.config))

            self.fs_lst.append(GFS(self.config))

        self.glu = GLU(self.C * 4)
        self.output_layer = OutputLayer(self.config)

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

    def __init__(self, config, **kwargs):
        super(SAPE, self).__init__(**kwargs)

        self.sape = nn.Embedding(config.V, config.d)

    def forward(self):
        # return self.sape.data()
        return self.sape.weight


class TAPE(nn.Module):
    """Temporal Absolute-Position Embedding"""

    def __init__(self, config, **kwargs):
        super(TAPE, self).__init__(**kwargs)

        self.dow_emb = nn.Embedding(config.week_len, config.d)
        self.tod_emb = nn.Embedding(config.day_len, config.d)
def forward(self, pos_w, pos_d):
        # B,T,i -> B,T,1,C
        dow = self.dow_emb(pos_w).unsqueeze(2)
        tod = self.tod_emb(pos_d).unsqueeze(2)
        return dow + tod



class SRPE(nn.Module):
    """Spatial Relative-Position Embedding"""

    def __init__(self, config, **kwargs):
        super(SRPE, self).__init__(**kwargs)

        self.SDist = torch.from_numpy(config.spatial_distance).to(config.device).long().squeeze(-1)
        self.srpe = nn.Embedding(config.alpha + 1, config.d)

    def forward(self):
        return self.srpe(self.SDist)


class TRPE(nn.Module):
    """Temporal Relative-Position Embedding"""

    def __init__(self, config, **kwargs):
        super(TRPE, self).__init__(**kwargs)

        self.TDist = torch.from_numpy(np.expand_dims(range(config.t_size), -1)).to(config.device).long()
        self.trpe = nn.Embedding(config.t_size, config.d)

    def forward(self):
        # return self.trpe.data()[self.TDist]
        return self.trpe(self.TDist)


class GeneratePad(nn.Module):
    def __init__(self, config, **kwargs):
        super(GeneratePad, self).__init__(**kwargs)
        self.device = config.device
        self.C = config.C
        self.V = config.V
        self.d = config.d
        self.pad_size = config.beta

    def forward(self, x):
        B = x.shape[0]
        return torch.zeros((B, self.pad_size, self.V, self.C), device=self.device), torch.zeros(
            (B, self.pad_size, 1, self.d), device=self.device)


class STPGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):

        class ModelConfig:

            def __init__(self, config, data_feature) -> None:
                # device
                self.device = config.get('device', torch.device('cpu'))

                # from data feature
                self.range_mask = data_feature.get("range_mask")
                self.spatial_distance = data_feature.get("spatial_distance")
                self.V = data_feature.get("num_nodes")

                # model config param
                self.num_features = config.get("input_dim", 1)
                self.D = self.num_features
                self.T = config.get("output_window", 12)
                self.num_prediction = self.T
                self.C = config.get("C", 64)
                self.L = config.get("L", 3)
                self.d = config.get("d", 8)
                self.alpha = config.get("alpha", 4)
                self.beta = config.get("beta", 2)
                self.t_size = self.beta + 1
                self.week_len = 7
                self.day_len = config.get("points_per_hour") * 24

        super().__init__(config, data_feature)

        self._logger = getLogger()

        # data
        self._scaler = self.data_feature.get('scaler')

        # model config
        self.output_dim = config.get('output_dim', 1)

        config = ModelConfig(config, self.data_feature)
        self.config = config
        self.T = config.T
        self.V = config.V
        self.C = config.C
        self.L = config.L
        self.range_mask = torch.Tensor(config.range_mask).to(config.device)

        self.PAD = GeneratePad(self.config)
        self.SAPE = SAPE(self.config).to(config.device)
        self.TAPE = TAPE(self.config).to(config.device)
        self.SRPE = SRPE(self.config).to(config.device)
        self.TRPE = TRPE(self.config).to(config.device)
        self.net = STPGCNs(self.config).to(config.device)

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
