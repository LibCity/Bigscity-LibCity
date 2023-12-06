import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss

class SNorm(nn.Module):
    def __init__(self,  channels):
        super(SNorm, self).__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        x_norm = (x - x.mean(2, keepdims=True)) / (x.var(2, keepdims=True, unbiased=True) + 0.00001) ** 0.5

        out = x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return out

class TNorm(nn.Module):
    def __init__(self,  num_nodes, channels, track_running_stats=True, momentum=0.1):
        super(TNorm, self).__init__()
        self.track_running_stats = track_running_stats
        self.beta = nn.Parameter(torch.zeros(1, channels, num_nodes, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, num_nodes, 1))
        self.register_buffer('running_mean', torch.zeros(1, channels, num_nodes, 1))
        self.register_buffer('running_var', torch.ones(1, channels, num_nodes, 1))
        self.momentum = momentum

    def forward(self, x):
        if self.track_running_stats:
            mean = x.mean((0, 3), keepdims=True)
            var = x.var((0, 3), keepdims=True, unbiased=False)
            if self.training:
                n = x.shape[3] * x.shape[0]
                with torch.no_grad():
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                    self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = x.mean((3), keepdims=True)
            var = x.var((3), keepdims=True, unbiased=True)
        x_norm = (x - mean) / (var + 0.00001) ** 0.5
        out = x_norm * self.gamma + self.beta
        return out

class STNorm(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.in_dim = self.data_feature.get('feature_dim', 1)
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        
        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()
        
        self.out_dim = config.get('n_pred', 1)
        self.blocks = config.get('blocks', 1)
        self.layers = config.get('layers', 4)
        self.kernel_size = config.get('kernel_size', 2)
        self.snorm_bool = config.get('snorm_bool', True)
        self.tnorm_bool = config.get('tnorm_bool', True)
        self.channels = config.get('hidden_channels', 16)
        
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        if self.snorm_bool:
            self.sn = nn.ModuleList()
        if self.tnorm_bool:
            self.tn = nn.ModuleList()
        num = int(self.tnorm_bool) + int(self.snorm_bool) + 1

        self.mlps = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.cross_product = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=self.in_dim,
                                    out_channels=self.channels,
                                    kernel_size=(1,1))

        receptive_field = 1
        self.dropout = nn.Dropout(0.2)

        self.dilation = []

        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilated convolutions
                self.dilation.append(new_dilation)
                if self.tnorm_bool:
                    self.tn.append(TNorm(self.num_nodes, self.channels))
                if self.snorm_bool:
                    self.sn.append(SNorm(self.channels))
                self.filter_convs.append(nn.Conv2d(in_channels=num * self.channels,
                                                   out_channels=self.channels,
                                                   kernel_size=(1,self.kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=num * self.channels,
                                                 out_channels=self.channels,
                                                 kernel_size=(1, self.kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=self.channels,
                                                     out_channels=self.channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=self.channels,
                                                 out_channels=self.channels,
                                                 kernel_size=(1, 1)))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(in_channels=self.channels,
                                  out_channels=self.channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=self.channels,
                                    out_channels=self.out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            if m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)
            else:
                nn.init.uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def predict(self, batch):
        input = batch['X']
        input = input.permute(0, 3, 2, 1)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            x_list = []
            x_list.append(x)
            b, c, n, t = x.shape
            if self.tnorm_bool:
                x_tnorm = self.tn[i](x)
                x_list.append(x_tnorm)
            if self.snorm_bool:
                x_snorm = self.sn[i](x)
                x_list.append(x_snorm)
            # dilated convolution
            x = torch.cat(x_list, dim=1)
            filter = self.filter_convs[i](x)
            b, c, n, t = filter.shape
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

        x = F.relu(skip)
        rep = F.relu(self.end_conv_1(x))
        out = self.end_conv_2(rep)
        return out
                
    def calculate_loss(self, batch):
        y_true = batch['y']  # ground-truth value
        y_predicted = self.predict(batch)  # prediction results
        y_true = y_true[..., :self.out_dim]
        y_predicted = y_predicted[..., :self.out_dim]
        res = loss.masked_mae_torch(y_predicted, y_true, 0)
        return res