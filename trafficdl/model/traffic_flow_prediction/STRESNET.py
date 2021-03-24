from collections import OrderedDict

import torch
import torch.nn as nn

from trafficdl.model import loss
from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class ResUnit(nn.Module):
    def __init__(self, nb_filter, use_bn=True):
        super().__init__()
        if use_bn:
            self.seq = nn.Sequential(OrderedDict([
                ('bn1', nn.BatchNorm2d(nb_filter)),
                ('relu1', nn.ReLU()),
                ('conv1', conv3x3(nb_filter, nb_filter)),
                ('bn2', nn.BatchNorm2d(nb_filter)),
                ('relu2', nn.ReLU()),
                ('conv2', conv3x3(nb_filter, nb_filter)),
            ]))
        else:
            self.seq = nn.Sequential(OrderedDict([
                ('relu1', nn.ReLU()),
                ('conv1', conv3x3(nb_filter, nb_filter)),
                ('relu2', nn.ReLU()),
                ('conv2', conv3x3(nb_filter, nb_filter)),
            ]))

    def forward(self, x):
        residual = x

        output = self.seq(x)
        output += residual  # short cut

        return output


class ResUnits(nn.Module):
    def __init__(self, res_unit, nb_filter, repetitions=1):
        super().__init__()
        self.res_units = nn.Sequential(
            *(res_unit(nb_filter) for _ in range(repetitions)),
            nn.ReLU()
        )

    def forward(self, x):
        output = self.res_units(x)

        return output


# Matrix-based fusion
class FusionLayer(nn.Module):
    def __init__(self, n, h, w):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, n, h, w),
                                    requires_grad=True)  # define the trainable parameter

    def forward(self, x):
        # assuming x is of size b-1-h-w
        x = x * self.weights  # element-wise multiplication

        return x


class STRESNET(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        # c_conf=(3, 2, 32, 32), p_conf=(3, 2, 32, 32),
        # t_conf=(3, 2, 32, 32), external_dim=8, residual_unit_num=3):
        """
            C - Temporal Closeness
            P - Period
            T - Trend
            conf = (len_seq, flow_type_num, map_height, map_width)
            external_dim
        """

        super().__init__(config, data_feature)

        self.device = config.get('device', torch.device('cpu'))
        self.residual_unit_num = config.get('residual_unit_num', 4)
        self.load_external = config.get('load_external', False)
        if self.load_external:
            self.external_dim = config.get('external_dim', 8)

        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.len_period = self.data_feature.get('len_period', 0)
        self.len_trend = self.data_feature.get('len_trend', 0)
        self.len_closeness = self.data_feature.get('len_closeness', 0)
        if self.len_period == 0 and self.len_trend == 0 and self.len_closeness == 0:
            raise ValueError('Num of days/weeks/hours are all zero! Set at least one of them not zero!')

        self.len_row = self.data_feature.get('len_row', 32)
        self.len_column = self.data_feature.get('len_column', 32)
        self.flow_type_num = self.output_dim

        # adj_mx = self.data_feature.get('adj_mx')

        self._scaler = self.data_feature.get('scaler')
        #
        # "scaler": self.scaler, "adj_mx": self.adj_mx,
        # "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
        # "output_dim": self.output_dim, "len_row": self.len_row, "len_column": self.len_column,
        # "len_closeness": self.len_closeness, "len_period": lp, "len_trend": lt}

        # c_conf = (3, 2, 32, 32)
        # p_conf = (3, 2, 32, 32)
        # t_conf = (3, 2, 32, 32)
        #
        # self.c_conf = c_conf
        # self.p_conf = p_conf
        # self.t_conf = t_conf
        #
        # self.flow_type_num, self.map_height, self.map_width = t_conf[1], t_conf[2], t_conf[3]

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Branch c
        if self.len_closeness != 0:
            self.c_way = self.make_one_way(in_channels=self.len_closeness * self.flow_type_num)
        # Branch p
        if self.len_period != 0:
            self.p_way = self.make_one_way(in_channels=self.len_period * self.flow_type_num)
        # Branch t
        if self.len_trend != 0:
            self.t_way = self.make_one_way(in_channels=self.len_trend * self.flow_type_num)

        # Operations of external component
        if self.load_external:
            self.external_ops = nn.Sequential(OrderedDict([
                ('embed', nn.Linear(self.external_dim, 10, bias=True)),
                ('relu1', nn.ReLU()),
                ('fc', nn.Linear(10, self.flow_type_num * self.len_row * self.len_column, bias=True)),
                ('relu2', nn.ReLU()),
            ]))

    def make_one_way(self, in_channels):
        return nn.Sequential(OrderedDict([
            ('conv1', conv3x3(in_channels=in_channels, out_channels=64)),
            ('ResUnits', ResUnits(ResUnit, nb_filter=64, repetitions=self.residual_unit_num)),
            ('conv2', conv3x3(in_channels=64, out_channels=2)),
            ('FusionLayer', FusionLayer(n=self.flow_type_num, h=self.len_row, w=self.len_column))
        ]))

    def forward(self, batch):
        x = batch['X']

        def reshape_input(input):
            """
            Args:
                input: df[batch_num, seq_num, row_num, col_num, flow_num(in/out)]

            Returns:
                shaped_input: df[batch_num, flow_num*seq_num, row_num, col_num]
            """
            input_tmp = input.split(1, dim=1)
            input_tmp = torch.cat(input_tmp, dim=4)
            input_tmp = torch.squeeze(input_tmp, dim=1)
            shaped_input = input_tmp.permute(0, 3, 1, 2)
            return shaped_input

        input_c = reshape_input(x[:, 0:self.len_closeness, :, :, :])
        input_p = reshape_input(x[:, self.len_closeness:self.len_closeness + self.len_period, :, :, :])
        input_t = reshape_input(x[:, self.len_closeness + self.len_period:, :, :, :])

        output = 0
        # Three-way Convolution
        # closeness
        if self.len_closeness != 0:
            out_c = self.c_way(input_c)
            output += out_c
        # period
        if self.len_period != 0:
            out_p = self.p_way(input_p)
            output += out_p
        # trend
        if self.len_trend != 0:
            out_t = self.t_way(input_t)
            output += out_t

        # fusing with external component
        if self.load_external:
            # external input
            # TODO: need reshape
            raise NotImplementedError
            input_ext = batch['X_ext']
            external_output = self.external_ops(input_ext)
            external_output = self.relu(external_output)
            external_output = external_output.view(-1, self.flow_type_num, self.len_row, self.len_column)
            # output = torch.add(main_output, external_output)
            output += external_output
        # else:
        #     print('without external')

        # output: df[batch_num, flow_num, row_num, col_num]
        output = self.tanh(output)
        output_shaped = output.permute(0, 2, 3, 1)
        # output_shaped: df[batch_num, 1, row_num, col_num, flow_num]
        output_shaped = torch.unsqueeze(output_shaped, 1)
        # output_shaped = output.view(-1, 1, self.len_row, self.len_column, self.output_dim)

        return output_shaped

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true_inverse = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted_inverse = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        c_loss = loss.masked_mse_torch(y_predicted_inverse, y_true_inverse)
        return c_loss

    def predict(self, batch):
        return self.forward(batch)
