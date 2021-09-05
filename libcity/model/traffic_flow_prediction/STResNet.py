import torch
import torch.nn as nn

from collections import OrderedDict
from logging import getLogger

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class BnReluConv(nn.Module):
    def __init__(self, nb_filter, bn=False):
        super(BnReluConv, self).__init__()
        self.has_bn = bn
        self.bn1 = nn.BatchNorm2d(nb_filter)
        self.relu = torch.relu
        self.conv1 = conv3x3(nb_filter, nb_filter)

    def forward(self, x):
        if self.has_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        return x


class ResidualUnit(nn.Module):
    def __init__(self, nb_filter, bn=False):
        super(ResidualUnit, self).__init__()
        self.bn_relu_conv1 = BnReluConv(nb_filter, bn)
        self.bn_relu_conv2 = BnReluConv(nb_filter, bn)

    def forward(self, x):
        residual = x
        out = self.bn_relu_conv1(x)
        out = self.bn_relu_conv2(out)
        out += residual  # short cut
        return out


class ResUnits(nn.Module):
    def __init__(self, residual_unit, nb_filter, repetations=1, bn=False):
        super(ResUnits, self).__init__()
        self.stacked_resunits = self.make_stack_resunits(residual_unit, nb_filter, repetations, bn)

    def make_stack_resunits(self, residual_unit, nb_filter, repetations, bn):
        layers = []
        for i in range(repetations):
            layers.append(residual_unit(nb_filter, bn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stacked_resunits(x)
        return x


class TrainableEltwiseLayer(nn.Module):
    # Matrix-based fusion
    def __init__(self, n, h, w, device):
        super(TrainableEltwiseLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(1, n, h, w).to(device),
                                    requires_grad=True)  # define the trainable parameter

    def forward(self, x):
        # assuming x is of size b-1-h-w
        # print('x', x.shape)
        # print('weight', self.weights.shape)
        x = x * self.weights  # element-wise multiplication
        return x


class STResNet(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._scaler = self.data_feature.get('scaler')
        self.adj_mx = self.data_feature.get('adj_mx')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 2)  # 这种情况下不包括外部数据的维度
        self.ext_dim = self.data_feature.get('ext_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 2)  # feature_dim = output_dim
        self.len_row = self.data_feature.get('len_row', 32)
        self.len_column = self.data_feature.get('len_column', 32)
        self.len_closeness = self.data_feature.get('len_closeness', 4)
        self.len_period = self.data_feature.get('len_period', 2)
        self.len_trend = self.data_feature.get('len_trend', 0)
        self._logger = getLogger()

        self.nb_residual_unit = config.get('nb_residual_unit', 12)
        self.bn = config.get('batch_norm', False)
        self.device = config.get('device', torch.device('cpu'))
        self.relu = torch.relu
        self.tanh = torch.tanh

        if self.len_closeness > 0:
            self.c_way = self.make_one_way(in_channels=self.len_closeness * self.feature_dim)

        if self.len_period > 0:
            self.p_way = self.make_one_way(in_channels=self.len_period * self.feature_dim)

        if self.len_trend > 0:
            self.t_way = self.make_one_way(in_channels=self.len_trend * self.feature_dim)

        # Operations of external component
        if self.ext_dim > 0:
            self.external_ops = nn.Sequential(OrderedDict([
                ('embd', nn.Linear(self.ext_dim, 10, bias=True)),
                ('relu1', nn.ReLU()),
                ('fc', nn.Linear(10, self.output_dim * self.len_row * self.len_column, bias=True)),
                ('relu2', nn.ReLU()),
            ]))

    def make_one_way(self, in_channels):
        return nn.Sequential(OrderedDict([
            ('conv1', conv3x3(in_channels=in_channels, out_channels=64)),
            ('ResUnits', ResUnits(ResidualUnit, nb_filter=64, repetations=self.nb_residual_unit, bn=self.bn)),
            ('relu', nn.ReLU()),
            ('conv2', conv3x3(in_channels=64, out_channels=2)),
            ('FusionLayer', TrainableEltwiseLayer(n=self.output_dim, h=self.len_row,
                                                  w=self.len_column, device=self.device))
        ]))

    def forward(self, batch):
        inputs = batch['X']  # (batch_size, T_c+T_p+T_t, len_row, len_column, feature_dim)
        input_ext = batch['y_ext']  # (batch_size, ext_dim)
        batch_size, len_time, len_row, len_column, input_dim = inputs.shape
        assert len_row == self.len_row
        assert len_column == self.len_column
        assert len_time == self.len_closeness + self.len_period + self.len_trend
        assert input_dim == self.feature_dim

        # Three-way Convolution
        # parameter-matrix-based fusion
        main_output = 0
        if self.len_closeness > 0:
            begin_index = 0
            end_index = begin_index + self.len_closeness
            input_c = inputs[:, begin_index:end_index, :, :, :]
            input_c = input_c.view(-1, self.len_closeness * self.feature_dim, self.len_row, self.len_column)
            out_c = self.c_way(input_c)
            main_output += out_c
        if self.len_period > 0:
            begin_index = self.len_closeness
            end_index = begin_index + self.len_period
            input_p = inputs[:, begin_index:end_index, :, :, :]
            input_p = input_p.view(-1, self.len_period * self.feature_dim, self.len_row, self.len_column)
            out_p = self.p_way(input_p)
            main_output += out_p
        if self.len_trend > 0:
            begin_index = self.len_closeness + self.len_period
            end_index = begin_index + self.len_trend
            input_t = inputs[:, begin_index:end_index, :, :, :]
            input_t = input_t.view(-1, self.len_trend * self.feature_dim, self.len_row, self.len_column)
            out_t = self.t_way(input_t)
            main_output += out_t

        # fusing with external component
        if self.ext_dim > 0:
            external_output = self.external_ops(input_ext)
            external_output = self.relu(external_output)
            external_output = external_output.view(-1, self.feature_dim, self.len_row, self.len_column)
            main_output += external_output
        main_output = self.tanh(main_output)
        main_output = main_output.view(batch_size, 1, len_row, len_column, self.output_dim)
        return main_output

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true)

    def predict(self, batch):
        return self.forward(batch)
