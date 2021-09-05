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


class STResNetCommon(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._scaler = self.data_feature.get('scaler')
        self.adj_mx = self.data_feature.get('adj_mx')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 2)  # 这种情况下包括外部数据的维度
        self.ext_dim = self.data_feature.get('ext_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 2)
        self.len_row = self.data_feature.get('len_row', 32)
        self.len_column = self.data_feature.get('len_column', 32)
        self._logger = getLogger()

        self.nb_residual_unit = config.get('nb_residual_unit', 12)
        self.bn = config.get('batch_norm', False)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        self.relu = torch.relu
        self.tanh = torch.tanh

        self.model = nn.Sequential(OrderedDict([
            ('conv1', conv3x3(in_channels=self.input_window * self.output_dim, out_channels=64)),
            ('ResUnits', ResUnits(ResidualUnit, nb_filter=64, repetations=self.nb_residual_unit, bn=self.bn)),
            ('relu', nn.ReLU()),
            ('conv2', conv3x3(in_channels=64, out_channels=2)),
            ('FusionLayer', TrainableEltwiseLayer(n=self.output_dim, h=self.len_row,
                                                  w=self.len_column, device=self.device))
        ]))

        # Operations of external component
        if self.ext_dim > 0:
            self.external_ops = nn.Sequential(OrderedDict([
                ('embd', nn.Linear(self.ext_dim, 10, bias=True)),
                ('relu1', nn.ReLU()),
                ('fc', nn.Linear(10, self.output_dim * self.len_row * self.len_column, bias=True)),
                ('relu2', nn.ReLU()),
            ]))

    def forward(self, batch):
        inputs = batch['X'][:, :, :, :, :self.output_dim]  # (batch_size, input_window, len_row, len_column, output_dim)
        input_ext = batch['X'][:, -1, 0, 0, self.output_dim:]  # (batch_size, ext_dim)
        # print(inputs.shape, input_ext.shape)
        batch_size, len_time, len_row, len_column, input_dim = inputs.shape
        assert len_row == self.len_row
        assert len_column == self.len_column
        assert len_time == self.input_window
        assert input_dim == self.output_dim

        inputs = inputs.contiguous().view(-1, self.input_window * self.output_dim,
                                          self.len_row, self.len_column).to(self.device)
        output = self.model(inputs)
        # print('output', output.shape)

        # fusing with external component
        if self.ext_dim > 0:
            # print('4', input_ext.shape)
            input_ext = input_ext.contiguous().view(-1, self.ext_dim)
            # print('3', input_ext.shape)
            external_output = self.external_ops(input_ext)
            # print('2', external_output.shape)
            external_output = self.relu(external_output)
            # print('1', external_output.shape)
            external_output = external_output.view(-1, self.output_dim, self.len_row, self.len_column)
            # print('external_output', external_output.shape)
            output += external_output
        output = self.tanh(output)
        output = output.view(batch_size, 1, len_row, len_column, self.output_dim)
        return output  # (batch_size, 1, len_row, len_column, output_dim)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        # print(y_true.shape, y_predicted.shape)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true)

    def predict(self, batch):
        # 多步预测
        x = batch['X']  # (batch_size, input_window, len_row, len_column, feature_dim)
        y = batch['y']  # (batch_size, input_window, len_row, len_column, feature_dim)
        y_preds = []
        x_ = x.clone()
        for i in range(self.output_window):
            batch_tmp = {'X': x_}
            y_ = self.forward(batch_tmp)  # (batch_size, 1, len_row, len_column, output_dim)
            y_preds.append(y_.clone())
            if y_.shape[-1] < x_.shape[-1]:  # output_dim < feature_dim
                y_ = torch.cat([y_, y[:, i:i + 1, :, :, self.output_dim:]], dim=-1)
            x_ = torch.cat([x_[:, 1:, :, :, :], y_], dim=1)
        y_preds = torch.cat(y_preds, dim=1)  # (batch_size, output_length, len_row, len_column, output_dim)
        return y_preds
