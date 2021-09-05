from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

"""
输入流入和流出的2维数据
"""


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, device, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features).to(device))  # FloatTensor建立tensor
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, out_features, 1, 1)).to(device)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 初始化权重
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adjT):
        # print('input.shape:', input.shape)
        # print('self.weight.shape:', self.weight.shape)
        support = torch.einsum("ijkl, jm->imkl", [input, self.weight])
        # print('support.shape:', support.shape)
        # print('adjT.shape:', adjT.shape)
        output = torch.einsum("ak, ijkl->ijal", [adjT, support])
        # print('output.shape:', output.shape)
        # print('self.bias.shape:', self.bias.shape)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    # def __repr__(self):
    #     return self.__class__.__name__ + ' (' \
    #            + str(self.in_features) + ' -> ' \
    #            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(input_size, hidden_size, device)
        self.gc2 = GraphConvolution(hidden_size, output_size, device)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x


class CONVGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.device = config.get('device', torch.device('cpu'))

        self._scaler = self.data_feature.get('scaler')
        self.adj_mx = torch.tensor(self.data_feature.get('adj_mx'), device=self.device)
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        for i in range(self.num_nodes):
            self.adj_mx[i, i] = 1
        self.feature_dim = self.data_feature.get('feature_dim', 1)  # 输入维度
        self.output_dim = self.data_feature.get('output_dim', 1)  # 输出维度
        # self._logger = getLogger()
        self.conv_depth = config.get('conv_depth', 5)
        self.conv_height = config.get('conv_height', 3)
        self.hidden_size = config.get('hidden_size', 16)
        self.time_lag = config.get('time_lag', 1)
        self.output_window = config.get('output_window', 1)

        self.gc = GCN(
            (self.time_lag-1) * 3,
            self.hidden_size,
            self.conv_depth * self.conv_height,
            self.device
        )
        self.Conv = nn.Conv3d(
            in_channels=self.num_nodes,
            out_channels=self.num_nodes,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(
            self.num_nodes * self.conv_depth * self.conv_height * self.feature_dim,
            self.num_nodes * self.output_window * self.output_dim
        )

    def forward(self, batch):

        x = batch['X']

        out = self.gc(x, self.adj_mx)
        # print('gc_output:', out.shape)
        out = torch.reshape(
            out,
            (-1, self.num_nodes, self.conv_depth, self.conv_height, self.feature_dim)
        )
        # print('conv_in:', out.shape)
        out = self.relu(self.Conv(out))
        # print('conv_out:', out.shape)

        out = out.view(-1, self.num_nodes * self.conv_depth * self.conv_height * self.feature_dim)
        out = self.fc(out)
        out = torch.reshape(out, [-1, self.output_window, self.num_nodes, self.output_dim])
        return out

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        # print('size of y_true:', y_true.shape)
        # print('size of y_predict:', y_predicted.shape)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true, 0)
