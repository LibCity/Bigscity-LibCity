import numpy as np
from math import sqrt
import torch
import torch.nn as nn
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss

"""
输入流入和流出的2维数据
"""


def calculate_normalized_laplacian(adj):
    adjacency = np.array(adj)
    # use adjacency matrix to calculate D_hat**-1/2 * A_hat *D_hat**-1/2
    I = np.matrix(np.eye(adj.shape[0]))
    A_hat = adjacency + I
    D_hat = np.array(np.sum(A_hat, axis=0))[0]
    D_hat_sqrt = [sqrt(x) for x in D_hat]
    D_hat_sqrt = np.array(np.diag(D_hat_sqrt))
    D_hat_sqrtm_inv = np.linalg.inv(D_hat_sqrt)  # get the D_hat**-1/2 (开方后求逆即为矩阵的-1/2次方)
    # D_A_final = D_hat**-1/2 * A_hat *D_hat**-1/2
    D_A_final = np.dot(D_hat_sqrtm_inv, A_hat)
    D_A_final = np.dot(D_A_final, D_hat_sqrtm_inv)
    return D_A_final


class Unit(nn.Module):
    def __init__(self, in_c, out_c, pool=False):
        super(Unit, self).__init__()
        self.pool1 = nn.MaxPool2d((2, 2), padding=(0, 1))
        self.conv1 = nn.Conv2d(in_c, out_c, 1, 2, 0)

        self.bn1 = nn.BatchNorm2d(in_c)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_c, out_c, 3, 1, 1)

        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_c, out_c, 3, 1, 1)

        self.pool = pool

    def forward(self, x):
        res = x
        if self.pool:
            x = self.pool1(x)
            res = self.conv1(res)
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = res+out
        return out


class Attention3dBlock(nn.Module):
    def __init__(self, num_nodes):
        super(Attention3dBlock, self).__init__()
        self.fc = nn.Linear(num_nodes, num_nodes)

    def forward(self, x_):
        x = x_.permute(0, 2, 1)  # (64, 128, 276)
        x = self.fc(x)
        x_probs = x.permute(0, 2, 1)
        xx = torch.mul(x_, x_probs)
        return xx


class ConvBlock(nn.Module):
    def __init__(self, c_in, num_nodes):
        super(ConvBlock, self).__init__()
        self.conv_layer = nn.Conv2d(c_in, 32, 3, 1, 1)
        self.unit1 = Unit(32, 32)
        self.unit2 = Unit(32, 64, pool=True)
        self.fc = nn.Linear(num_nodes * 96, num_nodes)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class ResLSTM(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._scaler = self.data_feature.get('scaler')
        self.adj_mx = self.data_feature.get('adj_mx')
        self.adj_mx = calculate_normalized_laplacian(self.adj_mx)
        self.num_nodes = data_feature.get('num_nodes', 276)
        self.output_dim = self.data_feature.get('output_dim', 2)
        self.ext_dim = self.data_feature.get('ext_dim', 11)
        self.batch_size = config.get('batch_size', 64)
        self.time_lag = config.get('time_lag', 6)
        self.output_window = config.get('output_window', 1)
        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))

        self.conv_block1 = ConvBlock(3, self.num_nodes)
        self.conv_block2 = ConvBlock(3, self.num_nodes)
        self.conv_block3 = ConvBlock(1, self.num_nodes)
        if self.ext_dim > 0:
            self.fc1 = nn.Linear(self.ext_dim * (self.time_lag - 1), self.num_nodes)
            self.lstm1 = nn.LSTM(input_size=1, hidden_size=128, num_layers=2)
            self.lstm2 = nn.LSTM(input_size=128, hidden_size=1, num_layers=2)
            self.fc2 = nn.Linear(self.num_nodes, self.num_nodes)
        self.lstm3 = nn.LSTM(input_size=1, hidden_size=128, num_layers=2)
        self.att = Attention3dBlock(self.num_nodes)
        self.fc_last = nn.Linear(self.num_nodes*128, self.num_nodes)

    def fourth_pro(self, x):
        x = x.contiguous().view(x.shape[0], -1)
        x = self.fc1(x)
        x = x.view(x.shape[0], self.num_nodes, 1)  # (64, 276, 1)
        x, _ = self.lstm1(x)  # (64, 276, 128)
        x, _ = self.lstm2(x)  # (64, 276, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc2(x)
        return x

    def forward(self, batch):
        input1_ = batch["X"][:, :, :, 0].permute(0, 2, 1)
        input1_ = input1_.reshape(input1_.shape[0], self.num_nodes, self.time_lag - 1, -1)
        input1_ = input1_.permute(0, 3, 1, 2)
        input2_ = batch["X"][:, :, :, 1].permute(0, 2, 1)
        input2_ = input2_.reshape(input2_.shape[0], self.num_nodes, self.time_lag - 1, -1)
        input2_ = input2_.permute(0, 3, 1, 2)
        input3_ = batch["X"][:, -self.time_lag+1:, :, 0].permute(0, 2, 1)
        input3_ = torch.tensor(self.adj_mx, device=self.device, dtype=torch.float32).matmul(input3_)
        input3_ = input3_.unsqueeze(1)

        p1 = self.conv_block1(input1_)
        p2 = self.conv_block2(input2_)
        p3 = self.conv_block3(input3_)

        out = p1 + p2 + p3  # (64, 276)
        if self.ext_dim > 0:
            input4_ = batch["X"][:, -self.time_lag + 1:, 0, -self.ext_dim:].permute(0, 2, 1)
            p4 = self.fourth_pro(input4_)
            out += p4

        out = out.view(out.shape[0], self.num_nodes, 1)  # (64, 276, 1)
        out, _ = self.lstm3(out)  # (64, 276, 128)
        out = self.att(out)  # (64, 276, 128)
        out = out.view(out.shape[0], -1)
        out = self.fc_last(out)
        out = out.unsqueeze(1).unsqueeze(-1)
        return out

    def calculate_loss(self, batch, batches_seen=None):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true)

    def predict(self, batch):
        return self.forward(batch)
