import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


def get_spatial_matrix(adj_mx):
    h, w = adj_mx.shape
    inf = float("inf")

    S_near = np.zeros((h, w))
    S_middle = np.zeros((h, w))
    S_distant = np.zeros((h, w))

    i = 0
    for row in adj_mx:
        L_min = np.min(row)
        np.place(row, row == inf, [-1])
        L_max = np.max(row)
        eta = (L_max-L_min)/3
        S_near[i] = np.logical_and(row >= L_min, row < L_min + eta)
        S_middle[i] = np.logical_and(row >= L_min + eta, row < L_min + 2 * eta)
        S_distant[i] = np.logical_and(row >= L_min + 2*eta, row < L_max)
        i = i + 1

    S_near = S_near.astype(np.float32)
    S_middle = S_middle.astype(np.float32)
    S_distant = S_distant.astype(np.float32)
    return torch.tensor(S_near), torch.tensor(S_middle), torch.tensor(S_distant)


class SpatialBlock(nn.Module):
    def __init__(self, n, Smatrix, feature_dim, device):
        super(SpatialBlock, self).__init__()
        self.device = device
        self.S = Smatrix.to(self.device)

        self.linear1 = nn.Linear(n * feature_dim, n * feature_dim)
        self.linear2 = nn.Linear(n * feature_dim, n * feature_dim)

        self.hidden_num = 3

        self.lstm = nn.LSTM(n * feature_dim, n * feature_dim, 1)
        self.lstm2 = nn.LSTM(n * feature_dim, n * feature_dim, 1)

        self.linear3 = nn.Linear(n * feature_dim, n * feature_dim)

    def forward(self, x):
        batch, time, node, feature = x.shape  # (batch, time, node, feature)
        # gcn1
        out = self.S.matmul(x.to(self.device))  # (batch, time, node, feature)
        out = out.reshape(batch, time, node * feature)  # (batch, time, node * feature)
        out = self.linear1(out)  # (batch, time, node * feature)
        out = F.relu(out)

        # gcn2
        out = out.reshape(batch, time, node, feature)
        out = self.S.matmul(out.to(self.device))  # (batch, time, node, feature)
        out = out.reshape(batch, time, node * feature)
        out = self.linear2(out)  # (batch, time, node * feature)
        out = F.relu(out)

        out = out.permute(1, 0, 2)  # (time, batch, node * feature)
        # LSTM
        out, (a, b) = self.lstm(out)  # (time, batch, node * feature)
        out, (a, b) = self.lstm2(out)  # (time, batch, node * feature)
        out = out[-1, :, :]  # (batch, node * feature)

        # Dense
        out = self.linear3(out)  # (batch, node * feature)
        out = F.relu(out)
        return out


class SpatialComponent(nn.Module):
    def __init__(self, n, adj_mx, len_closeness, feature_dim, output_dim, device):
        super(SpatialComponent, self).__init__()

        self.device = device
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_nodes = n
        self.len_closeness = len_closeness

        self.near_matrix, self.middle_matrix, self.distant_matrix = get_spatial_matrix(adj_mx)

        self.near_block = SpatialBlock(self.num_nodes, self.near_matrix, self.feature_dim, self.device)
        self.middle_block = SpatialBlock(self.num_nodes, self.middle_matrix, self.feature_dim, self.device)
        self.distant_block = SpatialBlock(self.num_nodes, self.distant_matrix, self.feature_dim, self.device)

        self.linear = nn.Linear(3 * n * feature_dim, n * output_dim)

    def forward(self, x):
        # (batch, time, node, feature)
        x = x[:, :self.len_closeness, :, :]

        y_near = self.near_block(x)           # (batch, node * feature)
        y_middle = self.middle_block(x)       # (batch, node * feature)
        y_distant = self.distant_block(x)     # (batch, node * feature)

        out = torch.cat((y_near, y_middle, y_distant), 1)

        out = F.relu(self.linear(out))  # (batch, node * output_dim)

        return out


class TemporalBlock(nn.Module):
    def __init__(self, n, feature_dim, device):
        super(TemporalBlock, self).__init__()
        self.device = device
        self.lstm = nn.LSTM(n * feature_dim, n * feature_dim, 1)
        self.lstm2 = nn.LSTM(n * feature_dim, n * feature_dim, 1)
        self.linear = nn.Linear(n * feature_dim, n * feature_dim)

    def forward(self, x):
        # (batch, time, node, feature)
        batch, time, node, feature = x.shape
        out = x.reshape(batch, time, node * feature)

        # (time, batch, node * feature)
        out = out.permute(1, 0, 2)

        # (time, batch, node * feature)
        out, (a, b) = self.lstm(out)
        out, (a, b) = self.lstm2(out)
        out = out[-1, :, :]

        # (batch, node * feature)
        out = F.relu(self.linear(out))
        # (batch, node * feature)
        return out


class TemporalComponent(nn.Module):
    def __init__(self, n, len_closeness, len_period, len_trend, feature_dim, output_dim, device):
        super(TemporalComponent, self).__init__()

        self.num_nodes = n
        self.len_closeness = len_closeness
        self.len_period = len_period
        self.len_trend = len_trend
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.device = device

        self.daily_block = TemporalBlock(self.num_nodes, self.feature_dim, self.device)
        self.interval_block = TemporalBlock(self.num_nodes, self.feature_dim, self.device)
        self.weekly_block = TemporalBlock(self.num_nodes, self.feature_dim, self.device)

        count = 0
        if self.len_closeness > 0:
            count = count + 1
        if self.len_period > 0:
            count = count + 1
        if self.len_trend > 0:
            count = count + 1
        self.linear = nn.Linear(count * n * feature_dim, n * output_dim)

    def forward(self, x):
        # (batch, time, node, feature)
        list_y = []
        if self.len_closeness > 0:
            begin_index = 0
            end_index = begin_index + self.len_closeness
            x_interval = x[:, begin_index:end_index, :, :]
            y_interval = self.daily_block(x_interval)  # batch*n
            list_y.append(y_interval)  # (batch, node * feature)
        if self.len_period > 0:
            begin_index = self.len_closeness
            end_index = begin_index + self.len_period
            x_daily = x[:, begin_index:end_index, :, :]
            y_daily = self.daily_block(x_daily)  # batch*n
            list_y.append(y_daily)  # (batch, node * feature)
        if self.len_trend > 0:
            begin_index = self.len_closeness + self.len_period
            end_index = begin_index + self.len_trend
            x_weekly = x[:, begin_index:end_index, :, :]
            y_weekly = self.weekly_block(x_weekly)   # batch*n
            list_y.append(y_weekly)  # (batch, node * feature)

        out = torch.cat(list_y, 1)

        out = F.relu(self.linear(out))  # (batch, node * output_dim)
        return out


class MultiSTGCnet(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.adj_mx = self.data_feature.get("adj_mx")
        self.num_nodes = self.data_feature.get("num_nodes", 1)
        self.feature_dim = self.data_feature.get("feature_dim", 1)
        self.output_dim = self.data_feature.get('output_dim', 1)

        self.len_period = self.data_feature.get('len_period', 0)
        self.len_trend = self.data_feature.get('len_trend', 0)
        self.len_closeness = self.data_feature.get('len_closeness', 0)
        if self.len_period == 0 and self.len_trend == 0 and self.len_closeness == 0:
            raise ValueError('Num of days/weeks/hours are all zero! Set at least one of them not zero!')

        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self._scaler = self.data_feature.get('scaler')
        self._logger = getLogger()

        # get model config
        self.hidden_size = config.get("hidden_size", 64)
        self.num_layers = config.get("num_layers", 1)
        self.device = config.get('device', torch.device('cpu'))

        # define the model structure
        self.spatial_component = SpatialComponent(self.num_nodes, self.adj_mx,
                                                  self.len_closeness, self.feature_dim, self.output_dim, self.device)
        self.temporal_component = TemporalComponent(self.num_nodes,
                                                    self.len_closeness, self.len_period,
                                                    self.len_trend, self.feature_dim, self.output_dim, self.device)
        # fusion的参数
        self.Ws = nn.Parameter(torch.tensor(np.random.normal(0, 0.01, (1, self.num_nodes * self.output_dim)),
                                            dtype=torch.float32).to(self.device))
        self.Wt = nn.Parameter(torch.tensor(np.random.normal(0, 0.01, (1, self.num_nodes * self.output_dim)),
                                            dtype=torch.float32).to(self.device))
        self.count = 0

    def forward(self, batch):
        x = batch['X']  # (batch, time, node, feature)

        y_spatial = self.spatial_component(x)  # (batch, node * output_dim)

        y_temporal = self.temporal_component(x)  # (batch, node * output_dim)

        y = torch.mul(self.Ws, y_spatial) + torch.mul(self.Wt, y_temporal)  # (batch, node * output_dim)

        return y.reshape(-1, 1, self.num_nodes, self.output_dim)

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        y_true = batch['y']  # ground-truth value
        y_predicted = self.predict(batch)  # prediction results
        # print('y_true', y_true.shape)
        # print('y_predicted', y_predicted.shape)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true)
