from logging import getLogger

import torch
import torch.nn as nn

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))  # MLP
        hidden = hidden + input_data  # residual
        return hidden


class STID(AbstractTrafficStateModel):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.num_nodes = data_feature.get('num_nodes')
        self.input_window = config.get('input_window')
        self.output_window = config.get('output_window')
        self.feature_dim = data_feature.get('feature_dim', 2)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.time_intervals = config.get('time_intervals')
        self._scaler = self.data_feature.get('scaler')

        self.num_block = config.get('num_block')
        self.time_series_emb_dim = config.get('time_series_emb_dim')
        self.spatial_emb_dim = config.get('spatial_emb_dim')
        self.temp_dim_tid = config.get('temp_dim_tid')
        self.temp_dim_diw = config.get('temp_dim_diw')
        self.if_spatial = config.get('if_spatial')
        self.if_time_in_day = config.get('if_TiD')
        self.if_day_in_week = config.get('if_DiW')

        self.device = config.get('device', torch.device('cpu'))

        assert (24 * 60 * 60) % self.time_intervals == 0, "time_of_day_size should be Int"
        self.time_of_day_size = int((24 * 60 * 60) / self.time_intervals)
        self.day_of_week_size = 7

        self._logger = getLogger()

        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.spatial_emb_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.output_dim * self.input_window, out_channels=self.time_series_emb_dim, kernel_size=(1, 1),
            bias=True)

        # encoding
        self.hidden_dim = self.time_series_emb_dim + self.spatial_emb_dim * int(self.if_spatial) + \
                          self.temp_dim_tid * int(self.if_day_in_week) + self.temp_dim_diw * int(self.if_time_in_day)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_block)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_window, kernel_size=(1, 1), bias=True)

    def forward(self, batch):
        # prepare data
        input_data = batch['X']  # [B, L, N, C]
        time_series = input_data[..., :1]

        if self.if_time_in_day:
            tid_data = input_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[(tid_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            diw_data = torch.argmax(input_data[..., 2:], dim=-1)
            day_in_week_emb = self.day_in_week_emb[(diw_data[:, -1, :]).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = time_series.shape
        time_series = time_series.transpose(1, 2).contiguous()
        time_series = time_series.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(time_series)

        node_emb = []
        if self.if_spatial:
            node_emb.append(self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)  # concat all embeddings

        hidden = self.encoder(hidden)
        prediction = self.regression_layer(hidden)

        return prediction

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)
