import torch
import torch.nn as nn
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class AutoEncoder(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)

        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        self.encoder = nn.Sequential(
            nn.Linear(self.input_window * self.num_nodes * self.feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_window * self.num_nodes * self.output_dim)
        )

    def forward(self, batch):
        x = batch['X']  # [batch_size, input_window, num_nodes, feature_dim]
        x = x.reshape(-1, self.input_window * self.num_nodes * self.feature_dim)
        # [batch_size, output_window * num_nodes * feature_dim]
        x = self.encoder(x)  # [batch_size, 16]
        x = self.decoder(x)
        # [batch_size, output_window * num_nodes * output_dim]
        return x.reshape(-1, self.output_window, self.num_nodes, self.output_dim)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)
