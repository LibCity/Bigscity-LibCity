import torch
import torch.nn as nn
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class NewModel(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # section 1: data_feature
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._logger = getLogger()
        # section 2: model config
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 1)
        self.dropout = config.get('dropout', 0)
        # section 3: model structure
        self.rnn = nn.LSTM(input_size=self.num_nodes * self.feature_dim, hidden_size=self.hidden_size,
                           num_layers=self.num_layers, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size, self.num_nodes * self.output_dim)

    def forward(self, batch):
        src = batch['X'].clone()
        src = src.permute(1, 0, 2, 3)
        batch_size = src.shape[1]
        src = src.reshape(self.input_window, batch_size, self.num_nodes * self.feature_dim)
        outputs = []
        for i in range(self.output_window):
            out, _ = self.rnn(src)
            out = self.fc(out[-1])
            out = out.reshape(batch_size, self.num_nodes, self.output_dim)
            outputs.append(out.clone())
            src = torch.cat((src[1:, :, :], out.reshape(
                batch_size, self.num_nodes * self.feature_dim).unsqueeze(0)), dim=0)
        outputs = torch.stack(outputs)
        return outputs.permute(1, 0, 2, 3)

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)
