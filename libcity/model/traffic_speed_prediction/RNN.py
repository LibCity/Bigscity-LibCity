import torch
import torch.nn as nn
import random
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class RNN(AbstractTrafficStateModel):
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

        self.rnn_type = config.get('rnn_type', 'RNN')
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 1)
        self.dropout = config.get('dropout', 0)
        self.bidirectional = config.get('bidirectional', False)
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 0)
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.input_size = self.num_nodes * self.feature_dim

        self._logger.info('You select rnn_type {} in RNN!'.format(self.rnn_type))
        if self.rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                              num_layers=self.num_layers, dropout=self.dropout,
                              bidirectional=self.bidirectional)
        elif self.rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                               num_layers=self.num_layers, dropout=self.dropout,
                               bidirectional=self.bidirectional)
        elif self.rnn_type.upper() == 'RNN':
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size,
                              num_layers=self.num_layers, dropout=self.dropout,
                              bidirectional=self.bidirectional)
        else:
            raise ValueError('Unknown RNN type: {}'.format(self.rnn_type))
        self.fc = nn.Linear(self.hidden_size * self.num_directions, self.num_nodes * self.output_dim)

    def forward(self, batch):
        src = batch['X'].clone()  # [batch_size, input_window, num_nodes, feature_dim]
        target = batch['y']  # [batch_size, output_window, num_nodes, feature_dim]
        src = src.permute(1, 0, 2, 3)  # [input_window, batch_size, num_nodes, feature_dim]
        target = target.permute(1, 0, 2, 3)  # [output_window, batch_size, num_nodes, output_dim]

        batch_size = src.shape[1]
        src = src.reshape(self.input_window, batch_size, self.num_nodes * self.feature_dim)
        # src = [self.input_window, batch_size, self.num_nodes * self.feature_dim]
        outputs = []
        for i in range(self.output_window):
            # src: [input_window, batch_size, num_nodes * feature_dim]
            out, _ = self.rnn(src)
            # out: [input_window, batch_size, hidden_size * num_directions]
            out = self.fc(out[-1])
            # out: [batch_size, num_nodes * output_dim]
            out = out.reshape(batch_size, self.num_nodes, self.output_dim)
            # out: [batch_size, num_nodes, output_dim]
            outputs.append(out.clone())
            if self.output_dim < self.feature_dim:  # output_dim可能小于feature_dim
                out = torch.cat([out, target[i, :, :, self.output_dim:]], dim=-1)
            # out: [batch_size, num_nodes, feature_dim]
            if self.training and random.random() < self.teacher_forcing_ratio:
                src = torch.cat((src[1:, :, :], target[i].reshape(
                    batch_size, self.num_nodes * self.feature_dim).unsqueeze(0)), dim=0)
            else:
                src = torch.cat((src[1:, :, :], out.reshape(
                    batch_size, self.num_nodes * self.feature_dim).unsqueeze(0)), dim=0)
        outputs = torch.stack(outputs)
        # outputs = [output_window, batch_size, num_nodes, output_dim]
        return outputs.permute(1, 0, 2, 3)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)
