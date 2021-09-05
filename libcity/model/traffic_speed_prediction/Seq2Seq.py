import torch
import torch.nn as nn
import random
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class Encoder(nn.Module):
    def __init__(self, device, rnn_type, input_size, hidden_size=64,
                 num_layers=1, dropout=0, bidirectional=False):
        super().__init__()
        self.device = device
        self.rnn_type = rnn_type
        self.layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        if self.rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type.upper() == 'RNN':
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError('Unknown RNN type: {}'.format(self.rnn_type))

    def forward(self, x):
        # x = [seq_len, batch_size, input_size]
        # h_0 = [layers * num_directions, batch_size, hidden_size]
        h_0 = torch.zeros(self.layers * self.num_directions, x.shape[1], self.hidden_size).to(self.device)
        if self.rnn_type == 'LSTM':
            c_0 = torch.zeros(self.layers * self.num_directions, x.shape[1], self.hidden_size).to(self.device)
            out, (hn, cn) = self.rnn(x, (h_0, c_0))
            # output = [seq_len, batch_size, hidden_size * num_directions]
            # hn/cn = [layers * num_directions, batch_size, hidden_size]
        else:
            out, hn = self.rnn(x, h_0)
            cn = torch.zeros(hn.shape)
            # output = [seq_len, batch_size, hidden_size * num_directions]
            # hn = [layers * num_directions, batch_size, hidden_size]
        return hn, cn


class Decoder(nn.Module):
    def __init__(self, device, rnn_type, input_size, hidden_size=64,
                 num_layers=1, dropout=0, bidirectional=False):
        super().__init__()
        self.device = device
        self.rnn_type = rnn_type
        self.layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        if self.rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type.upper() == 'RNN':
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError('Unknown RNN type: {}'.format(self.rnn_type))
        self.fc = nn.Linear(hidden_size * self.num_directions, input_size)

    def forward(self, x, hn, cn):
        # x = [batch_size, input_size]
        # hn, cn = [layers * num_directions, batch_size, hidden_size]
        x = x.unsqueeze(0)
        # x = [seq_len = 1, batch_size, input_size]
        if self.rnn_type == 'LSTM':
            out, (hn, cn) = self.rnn(x, (hn, cn))
        else:
            out, hn = self.rnn(x, hn)
            cn = torch.zeros(hn.shape)
        # out = [seq_len = 1, batch_size, hidden_size * num_directions]
        # hn = [layers * num_directions, batch_size, hidden_size]
        out = self.fc(out.squeeze(0))
        # out = [batch_size, input_size]
        return out, hn, cn


class Seq2Seq(AbstractTrafficStateModel):
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

        self.rnn_type = config.get('rnn_type', 'GRU')
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 1)
        self.dropout = config.get('dropout', 0)
        self.bidirectional = config.get('bidirectional', False)
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 0)
        self.encoder = Encoder(self.device, self.rnn_type, self.num_nodes * self.feature_dim,
                               self.hidden_size, self.num_layers, self.dropout, self.bidirectional)
        self.decoder = Decoder(self.device, self.rnn_type, self.num_nodes * self.output_dim,
                               self.hidden_size, self.num_layers, self.dropout, self.bidirectional)
        self._logger.info('You select rnn_type {} in Seq2Seq!'.format(self.rnn_type))

    def forward(self, batch):
        src = batch['X']  # [batch_size, input_window, num_nodes, feature_dim]
        target = batch['y']  # [batch_size, output_window, num_nodes, feature_dim]
        src = src.permute(1, 0, 2, 3)  # [input_window, batch_size, num_nodes, feature_dim]
        target = target.permute(1, 0, 2, 3)  # [output_window, batch_size, num_nodes, feature_dim]

        batch_size = src.shape[1]
        src = src.reshape(self.input_window, batch_size, self.num_nodes * self.feature_dim)
        target = target[..., :self.output_dim].contiguous().reshape(
            self.output_window, batch_size, self.num_nodes * self.output_dim)
        # src = [self.input_window, batch_size, self.num_nodes * self.feature_dim]
        # target = [self.output_window, batch_size, self.num_nodes * self.output_dim]

        encoder_hn, encoder_cn = self.encoder(src)
        decoder_hn = encoder_hn
        decoder_cn = encoder_cn
        # encoder_hidden_state = [layers * num_directions, batch_size, hidden_size]
        decoder_input = torch.randn(batch_size, self.num_nodes * self.output_dim).to(self.device)
        # decoder_input = [batch_size, self.num_nodes * self.output_dim]

        outputs = []
        for i in range(self.output_window):
            decoder_output, decoder_hn, decoder_cn = \
                self.decoder(decoder_input, decoder_hn, decoder_cn)
            # decoder_output = [batch_size, self.num_nodes * self.output_dim]
            # decoder_hn = [layers * num_directions, batch_size, hidden_size]
            outputs.append(decoder_output.reshape(batch_size, self.num_nodes, self.output_dim))
            # 只有训练的时候才考虑用真值
            if self.training and random.random() < self.teacher_forcing_ratio:
                decoder_input = target[i]
            else:
                decoder_input = decoder_output
        outputs = torch.stack(outputs)
        # outputs = [self.output_window, batch_size, self.num_nodes, self.output_dim]
        return outputs.permute(1, 0, 2, 3)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)
