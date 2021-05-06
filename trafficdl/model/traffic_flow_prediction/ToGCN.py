from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel
from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from trafficdl.model import loss
import math
import random
dtype = torch.float


class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(GraphConvolution, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.zeros((input_size, output_size), device=device, dtype=dtype),
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(output_size, device=device, dtype=dtype), requires_grad=True)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, A):
        x = torch.einsum("ijk, kl->ijl", [x, self.weight])
        x = torch.einsum("ij, kjl->kil", [A, x])
        x = x + self.bias

        return x


class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, extra_dim):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_size * extra_dim, hidden_size, device)
        self.gcn2 = GraphConvolution(hidden_size, output_size * extra_dim, device)
        # self.gcn = GraphConvolution(input_size, output_size)

    def forward(self, x, A):
        x = self.gcn1(x, A)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.gcn2(x, A)
        x = F.relu(x)
        # x = self.gcn(x, A)

        return x


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, device, extra_dim):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gcn = GCN(input_size=1, hidden_size=128, output_size=1, device=device, extra_dim=extra_dim)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5)
        self.device = device

    def forward(self, x, A, hidden=None):
        # batch_size, timestep, N = x.size()
        # gcn_in = x.view((batch_size * timestep, -1))
        # gcn_out = self.gcn(gcn_in, A)
        # encoder_in = gcn_out.view((batch_size, timestep, -1))
        gcn_in = x.view((x.size(0), x.size(1), -1))
        gcn_out = self.gcn(gcn_in, A)
        encoder_in = gcn_out.reshape((x.size(0), 1, -1))
        encoder_out, encoder_states = self.lstm(encoder_in, hidden)

        return encoder_out, encoder_states

    def init_hidden(self, x):
        return torch.zeros((2, x.size(0), self.hidden_size), device=self.device, dtype=dtype)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, extra_dim):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(
            input_size=self.input_size * extra_dim,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5)
        self.dense = nn.Linear(self.hidden_size, self.output_size)
        self.device = device

    def forward(self, x, hidden=None):
        x_view = x.view(x.size(0), 1, -1)
        x_a, decoder_states = self.lstm(x_view, hidden)
        x_a_view = x_a.view(x_a.size(0), -1)
        # x = F.relu(x)
        x_b = self.dense(x_a_view)
        decoder_out = F.relu(x_b)

        return decoder_out, decoder_states

    def init_hidden(self, x):
        return torch.zeros((2, x.size(0), self.hidden_size), device=self.device, dtype=dtype)


class ToGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        torch.autograd.set_detect_anomaly(True)
        # get data feature
        self.device = config.get('device', torch.device('cpu'))
        # print('self.device=', self.device)
        self.adj_mx = torch.tensor(self.data_feature.get('adj_mx'), device=self.device)
        # print('self.adj_mx=', self.adj_mx)
        # print('self.adj_mx.shape=', self.adj_mx.shape)
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        # print('self.num_nodes=', self.num_nodes)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        # print('self.feature_dim=', self.feature_dim)
        self.output_dim = self.data_feature.get('output_dim', 1)
        # print('self.output_dim=', self.output_dim)
        self._scaler = self.data_feature.get('scaler')
        # print('self._scaler', self._scaler)

        # get model config
        self.hidden_size = config.get('hidden_size', 128)
        # print('self.hidden_size=', self.hidden_size)
        self.decoder_t = config.get('decoder_t', 3)
        # print('self.decoder_t=', self.decoder_t)
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 0.5)
        # print('self.teacher_forcing_ratio=', self.teacher_forcing_ratio)
        self.load_external = config.get('load_external')
        self.add_time_in_day = config.get('add_time_in_day')
        self.add_day_in_week = config.get('add_day_in_week')
        # init logger
        self._logger = getLogger()
        # define the model structure
        extra_dim = 1
        if self.load_external:
            if self.add_time_in_day:
                extra_dim += 1
            if self.add_day_in_week:
                extra_dim += 1
        self.encoder = Encoder(self.num_nodes * self.feature_dim, self.hidden_size, self.device, extra_dim)
        self.decoder = Decoder(self.num_nodes,
                               self.hidden_size, self.num_nodes * self.output_dim, self.device, extra_dim)
        if self.device == 'cuda':
            self.encoder.cuda()
            self.decoder.cuda()

    def forward(self, batch):
        input_tensor = batch['X']  # (batch_size, input_window, feature_size, ?)
        target_tensor = batch['y']  # (batch_size, input_window, feature_size, ?)
        timestep_1 = input_tensor.shape[1]  # Length of input time interval (10 min each)
        timestep_2 = target_tensor.shape[1]  # Length of output time interval (10 min each)
        # print('timestep_2.size():', timestep_2)
        # Encode history flow map
        encoder_hidden = None
        for ei in range(timestep_1):
            encoder_input = input_tensor[:, ei]
            encoder_output, encoder_hidden = self.encoder(encoder_input, self.adj_mx, encoder_hidden)

        # Decode to predict future flow map
        decoder_hidden = encoder_hidden
        for di in range(self.decoder_t):
            decoder_input = input_tensor[:, timestep_1 - (self.decoder_t - di) - 1].clone().detach()
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

        decoder_input = input_tensor[:, timestep_1 - 1].clone().detach()

        # Teacher forcing mechanism.
        if random.random() < self.teacher_forcing_ratio:
            use_teacher_forcing = True
        else:
            use_teacher_forcing = False
        decoder_outputs = []
        if use_teacher_forcing:
            for di in range(timestep_2):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                decoder_outputs.append(decoder_output)
                decoder_input = target_tensor[:, di].clone()
        else:
            for di in range(timestep_2):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                decoder_outputs.append(decoder_output)
                decoder_input = decoder_output
        y_preds = torch.stack(decoder_outputs, dim=1)  # multi-step prediction
        y_preds = y_preds.unsqueeze(3)
        return y_preds

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        y_true = batch['y']  # ground-truth value
        y_predicted = self.predict(batch)  # prediction results
        # denormalization the value
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        # call the mask_mae loss function defined in `loss.py`
        return loss.masked_mae_torch(y_predicted, y_true, 0)
