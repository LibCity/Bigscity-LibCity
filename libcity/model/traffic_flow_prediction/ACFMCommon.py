import torch
import torch.nn as nn
from logging import getLogger

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


def split_cpt(value, cpt):
    if not isinstance(value, torch.Tensor):
        raise ValueError('Parameter Value should be a Tensor.')
    scale = int(value.size()[1]) // sum(cpt)
    split_list = []
    for i in cpt:
        if i > 0:
            split_list.append(i * scale)
    if len(split_list) <= 0:
        raise ValueError('Get empty split_list.')
    return torch.split(value, split_size_or_sections=split_list, dim=1)


class ConcatConv(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, inter_channels, relu_conv=False, seq_len=None):
        super().__init__()
        self.in_channels1 = in_channels1
        self.in_channels2 = in_channels2
        self.in_channels = in_channels1 + in_channels2
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.relu_conv = relu_conv
        self.seq_len = seq_len
        if seq_len is not None:
            self.in_channels //= seq_len
            self.out_channels //= (seq_len)

            self.model = nn.ModuleList()
            for _ in range(self.seq_len):
                self.model.append(self._layer())
        else:
            self.model = self._layer()

    def _conv_layer(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def _layer(self):
        if not self.relu_conv:
            return self._conv_layer(self.in_channels, self.out_channels)
        else:
            conv1 = self._conv_layer(self.in_channels, self.inter_channels)
            # relu = nn.ReLU(inplace=True)
            selu = nn.SELU(inplace=True)
            conv2 = self._conv_layer(self.inter_channels, self.out_channels)
            # return nn.Sequential(conv1, relu, conv2)
            return nn.Sequential(conv1, selu, conv2)

    def forward(self, x, y):
        if self.seq_len is not None:
            x_splited_list = torch.split(x, self.in_channels1 // self.seq_len, dim=1)
            y_splited_list = torch.split(y, self.in_channels2 // self.seq_len, dim=1)

            outlist = []
            for i in range(self.seq_len):
                input = torch.cat([x_splited_list[i], y_splited_list[i]], dim=1)
                outlist.append(self.model[i](input))
            return torch.cat(outlist, dim=1)
        else:
            input = torch.cat([x, y], dim=1)
            return self.model(input)


class ConvGate(nn.Module):
    def __init__(self, in_channels, height, width, lstm_channels=16, peephole_conn=True):
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.lstm_channels = lstm_channels
        self.peephole_conn = peephole_conn

        self.conv_x = self._conv_layer(in_channels)
        self.conv_h = self._conv_layer(lstm_channels)

        if peephole_conn:
            self.w = nn.Parameter(torch.Tensor(lstm_channels, height, width))
            self.b = nn.Parameter(torch.Tensor(lstm_channels, 1, 1))
            nn.init.kaiming_normal_(self.w.data, a=0, mode='fan_in')

    def _linear(self, x):
        return x * self.w + self.b

    def _conv_layer(self, in_channels):
        return nn.Conv2d(in_channels, self.lstm_channels, 3, 1, 1, bias=True)  # has bias

    def forward(self, input, state):
        hidden_state, cell_state = state
        convx = self.conv_x(input)
        convh = self.conv_h(hidden_state)
        if self.peephole_conn:
            conv = convx + convh + self._linear(cell_state)
            return torch.sigmoid(conv)
        else:
            conv = convx + convh
            return torch.tanh(conv)


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, height, width, lstm_channels=16):
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.lstm_channels = lstm_channels

        self.f_gate = ConvGate(in_channels, height, width, lstm_channels)
        self.i_gate = ConvGate(in_channels, height, width, lstm_channels)
        self.c_gate = ConvGate(in_channels, height, width, lstm_channels, False)
        # The naming of 'c_gate' may be confusing, but it's just like the gate's operation.
        self.o_gate = ConvGate(in_channels, height, width, lstm_channels)

    def forward(self, input, state):
        hidden_pre, cell_pre = state
        f = self.f_gate(input, state)
        i = self.i_gate(input, state)
        cell_cur = f * cell_pre + i * self.c_gate(input, state)
        o = self.o_gate(input, (hidden_pre, cell_cur))
        hidden_cur = o * torch.tanh(cell_cur)

        return hidden_cur, cell_cur


class ConvGRUCell(nn.Module):
    def __init__(self, in_channels, height, width, lstm_channels=16):
        super().__init__()

        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.lstm_channels = lstm_channels

        self.z_conv = self._conv_layer(in_channels + lstm_channels)
        self.r_conv = self._conv_layer(in_channels + lstm_channels)
        self.h_conv = self._conv_layer(in_channels + lstm_channels)

    def _conv_layer(self, in_channels):
        return nn.Conv2d(in_channels, self.lstm_channels, 3, 1, 1, bias=True)  # has bias

    def forward(self, input, state):
        hidden_pre = state
        mix_input = torch.cat([hidden_pre, input], dim=1)
        z_t = torch.sigmoid(self.z_conv(mix_input))
        r_t = torch.sigmoid(self.r_conv(mix_input))
        mix_input = torch.cat([r_t * hidden_pre, input], dim=1)
        h_t_hat = torch.tanh(self.h_conv(mix_input))
        h_t = (1 - z_t) * hidden_pre + z_t * h_t_hat
        return h_t


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, height, width, lstm_channels=16, all_hidden=False,
                 mode='merge', cpt=None, dropout_rate=0.5, last_conv=False,
                 conv_channels=None, gru=False):
        super().__init__()

        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.lstm_channels = lstm_channels
        self.all_hidden = all_hidden
        self.mode = mode
        self.cpt = cpt
        self.dropout_rate = dropout_rate
        self.last_conv = last_conv
        self.conv_channels = conv_channels
        self.gru = gru

        if gru:
            self._lstm_cell = ConvGRUCell(in_channels, height, width, lstm_channels)
        else:
            self._lstm_cell = ConvLSTMCell(in_channels, height, width, lstm_channels)
        if last_conv:
            if self.conv_channels is None:
                raise ValueError('Parameter Out Channel is needed to enable last_conv')

            self._conv_layer = nn.Conv2d(lstm_channels, conv_channels, 3, 1, 1, bias=True)

        if dropout_rate > 0:
            self._dropout_layer = nn.Dropout2d(dropout_rate)

    def lstm_layer(self, inputs):
        n_in, c_in, h_in, w_in = inputs.size()
        if self.gru:
            state = torch.zeros(n_in, self.lstm_channels, h_in, w_in).cuda()
        else:
            state = (torch.zeros(n_in, self.lstm_channels, h_in, w_in).cuda(),
                     torch.zeros(n_in, self.lstm_channels, h_in, w_in).cuda())
        seq = torch.split(inputs, self.in_channels, dim=1)
        hiddent_list = []
        for idx, input in enumerate(seq[::-1]):  # using reverse order
            state = self._lstm_cell(input, state)
            if self.gru:
                hidden = state
            else:
                hidden = state[0]

            if self.last_conv:
                if self.conv_channels is None:
                    raise ValueError('Parameter Out Channel is needed to enable last_conv')
                hidden = self._conv_layer(hidden)

            hiddent_list.append(hidden)

        if not self.all_hidden:
            return hiddent_list[-1]
        else:
            hiddent_list.reverse()
            return torch.cat(hiddent_list, 1)

    def forward(self, inputs):
        if self.dropout_rate > 0:
            inputs = self._dropout_layer(inputs)
        if self.mode == 'merge':
            output = self.lstm_layer(inputs)
            return output
        elif self.mode == 'cpt':
            if self.cpt is None:
                raise ValueError('Parameter \'cpt\' is required in mode \'cpt\' of ConvLSTM')
            cpt_seq = split_cpt(inputs, self.cpt)
            output_list = [
                self.lstm_layer(input_) for input_ in cpt_seq
            ]
            output = torch.cat(output_list, 1)
            return output
        else:
            raise ('Invalid LSTM mode: ' + self.mode)


class ExtNN(nn.Module):
    def __init__(self, in_features, out_height, out_width, out_channels,
                 inter_features=10, map=True, relu=True, mode='inter', dropout_rate=0):
        super().__init__()
        self.in_features = in_features
        self.out_height = out_height
        self.out_width = out_width
        self.out_channels = out_channels
        self.inter_features = inter_features
        self.map = map
        self.relu = relu
        self.mode = mode
        self.dropout_rate = dropout_rate

        self.out_features = self.out_height * self.out_width * self.out_channels

        self.model = self.external_block()

    def external_block(self):
        layers = []
        layers.append(nn.Linear(self.in_features, self.inter_features))
        # layers.append(nn.ReLU(inplace=True))
        if self.dropout_rate > 0:
            layers.append(nn.Dropout(self.dropout_rate))
        layers.append(nn.SELU(inplace=True))
        layers.append(nn.Linear(self.inter_features, self.out_features))
        if self.relu:
            # layers.append(nn.ReLU(inplace=True))
            layers.append(nn.SELU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.mode == 'inter':
            inputs = torch.split(x, 1, dim=1)
            exts = []
            for input in inputs:
                input = input.squeeze(1)
                out = self.model(input)
                if self.map:
                    out = out.view(-1, self.out_channels, self.out_height, self.out_width)
                exts.append(out)
            return torch.cat(exts, 1)
        else:
            out = self.model(x)
            if self.map:
                out = out.view(-1, self.out_channels, self.out_height, self.out_width)
            return out


class ResUnit(nn.Module):
    def __init__(self, filters, bnmode=True):
        super().__init__()
        self.filters = filters
        self.bnmode = bnmode

        self.layer1 = self._bn_relu_conv()
        self.layer2 = self._bn_relu_conv()

    def _conv_layer(self):
        return nn.Conv2d(self.filters, self.filters, 3, 1, 1)

    def _bn_relu_conv(self):
        layers = []
        if self.bnmode:
            layers.append(nn.BatchNorm2d(self.filters))
        # layers.append(nn.ReLU(inplace=True))
        layers.append(nn.SELU(inplace=True))
        layers.append(self._conv_layer())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = self.layer1(x)
        residual = self.layer2(residual)
        out = residual + x
        return out


class ResNN(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, repetation=1,
                 bnmode=True, splitmode='split', cpt=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.repetation = repetation
        self.bnmode = bnmode

        self.inlist = []
        self.resblocks = nn.ModuleList()
        if splitmode == 'split':
            seq_num = sum(cpt)
            inscale = int(in_channels) // seq_num
            outscale = int(out_channels) // seq_num
            resblock = self.residual_block(inscale, outscale)
            for i in range(seq_num):
                self.inlist.append(inscale)
                self.resblocks.append(resblock)
        elif splitmode == 'split-chans':
            seq_num = sum(cpt) * 2
            inscale = int(in_channels) // seq_num
            outscale = int(out_channels) // seq_num
            resblock = self.residual_block(inscale, outscale)
            for i in range(seq_num):
                self.inlist.append(inscale)
                self.resblocks.append(resblock)
        elif splitmode == 'concat':
            self.inlist.append(in_channels)
            self.resblocks.append(self.residual_block(in_channels, out_channels))
        elif splitmode == 'cpt':
            seq_num = sum(cpt)
            inscale = int(in_channels) // seq_num
            outscale = int(out_channels) // seq_num
            for i in cpt:
                if i > 0:
                    self.inlist.append(i * inscale)
                    self.resblocks.append(self.residual_block(i * inscale, i * outscale))
        elif splitmode == 'cpt-sameoutput':
            seq_num = sum(cpt)
            inscale = int(in_channels) // seq_num
            for i in cpt:
                if i > 0:
                    self.inlist.append(i * inscale)
                    self.resblocks.append(self.residual_block(i * inscale, 2))
        else:
            raise ValueError('Invalid ResNN split mode')

    def residual_block(self, in_channels, out_channels):
        layers = []
        layers.append(nn.Conv2d(in_channels, self.inter_channels, 3, 1, 1))
        for _ in range(self.repetation):
            layers.append(ResUnit(self.inter_channels, self.bnmode))
        # layers.append(nn.ReLU(inplace=True))
        layers.append(nn.SELU(inplace=True))
        layers.append(nn.Conv2d(self.inter_channels, out_channels, 3, 1, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        inputs = torch.split(x, split_size_or_sections=self.inlist, dim=1)
        if len(inputs) != len(self.resblocks):
            raise ValueError('Input length and network in_channels are inconsistent')

        outputs = []
        for i in range(len(inputs)):
            outputs.append(self.resblocks[i](inputs[i]))

        return torch.cat(outputs, dim=1)


class ACFMCommon(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._scaler = self.data_feature.get('scaler')
        self.adj_mx = self.data_feature.get('adj_mx')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 2)
        self.ext_dim = self.data_feature.get('ext_dim', 0)
        self.output_dim = self.data_feature.get('output_dim', 2)
        self.len_row = self.data_feature.get('len_row', 32)
        self.len_column = self.data_feature.get('len_column', 32)
        self._logger = getLogger()

        self.res_repetation = config.get('res_repetation', 12)
        self.res_nbfilter = config.get('res_nbfilter', 16)
        self.res_bn = config.get('res_bn', True)
        self.res_split_mode = config.get('res_split_mode', 'split')  # 'split', 'split-chans', 'cpt', 'concat', 'none'
        self.first_extnn_inter_channels = config.get('first_extnn_inter_channels', 40)
        self.first_extnn_dropout = config.get('first_extnn_dropout', 0.5)
        self.merge_mode = config.get('merge_mode', 'fuse')  # 'LSTM', 'fuse'
        self.lstm_channels = config.get('lstm_channels', 16)
        self.lstm_dropout = config.get('lstm_dropout', 0)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))

        self.resnn = ResNN(
            in_channels=self.output_dim * self.input_window,
            out_channels=self.lstm_channels * self.input_window,
            inter_channels=self.res_nbfilter,
            repetation=self.res_repetation,
            bnmode=self.res_bn,
            splitmode=self.res_split_mode,
            cpt=[self.input_window, 0, 0]
        )

        self.conv_lstm = ConvLSTM(
            in_channels=self.lstm_channels,
            height=self.len_row,
            width=self.len_column,
            lstm_channels=self.lstm_channels,
            all_hidden=True,
            mode='cpt',
            cpt=[self.input_window, 0, 0],
            dropout_rate=self.lstm_dropout,
            last_conv=False
        )

        self.concat_conv_c = ConcatConv(
            in_channels1=2 * self.input_window,
            in_channels2=self.lstm_channels * self.input_window,
            out_channels=self.lstm_channels * self.input_window,
            inter_channels=self.lstm_channels,
            relu_conv=True,
            seq_len=self.input_window
        )

        self.conv_lstm_c = ConvLSTM(
            in_channels=self.lstm_channels,
            height=self.len_row,
            width=self.len_column,
            lstm_channels=self.lstm_channels,
            all_hidden=False,
            mode='merge',
            dropout_rate=self.lstm_dropout,
            last_conv=True,
            conv_channels=2,
        )

        self.conv_lstm_t = ConvLSTM(
            in_channels=self.lstm_channels,
            height=self.len_row,
            width=self.len_column,
            lstm_channels=self.lstm_channels,
            all_hidden=False,
            mode='merge',
            dropout_rate=self.lstm_dropout,
            last_conv=True,
            conv_channels=2,
        )

        if self.ext_dim > 0:
            self.extnn = ExtNN(
                in_features=self.ext_dim,
                out_height=self.len_row,
                out_width=self.len_column,
                out_channels=self.lstm_channels,
                inter_features=self.first_extnn_inter_channels,
                mode='inter',
                dropout_rate=self.first_extnn_dropout
            )
            self.time_aware_extnn = ExtNN(
                in_features=self.ext_dim,
                out_height=1,
                out_width=1,
                out_channels=1,
                inter_features=32,
                map=False,
                relu=False,
                mode='last'
            )

    def forward(self, batch):
        x = batch['X'][:, :, :, :, :self.output_dim]  # (batch_size, input_window, len_row, len_column, output_dim)
        x_ext = batch['X'][:, :, 0, 0, self.output_dim:]  # (batch_size, input_window, ext_dim)
        y_ext = batch['X'][:, -1, 0, 0, :self.output_dim:]  # (batch_size, ext_dim)

        batch_size, len_time, len_row, len_column, input_dim = x.shape
        assert len_row == self.len_row
        assert len_column == self.len_column
        assert len_time == self.input_window
        assert input_dim == self.output_dim

        x = x.contiguous().view(batch_size, len_time * input_dim, len_row, len_column).to(self.device)

        features = self.resnn(x)   # (batch_size, lstm_channels * input_window, h, w)
        if self.ext_dim > 0:
            ext = self.extnn(x_ext)    # (batch_size, lstm_channels * input_window, h, w)
            print('ext', ext.shape)
            print('features', features.shape)
            features = features + ext  # (batch_size, lstm_channels * input_window, h, w)

        # calc attention using Conv-LSTM
        # (batch_size, lstm_channels * input_window, h, w)
        hidden_list = self.conv_lstm(features)

        # (batch_size, lstm_channels * input_window, h, w)
        hidden_list_c = hidden_list[:, :self.lstm_channels * self.input_window]
        features_c = features[:, :self.lstm_channels * self.input_window]
        attention_c = self.concat_conv_c(features_c, hidden_list_c)
        phase_c = features_c * (1 + attention_c)
        pred_c = self.conv_lstm_c(phase_c)  # (batch_size, 2, h, w)

        if self.ext_dim > 0:
            time_aware = self.time_aware_extnn(y_ext)  # (batch_size, 1)
            self.time_aware_c = torch.sigmoid(time_aware)       # (batch_size, 1)
            time_aware_c = self.time_aware_c.view(-1, 1, 1, 1)  # (batch_size, 1, 1, 1)
            pred = time_aware_c * pred_c  # (batch_size, 2, h, w)
        else:
            pred = pred_c

        h = torch.tanh(pred)  # (64, 2, h, w)
        h = h.view(batch_size, 1, len_row, len_column, self.output_dim)
        return h

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
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

