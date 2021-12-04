import torch
import torch.nn as nn

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W

    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory

    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class CNN(nn.Module):
    def __init__(self, height, width, n_layers):
        super(CNN, self).__init__()
        self.height = height
        self.width = width
        self.n_layers = n_layers
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=(1, 1)))
        for i in range(1, self.n_layers):
            self.conv.append(
                nn.ReLU()
            )
            self.conv.append(
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=(1, 1))
            )
        self.relu = nn.ReLU()
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels=self.height * self.width * 16, out_channels=32, kernel_size=(1, 1)),
            nn.ReLU()
        )

    def forward(self, x):
        # (B, T, H, W, H, W)
        x = x.reshape(-1, 1, self.height, self.width)
        # (B * T * H * W, 1, H, W)
        _x = x
        x = self.conv[0](x)
        for i in range(1, self.n_layers):
            x += _x
            x = self.conv[2 * i - 1](x)
            _x = x
            x = self.conv[2 * i](x)
        x += _x
        x = self.relu(x)
        x = x.reshape(-1, self.height * self.width * 16, self.height, self.width)
        # (B * T, H * W * 16, H, W)
        x = self.embed(x)
        # (B * T, 32, H, W)
        return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc_1 = nn.Sequential(nn.Linear(in_features=29, out_features=32), nn.ReLU())
        self.fc_2 = nn.Sequential(nn.Linear(in_features=32, out_features=16), nn.ReLU())
        self.fc_3 = nn.Sequential(nn.Linear(in_features=16, out_features=8), nn.ReLU())

    def forward(self, x):
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        return x


class LSC(nn.Module):
    def __init__(self, height, width, n_layers):
        super(LSC, self).__init__()
        self.height = height
        self.width = width
        self.n_layers = n_layers
        self.O_CNN = CNN(self.height, self.width, self.n_layers)
        self.D_CNN = CNN(self.height, self.width, self.n_layers)
        self.embeder = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1)), nn.ReLU())

    def forward(self, x):
        # (B, T, H, W, H, W)
        xt = x.permute((0, 1, 4, 5, 2, 3))
        x = self.O_CNN(x)
        xt = self.D_CNN(xt)
        # (B * T, 32, H, W)
        x = torch.cat([x, xt], dim=1)
        # (B * T, 64, H, W)
        x = self.embeder(x)
        # (B * T, 32, H, W)
        return x


class TEC(nn.Module):
    def __init__(self, c_lt):
        super(TEC, self).__init__()
        self.ConvLSTM = ConvLSTM(
            input_dim=40, hidden_dim=32, kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=c_lt, kernel_size=(1, 1)),
            nn.ReLU()
        )

    def forward(self, x):
        # (B, T, 40, H, W)
        (x, _) = self.ConvLSTM(x)[1][0]
        # (B, 40, H, W)
        x = self.conv(x)
        # (B, c_lt, H, W)
        return x


class GCC(nn.Module):
    def __init__(self):
        super(GCC, self).__init__()
        self.Softmax = nn.Softmax(dim=0)

    def forward(self, x: torch.Tensor):
        # x: (B, c_lt, H, W)
        _x = x
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        s = torch.bmm(x.transpose(1, 2), x)
        s = self.Softmax(s)
        # s: (B, H * W, H * W)
        x = torch.bmm(x, s)
        # x: (B, c_lt, H, W)
        x = x.reshape(_x.shape)
        x = torch.cat([x, _x], dim=1)
        # x: (B, 2 * c_lt, H, W)
        return x


class CSTN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self.device = config.get('device', torch.device('cpu'))
        self.batch_size = config.get('batch_size')
        self.output_dim = config.get('output_dim')

        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self._scaler = self.data_feature.get('scaler')
        self.height = data_feature.get('len_row', 15)
        self.width = data_feature.get('len_column', 5)

        self.n_layers = config.get('n_layers', 3)
        self.c_lt = config.get('c_lt', 75)

        self.LSC = LSC(self.height, self.width, self.n_layers)
        self.MLP = MLP()
        self.TEC = TEC(self.c_lt)
        self.GCC = GCC()
        self.OUT = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * self.c_lt,
                out_channels=self.height * self.width,
                kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Tanh()
        )

    def forward(self, batch):
        x = batch['X'][..., 0]
        # x : (B, T, H, W, H, W)
        x = self.LSC(x)
        # x : (B * T, 32, H, W)
        x = x.reshape((self.batch_size, self.input_window, 32, self.height, self.width))
        # x : (B, T, 32, H, W)

        w = batch['W']
        # w : (B, T, F)
        w = self.MLP(w)
        # w : (B, T, 8)
        w = w.repeat(1, 1, self.height * self.width)
        w = w.reshape((self.batch_size, self.input_window, 8, self.height, self.width))
        # w : (B, T, 8, H, W)
        x = torch.cat([x, w], dim=2)
        x = self.TEC(x)
        # x : (B, c_lt, H, W)
        x = self.GCC(x)
        # x : (B, 2 * c_lt, H, W)
        x = self.OUT(x)
        # x : (B, H * W, H, W)
        x = x.reshape((self.batch_size, 1, self.height, self.width, self.height, self.width, 1))
        # x : (B, 1, H, W, H, W, 1)
        return x

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        # print('y_true', y_true.shape, y_true.device, y_true.requires_grad)
        # print('y_predicted', y_predicted.shape, y_predicted.device, y_predicted.requires_grad)
        res = loss.masked_mse_torch(y_predicted, y_true)
        return res

    def predict(self, batch):
        # 多步预测
        x = batch['X']
        w = batch['W']
        y = batch['y']
        y_preds = []
        x_ = x.clone()
        for i in range(self.output_window):
            batch_tmp = {'X': x_, 'W': w[:, i:(i + self.input_window), ...]}
            y_ = self.forward(batch_tmp)  # (batch_size, 1, len_row, len_column, output_dim)
            y_preds.append(y_.clone())
            if y_.shape[-1] < x_.shape[-1]:  # output_dim < feature_dim
                y_ = torch.cat([y_, y[:, i:i + 1, :, :, self.output_dim:]], dim=-1)
            x_ = torch.cat([x_[:, 1:, :, :, :], y_], dim=1)
        y_preds = torch.cat(y_preds, dim=1)  # (batch_size, output_length, len_row, len_column, output_dim)
        return y_preds
