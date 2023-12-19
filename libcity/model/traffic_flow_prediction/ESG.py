import torch
import numbers
import torch.nn.functional as F

from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
from torch import nn, Tensor
from torch.nn import init


class NodeFeaExtractor(nn.Module):
    def __init__(self, hidden_size_st, fc_dim):
        super(NodeFeaExtractor, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 8, 10, stride=1)
        self.conv2 = torch.nn.Conv1d(8, 16, 10, stride=1)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(hidden_size_st)
        self.fc = torch.nn.Linear(fc_dim, hidden_size_st)

    def forward(self, node_fea):
        t, n = node_fea.shape
        x = node_fea.transpose(1, 0).reshape(n, 1, -1)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.bn2(x)
        x = x.view(n, -1)
        x = self.fc(x)
        x = torch.relu(x)
        x = self.bn3(x)
        return x


class normal_conv(nn.Module):
    def __init__(self, dg_hidden_size):
        super(normal_conv, self).__init__()
        self.fc1 = nn.Linear(dg_hidden_size, 1)
        self.fc2 = nn.Linear(dg_hidden_size * 2, dg_hidden_size)

    def forward(self, y, shape):
        support = self.fc1(torch.relu(self.fc2(y))).reshape(shape)
        return support


class EvolvingGraphLearner(nn.Module):
    def __init__(self, input_size: int, dg_hidden_size: int):
        super(EvolvingGraphLearner, self).__init__()
        self.rz_gate = nn.Linear(input_size + dg_hidden_size, dg_hidden_size * 2)
        self.dg_hidden_size = dg_hidden_size
        self.h_candidate = nn.Linear(input_size + dg_hidden_size, dg_hidden_size)
        self.conv = normal_conv(dg_hidden_size)
        self.conv2 = normal_conv(dg_hidden_size)

    def forward(self, inputs: Tensor, states):
        """
        :param inputs: inputs to cal dynamic relations   [B,N,C]
        :param states: recurrent state [B, N,C]
        :return:  graph[B,N,N]       states[B,N,C]
        """
        b, n, c = states.shape
        r_z = torch.sigmoid(self.rz_gate(torch.cat([inputs, states], -1)))
        r, z = r_z.split(self.dg_hidden_size, -1)
        h_ = torch.tanh(self.h_candidate(torch.cat([inputs, r * states], -1)))
        new_state = z * states + (1 - z) * h_

        dy_sent = torch.unsqueeze(torch.relu(new_state), dim=-2).repeat(1, 1, n, 1)
        dy_revi = dy_sent.transpose(1, 2)
        y = torch.cat([dy_sent, dy_revi], dim=-1)
        support = self.conv(y, (b, n, n))
        mask = self.conv2(y, (b, n, n))
        support = support * torch.sigmoid(mask)

        return support, new_state


class Dilated_Inception(nn.Module):
    def __init__(self, cin, cout, kernel_set, dilation_factor=2):
        super(Dilated_Inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = kernel_set
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        # input [B, D, N, T]
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        """
         :param X: tensor, [B, D, N, T]
         :param A: tensor [N, N] , [B, N, N] or [T*, B, N, N]
         :return: tensor [B, D, N, T]        
        """
        # x = torch.einsum('ncwl,vw->ncvl',(x,A))
        if len(A.shape) == 2:
            a_ = 'vw'
        elif len(A.shape) == 3:
            a_ = 'bvw'
        else:
            a_ = 'tbvw'
        x = torch.einsum(f'bcwt,{a_}->bcvt', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class MixProp(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(MixProp, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj)
            out.append(h)
        ho = torch.cat(out, dim=1)  # [B, D*(1+gdep), N, T]
        ho = self.mlp(ho)  # [B, c_out, N, T]
        return ho


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class TConv(nn.Module):
    def __init__(self, residual_channels: int, conv_channels: int, kernel_set, dilation_factor: int, dropout: float):
        super(TConv, self).__init__()
        self.filter_conv = Dilated_Inception(residual_channels, conv_channels, kernel_set, dilation_factor)
        self.gate_conv = Dilated_Inception(residual_channels, conv_channels, kernel_set, dilation_factor)
        self.dropout = dropout

    def forward(self, x: Tensor):
        _filter = self.filter_conv(x)
        filter = torch.tanh(_filter)
        _gate = self.gate_conv(x)
        gate = torch.sigmoid(_gate)
        x = filter * gate
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class Evolving_GConv(nn.Module):
    def __init__(self, conv_channels: int, residual_channels: int, gcn_depth: int, st_embedding_dim: int,
                 dy_embedding_dim: int, dy_interval: int, dropout=0.3, propalpha=0.05):
        super(Evolving_GConv, self).__init__()
        self.linear_s2d = nn.Linear(st_embedding_dim, dy_embedding_dim)
        self.scale_spc_EGL = EvolvingGraphLearner(conv_channels, dy_embedding_dim)
        self.dy_interval = dy_interval

        self.gconv = MixProp(conv_channels, residual_channels, gcn_depth, dropout, propalpha)

    def forward(self, x, st_node_fea):
        b, _, n, t = x.shape
        dy_node_fea = self.linear_s2d(st_node_fea).unsqueeze(0)
        states_dy = dy_node_fea.repeat(b, 1, 1)  # [B, N, C]

        x_out = []

        for i_t in range(0, t, self.dy_interval):
            x_i = x[..., i_t:min(i_t + self.dy_interval, t)]

            input_state_i = torch.mean(x_i.transpose(1, 2), dim=-1)

            dy_graph, states_dy = self.scale_spc_EGL(input_state_i, states_dy)
            x_out.append(self.gconv(x_i, dy_graph))

        x_out = torch.cat(x_out, dim=-1)  # [B, c_out, N, T]
        return x_out


class Extractor(nn.Module):
    def __init__(self, residual_channels: int, conv_channels: int, kernel_set, dilation_factor: int, gcn_depth: int,
                 st_embedding_dim, dy_embedding_dim,
                 skip_channels: int, t_len: int, num_nodes: int, layer_norm_affline, propalpha: float, dropout: float,
                 dy_interval: int):
        super(Extractor, self).__init__()

        self.t_conv = TConv(residual_channels, conv_channels, kernel_set, dilation_factor, dropout)
        self.skip_conv = nn.Conv2d(conv_channels, skip_channels, kernel_size=(1, t_len))

        self.s_conv = Evolving_GConv(conv_channels, residual_channels, gcn_depth, st_embedding_dim, dy_embedding_dim,
                                     dy_interval, dropout, propalpha)

        self.residual_conv = nn.Conv2d(conv_channels, residual_channels, kernel_size=(1, 1))

        self.norm = LayerNorm((residual_channels, num_nodes, t_len), elementwise_affine=layer_norm_affline)

    def forward(self, x: Tensor, st_node_fea: Tensor):
        residual = x  # [B, F, N, T]
        # dilated convolution
        x = self.t_conv(x)
        # parametrized skip connection
        skip = self.skip_conv(x)
        # graph convolution
        x = self.s_conv(x, st_node_fea)
        # residual connection
        x = x + residual[:, :, :, -x.size(3):]
        x = self.norm(x)
        return x, skip


class Block(nn.ModuleList):
    def __init__(self, block_id: int, total_t_len: int, kernel_set, dilation_exp: int, n_layers: int,
                 residual_channels: int, conv_channels: int,
                 gcn_depth: int, st_embedding_dim, dy_embedding_dim, skip_channels: int, num_nodes: int,
                 layer_norm_affline, propalpha: float, dropout: float, dy_interval: int):
        super(Block, self).__init__()
        kernel_size = kernel_set[-1]
        if dilation_exp > 1:
            rf_block = int(1 + block_id * (kernel_size - 1) * (dilation_exp ** n_layers - 1) / (dilation_exp - 1))
        else:
            rf_block = block_id * n_layers * (kernel_size - 1) + 1

        dilation_factor = 1
        for i in range(1, n_layers + 1):
            if dilation_exp > 1:
                rf_size_i = int(rf_block + (kernel_size - 1) * (dilation_exp ** i - 1) / (dilation_exp - 1))
            else:
                rf_size_i = rf_block + i * (kernel_size - 1)
            t_len_i = total_t_len - rf_size_i + 1

            self.append(
                Extractor(residual_channels, conv_channels, kernel_set, dilation_factor, gcn_depth, st_embedding_dim,
                          dy_embedding_dim,
                          skip_channels, t_len_i, num_nodes, layer_norm_affline, propalpha, dropout, dy_interval[i - 1])
            )
            dilation_factor *= dilation_exp

    def forward(self, x: Tensor, st_node_fea: Tensor, skip_list):
        flag = 0
        for layer in self:
            flag += 1
            x, skip = layer(x, st_node_fea)
            skip_list.append(skip)
        return x, skip_list


class ESG(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.device = config.get('device', torch.device('cpu'))
        # section 1: data_feature
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.static_feat = torch.tensor(self.data_feature.get('static_feat'), device=self.device, dtype=torch.float32)
        self._logger = getLogger()
        self.seq_length = config.get('input_window', 12)
        self.pred_len = config.get('output_window', 12)

        self.n_blocks = config.get('n_blocks', 1)
        self.dropout = config.get('dropout', 0.3)
        self.st_embedding_dim = config.get('st_embedding_dim', 40)
        self.kernel_set = config.get('kernel_set', [2, 6])
        self.dilation_exp = config.get('dilation_exp', 1)
        self.n_layers = config.get('n_layers', 3)
        self.residual_channels = config.get('residual_channels', 32)
        self.conv_channels = config.get('conv_channels', 32)
        self.gcn_depth = config.get('gcn_depth', 2)
        self.dy_embedding_dim = config.get('dy_embedding_dim', 20)
        self.skip_channels = config.get('skip_channels', 64)
        self.layer_norm_affline = config.get('layer_norm_affline', False)
        self.propalpha = config.get('propalpha', 0.05)
        self.dy_interval = config.get('dy_interval', [1, 1, 1])  # time intervals for each layer
        self.end_channels = config.get('end_channels', 128)

        kernel_size = self.kernel_set[-1]
        if self.dilation_exp > 1:
            self.receptive_field = int(
                1 + self.n_blocks * (kernel_size - 1) * (self.dilation_exp ** self.n_layers - 1) / (
                            self.dilation_exp - 1))
        else:
            self.receptive_field = self.n_blocks * self.n_layers * (kernel_size - 1) + 1
        self.total_t_len = max(self.receptive_field, self.seq_length)

        self.start_conv = nn.Conv2d(self.feature_dim, self.residual_channels, kernel_size=(1, 1))
        self.blocks = nn.ModuleList()
        for block_id in range(self.n_blocks):
            self.blocks.append(
                Block(block_id, self.total_t_len, self.kernel_set, self.dilation_exp, self.n_layers,
                      self.residual_channels, self.conv_channels,
                      self.gcn_depth,
                      self.st_embedding_dim, self.dy_embedding_dim, self.skip_channels, self.num_nodes,
                      self.layer_norm_affline, self.propalpha,
                      self.dropout, self.dy_interval))

        self.skip0 = nn.Conv2d(self.feature_dim, self.skip_channels, kernel_size=(1, self.total_t_len), bias=True)
        self.skipE = nn.Conv2d(self.residual_channels, self.skip_channels,
                               kernel_size=(1, self.total_t_len - self.receptive_field + 1), bias=True)

        in_channels = self.skip_channels
        final_channels = self.pred_len * self.output_dim

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, self.end_channels, kernel_size=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(self.end_channels, final_channels, kernel_size=(1, 1), bias=True)
        )
        self.fc_dim = (self.static_feat.shape[0] - 18) * 16
        self.stfea_encode = NodeFeaExtractor(self.st_embedding_dim, self.fc_dim)

    def predict(self, batch):
        """
        :param input: [B, in_dim, N, n_hist]
        :return: [B, n_pred, N, out_dim]
        """

        X = batch['X']
        # (B,T,N,F) -> (B, F, N, T)
        X = X.permute(0, 3, 2, 1)
        b, _, n, t = X.shape
        assert t == self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length < self.receptive_field:
            X = F.pad(X, (self.receptive_field - self.seq_length, 0, 0, 0), mode='replicate')

        x = self.start_conv(X)

        st_node_fea = self.stfea_encode(self.static_feat)

        skip_list = [self.skip0(F.dropout(X, self.dropout, training=self.training))]
        for j in range(self.n_blocks):
            x, skip_list = self.blocks[j](x, st_node_fea, skip_list)

        skip_list.append(self.skipE(x))
        skip_list = torch.cat(skip_list, -1)  # [B, skip_channels, N, n_layers+2]

        skip_sum = torch.sum(skip_list, dim=3, keepdim=True)  # [B, skip_channels, N, 1]
        x = self.out(skip_sum)  # [B, pred_len* out_dim, N, 1] 
        x = x.reshape(b, self.pred_len, -1, n).transpose(-1, -2)  # [B, pred_len, N, out_dim]
        return x

    def calculate_loss(self, batch):
        y_true = batch['y']  # ground-truth value
        y_predicted = self.predict(batch)  # prediction results
        # denormalization the value
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true)