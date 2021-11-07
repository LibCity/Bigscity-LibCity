import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList
import dgl
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class STMGAT(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.dropout = config.get('dropout', 0.3)
        self.blocks = config.get('blocks', 4)
        self.layers = config.get('layers', 2)
        self.run_gconv = config.get('run_gconv', True)
        self.residual_channels = config.get('residual_channels', 40)
        self.dilation_channels = config.get('dilation_channels', 40)
        self.skip_channels = config.get('skip_channels', 320)
        self.end_channels = config.get('end_channels', 640)
        self.kernel_size = config.get('kernel_size', 2)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.heads = config.get('heads', 8)
        self.feat_drop = config.get('feat_drop', 0.6)
        self.attn_drop = config.get('attn_drop', 0.6)
        self.negative_slope = config.get('negative_slope', 0.2)

        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.ext_dim = self.data_feature.get('ext_dim', 1)
        self._scaler = self.data_feature.get('scaler')
        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))

        self.g = self._get_adj()

        self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))
        self.cat_feature_conv = nn.Conv2d(in_channels=self.feature_dim,
                                          out_channels=self.residual_channels,
                                          kernel_size=(1, 1))

        receptive_field = self.output_dim
        depth = list(range(self.blocks * self.layers))
        # 1x1 convolution for residual and skip connections (slightly different see docstring)
        self.residual_convs = ModuleList([Conv1d(self.dilation_channels,
                                                 self.residual_channels, (1, 1)) for _ in depth])
        self.skip_convs = ModuleList([Conv1d(self.dilation_channels,
                                             self.skip_channels, (1, 1)) for _ in depth])
        self.bn = ModuleList([BatchNorm2d(self.residual_channels) for _ in depth])

        self.gat_layers = nn.ModuleList()
        self.gat_layers1 = nn.ModuleList()
        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()

        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            D = 1  # dilation
            for i in range(self.layers):
                # dilated convolutions
                self.filter_convs.append(Conv2d(self.residual_channels,
                                                self.dilation_channels, (1, self.kernel_size), dilation=D))
                self.gate_convs.append(Conv1d(self.residual_channels,
                                              self.dilation_channels, (1, self.kernel_size), dilation=D))
                # batch, channel, height, width
                # N,C,H,W
                # d = (d - kennel_size + 2 * padding) / stride + 1
                # H_out = [H_in + 2*padding[0] - dilation[0]*(kernal_size[0]-1)-1]/stride[0] + 1
                # W_out = [W_in + 2*padding[1] - dilation[1]*(kernal_size[1]-1)-1]/stride[1] + 1
                D *= 2
                receptive_field += additional_scope
                additional_scope *= 2
        self.receptive_field = receptive_field
        self._logger.info('receptive_field: ' + str(self.receptive_field))

        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            for i in range(self.layers):
                receptive_field -= additional_scope
                additional_scope *= 2
                self.gat_layers.append(GATConv(
                    self.dilation_channels * receptive_field,
                    self.dilation_channels * receptive_field,
                    self.heads, self.feat_drop, self.attn_drop, self.negative_slope,
                    residual=False, activation=F.elu))
                self.gat_layers1.append(GATConv(
                    self.dilation_channels * receptive_field,
                    self.dilation_channels * receptive_field,
                    self.heads, self.feat_drop, self.attn_drop, self.negative_slope,
                    residual=False, activation=F.elu))

        self.end_conv_1 = Conv2d(self.skip_channels, self.end_channels, (1, 1), bias=True)
        self.end_conv_2 = Conv2d(self.end_channels, self.output_window, (1, 1), bias=True)

    def _get_adj(self):
        adj_mx = self.data_feature.get('adj_mx', 1)
        edge_list = []
        for i in range(adj_mx.shape[0]):
            for j in range(adj_mx.shape[1]):
                if adj_mx[i][j] > 0:  # link
                    edge_list.append((i, j, adj_mx[i][j]))
        src, dst, cost = tuple(zip(*edge_list))
        g = dgl.DGLGraph()
        g.add_nodes(self.num_nodes)
        g.add_edges(src, dst)
        g = dgl.add_self_loop(g)
        g.edges[src, dst].data['w'] = torch.Tensor(cost)
        g = g.to(self.device)
        return g

    def forward(self, batch):
        x = batch['X']  # (batch_size, input_window, num_nodes, feature_dim)
        x = x.permute(0, 3, 2, 1)  # (batch_size, feature_dim, num_nodes, input_window)
        in_len = x.size(3)
        assert self.input_window == in_len
        if in_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - in_len, 0, 0, 0))
        # (batch_size, feature_dim, num_nodes, receptive_field)
        x1 = self.start_conv(x)
        x2 = F.leaky_relu(self.cat_feature_conv(x))
        # (batch_size, residual_channels, num_nodes, receptive_field)
        x = x1 + x2
        skip = 0

        # STGAT layers
        for i in range(self.blocks * self.layers):
            residual = x
            # dilated convolution
            filter = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter * gate
            # print('x', x.shape)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            # parametrized skip connection
            s = self.skip_convs[i](x)

            if i > 0:
                skip = skip[:, :, :, -s.size(3):]
            else:
                skip = 0
            skip = s + skip
            if i == (self.blocks * self.layers - 1):  # last X getting ignored anyway
                break

            # graph conv and mix
            if self.run_gconv:
                [batch_size, fea_size, num_of_vertices, step_size] = x.size()  # [64, 40, 207, 12]
                batched_g = dgl.batch(batch_size * [self.g])
                h = x.permute(0, 2, 1, 3).reshape(batch_size*num_of_vertices, fea_size*step_size)  # [64*207, 40*12]
                h = self.gat_layers[i](batched_g, h).mean(1)  # [64*207, 40*12]
                h = self.gat_layers1[i](batched_g, h).mean(1)  # [64*207, 40*12]
                gc = h.reshape(batch_size, num_of_vertices, fea_size, -1)  # [64, 207, 40, 12]
                graph_out = gc.permute(0, 2, 1, 3)  # [64, 40, 207, 12]
                x = x + graph_out
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]  # [64, 40, 207, 12]
            x = self.bn[i](x)
        # (batch_size, skip_channels, num_nodes, self.output_dim)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        # (batch_size, output_window, num_nodes, self.output_dim)
        return x

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        # print('y_true', y_true.shape)
        # print('y_predicted', y_predicted.shape)
        y_true = self._scaler.inverse_transform(y_true)
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true)
