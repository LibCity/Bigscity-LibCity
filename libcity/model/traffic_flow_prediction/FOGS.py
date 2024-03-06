from logging import getLogger

import numpy as np
import torch
from torch import nn

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class gcn_operation(nn.Module):
    def __init__(self, adj, in_dim, out_dim, num_vertices, activation='GLU'):
        """
        图卷积模块
        :param adj: 邻接图
        :param in_dim: 输入维度
        :param out_dim: 输出维度
        :param num_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(gcn_operation, self).__init__()
        self.adj = adj
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_vertices = num_vertices
        self.activation = activation

        assert self.activation in {'GLU', 'relu'}

        if self.activation == 'GLU':
            self.FC = nn.Linear(self.in_dim, 2 * self.out_dim, bias=True)
        else:
            self.FC = nn.Linear(self.in_dim, self.out_dim, bias=True)

    def forward(self, x, mask=None):
        """
        :param x: (3*N, B, Cin)
        :param mask:(3*N, 3*N)
        :return: (3*N, B, Cout)
        """
        adj = self.adj
        if mask is not None:
            adj = adj.to(mask.device) * mask

        x = torch.einsum('nm, mbc->nbc', adj.to(x.device), x)  # 3*N, B, Cin

        if self.activation == 'GLU':
            lhs_rhs = self.FC(x)  # 3*N, B, 2*Cout
            lhs, rhs = torch.split(lhs_rhs, self.out_dim, dim=-1)  # 3*N, B, Cout

            out = lhs * torch.sigmoid(rhs)
            del lhs, rhs, lhs_rhs

            return out

        elif self.activation == 'relu':
            return torch.relu(self.FC(x))  # 3*N, B, Cout


class STSGCM(nn.Module):
    def __init__(self, adj, in_dim, out_dims, num_of_vertices, activation='GLU'):
        """
        :param adj: 邻接矩阵
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(STSGCM, self).__init__()
        self.adj = adj
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation

        self.gcn_operations = nn.ModuleList()

        self.gcn_operations.append(
            gcn_operation(
                adj=self.adj,
                in_dim=self.in_dim,
                out_dim=self.out_dims[0],
                num_vertices=self.num_of_vertices,
                activation=self.activation
            )
        )

        for i in range(1, len(self.out_dims)):
            self.gcn_operations.append(
                gcn_operation(
                    adj=self.adj,
                    in_dim=self.out_dims[i - 1],
                    out_dim=self.out_dims[i],
                    num_vertices=self.num_of_vertices,
                    activation=self.activation
                )
            )

    def forward(self, x, mask=None):
        """
        :param x: (3N, B, Cin)
        :param mask: (3N, 3N)
        :return: (N, B, Cout)
        """
        need_concat = []

        for i in range(len(self.out_dims)):
            x = self.gcn_operations[i](x, mask)
            need_concat.append(x)

        # shape of each element is (1, N, B, Cout)
        need_concat = [
            torch.unsqueeze(
                h[self.num_of_vertices: 2 * self.num_of_vertices], dim=0
            ) for h in need_concat
        ]

        out = torch.max(torch.cat(need_concat, dim=0), dim=0).values  # (N, B, Cout)

        del need_concat

        return out


class STSGCL(nn.Module):
    def __init__(self,
                 adj,
                 history,
                 num_of_vertices,
                 in_dim,
                 out_dims,
                 strides=3,
                 activation='GLU',
                 temporal_emb=True,
                 spatial_emb=True):
        """
        :param adj: 邻接矩阵
        :param history: 输入时间步长
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param strides: 滑动窗口步长，local时空图使用几个时间步构建的，默认为3
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        :param temporal_emb: 加入时间位置嵌入向量
        :param spatial_emb: 加入空间位置嵌入向量
        """
        super(STSGCL, self).__init__()
        self.adj = adj
        self.strides = strides
        self.history = history
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        # STFGNN的扩张卷积
        self.dilation_conv_1 = nn.Conv1d(self.in_dim, self.in_dim, kernel_size=(1, 2), stride=(1, 1), dilation=(1, 3))
        self.dilation_conv_2 = nn.Conv1d(self.in_dim, self.in_dim, kernel_size=(1, 2), stride=(1, 1), dilation=(1, 3))
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb

        self.STSGCMS = nn.ModuleList()
        for i in range(self.history - self.strides + 1):
            self.STSGCMS.append(
                STSGCM(
                    adj=self.adj,
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation
                )
            )

        if self.temporal_emb:
            self.temporal_embedding = nn.Parameter(torch.FloatTensor(1, self.history, 1, self.in_dim))
            # 1, T, 1, Cin

        if self.spatial_emb:
            self.spatial_embedding = nn.Parameter(torch.FloatTensor(1, 1, self.num_of_vertices, self.in_dim))
            # 1, 1, N, Cin

        self.reset()

    def reset(self):
        if self.temporal_emb:
            nn.init.xavier_normal_(self.temporal_embedding, gain=0.0003)

        if self.spatial_emb:
            nn.init.xavier_normal_(self.spatial_embedding, gain=0.0003)

    def forward(self, x, mask=None):
        """
        :param x: B, T, N, Cin
        :param mask: (N, N)
        :return: B, T-2, N, Cout
        """
        if self.temporal_emb:
            x = x + self.temporal_embedding

        if self.spatial_emb:
            x = x + self.spatial_embedding

        # STFGNN版本
        x_temp = x.permute(0, 3, 2, 1)  # (B, Cin, N, T)
        x_left = torch.tanh(self.dilation_conv_1(x_temp))
        x_right = torch.sigmoid(self.dilation_conv_2(x_temp))

        x_time_axis = x_left * x_right

        x_res = x_time_axis.permute((0, 3, 2, 1))  # (B, T-3, N, C)

        need_concat = []
        batch_size = x.shape[0]

        for i in range(self.history - self.strides + 1):
            t = x[:, i: i + self.strides, :, :]  # (B, 3, N, Cin)

            t = torch.reshape(t, shape=[batch_size, self.strides * self.num_of_vertices, self.in_dim])
            # (B, 3*N, Cin)

            t = self.STSGCMS[i](t.permute(1, 0, 2), mask)  # (3*N, B, Cin) -> (N, B, Cout)

            t = torch.unsqueeze(t.permute(1, 0, 2), dim=1)  # (N, B, Cout) -> (B, N, Cout) ->(B, 1, N, Cout)

            need_concat.append(t)

        out = torch.cat(need_concat, dim=1)  # (B, T-2, N, Cout)

        del need_concat, batch_size

        layer_out = out + x_res
        return layer_out
        # return out


class output_layer(nn.Module):
    def __init__(self, num_of_vertices, history, in_dim,
                 hidden_dim=128, horizon=12):
        """
        预测层，注意在作者的实验中是对每一个预测时间step做处理的，也即他会令horizon=1
        :param num_of_vertices:节点数
        :param history:输入时间步长
        :param in_dim: 输入维度
        :param hidden_dim:中间层维度
        :param horizon:预测时间步长
        """
        super(output_layer, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.history = history
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        self.FC1 = nn.Linear(self.in_dim * self.history, self.hidden_dim, bias=True)

        self.FC2 = nn.Linear(self.hidden_dim, self.horizon, bias=True)

    def forward(self, x):
        """
        :param x: (B, Tin, N, Cin)
        :return: (B, Tout, N)
        """
        batch_size = x.shape[0]

        x = x.permute(0, 2, 1, 3)  # B, N, Tin, Cin

        out1 = torch.relu(self.FC1(x.reshape(batch_size, self.num_of_vertices, -1)))
        # (B, N, Tin, Cin) -> (B, N, Tin * Cin) -> (B, N, hidden)

        out2 = self.FC2(out1)  # (B, N, hidden) -> (B, N, horizon)

        del out1, batch_size

        return out2.permute(0, 2, 1)  # B, horizon, N


class FOGS(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        # data feature
        self._scaler = self.data_feature.get('scaler')
        self.adj_mx = self.data_feature.get('adj_mx')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self._logger = getLogger()

        # model config
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.hidden_dims = config.get('hidden_dims', [[64, 64, 64], [64, 64, 64], [64, 64, 64]])
        self.first_layer_embedding_size = config.get('first_layer_embedding_size', 64)
        self.out_layer_dim = config.get('out_layer_dim', 128)
        self.activation = config.get('activation', 'GLU')
        self.use_mask = config.get('use_mask', True)
        self.temporal_emb = config.get('temporal_emb', True)
        self.spatial_emb = config.get('spatial_emb', True)
        self.strides = config.get('strides', 3)
        self.use_trend = config.get('use_trend', True)
        self.trend_embedding = config.get('trend_embedding', False)
        if self.trend_embedding:
            self.trend_bias_embeddings = nn.Embedding(288, self.num_nodes * self.output_window)

        self.device = config.get('device', torch.device('cpu'))

        self.adj = self.adj_mx
        self.num_of_vertices = self.num_nodes
        self.horizon = self.output_window

        self.default_loss_function = torch.nn.SmoothL1Loss()
        history = self.input_window
        out_layer_dim = self.out_layer_dim
        first_layer_embedding_size = self.first_layer_embedding_size
        in_dim = self.feature_dim
        self.First_FC = nn.Linear(in_dim, first_layer_embedding_size, bias=True)
        self.STSGCLS = nn.ModuleList()
        self.STSGCLS.append(
            STSGCL(
                adj=self.adj,
                history=history,
                num_of_vertices=self.num_of_vertices,
                in_dim=first_layer_embedding_size,
                out_dims=self.hidden_dims[0],
                strides=self.strides,
                activation=self.activation,
                temporal_emb=self.temporal_emb,
                spatial_emb=self.spatial_emb
            )
        )

        in_dim = self.hidden_dims[0][-1]
        history -= (self.strides - 1)

        for idx, hidden_list in enumerate(self.hidden_dims):
            if idx == 0:
                continue
            self.STSGCLS.append(
                STSGCL(
                    adj=self.adj,
                    history=history,
                    num_of_vertices=self.num_of_vertices,
                    in_dim=in_dim,
                    out_dims=hidden_list,
                    strides=self.strides,
                    activation=self.activation,
                    temporal_emb=self.temporal_emb,
                    spatial_emb=self.spatial_emb
                )
            )

            history -= (self.strides - 1)
            in_dim = hidden_list[-1]

        self.predictLayer = nn.ModuleList()
        for t in range(self.horizon):
            self.predictLayer.append(
                output_layer(
                    num_of_vertices=self.num_of_vertices,
                    history=history,
                    in_dim=in_dim,
                    hidden_dim=out_layer_dim,
                    horizon=1
                )
            )

        if self.use_mask:
            mask = torch.zeros_like(self.adj)
            mask[self.adj != 0] = self.adj[self.adj != 0]
            self.mask = nn.Parameter(mask)
        else:
            self.mask = None

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.0003)
            else:
                nn.init.uniform_(p)

    def forward(self, x):
        """
        :param x: B, Tin, N, Cin)
        :return: B, Tout, N
        """

        x = torch.relu(self.First_FC(x))  # B, Tin, N, Cin  x原本是(B,T,N,1),经过全连接层变成(B,T,N,64)

        for model in self.STSGCLS:
            x = model(x, self.mask)
        # (B, T - 8, N, Cout)

        need_concat = []
        for i in range(self.horizon):
            out_step = self.predictLayer[i](x)  # (B, 1, N)
            need_concat.append(out_step)

        out = torch.cat(need_concat, dim=1)  # B, Tout, N

        del need_concat

        return out

    def predict(self, batch):
        return self.forward(batch['X'])

    def _compute_embedding_loss(self, x, y_true, y_pred, bias, null_val=np.nan):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim) tensor
        :param y_pred: shape (batch_size, seq_len, num_sensor * input_dim) tensor
        :param y_true: shape (batch_size, horizon, num_sensor * input_dim)
        :param bias: shape (batch_size, horizon, num_sensor * input_dim)
        """
        x = x.squeeze()  # (B, T, N)
        x_truth = self.scaler.inverse_transform(x)  # 把x逆标准化转换为真实流量值
        # (B, T, N) -> (T, B, N)
        x_truth = x_truth.permute(1, 0, 2)
        y_true = y_true.permute(1, 0, 2)
        y_pred = y_pred.permute(1, 0, 2)

        labels = (1 + y_true) * x_truth[-1] - (1 + y_pred) * x_truth[-1]  # (T, B, N)
        bias = bias.to(self.device)
        labels = labels.to(self.device)
        # (T, B, N) -> (B, T, N)
        labels = labels.permute(1, 0, 2)

        return loss.masked_mae_torch(bias, labels, null_val)

    def calculate_loss(self, batch):
        input = batch['X']
        real_val = batch['y'][:, :, :, 0]
        realy_slot = batch['y_slot']
        output = self.predict(batch)
        if self.trend_embedding:
            trend_time_bias = self.trend_bias_embeddings(realy_slot[:, 0]).to(self.device)  # (B, N * T)
            # (B, N, T) -> (B, T, N)
            trend_time_bias = torch.reshape(trend_time_bias, (-1, self.num_nodes, self.horizon)).permute(0, 2, 1)
            return loss.masked_mae_torch(output, real_val) + self._compute_embedding_loss(input, real_val, output,
                                                                                          trend_time_bias)
        else:
            if self.use_trend:
                return loss.masked_mae_torch(output, real_val)
            else:
                output = self.scaler.inverse_transform(output)
                return self.default_loss_function(output, real_val)
