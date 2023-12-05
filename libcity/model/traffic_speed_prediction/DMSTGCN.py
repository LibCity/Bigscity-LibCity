from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nwv->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class DMSTGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # section 1: data_feature
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.time_slots = self.data_feature.get('time_slots', 288)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._logger = getLogger()
        # section 2: model config
        self.output_window = config.get('output_window', 12)
        self.device = config.get('device', torch.device('cpu'))
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.3)
        self.residual_channels = config.get('residual_channels', 32)
        self.dilation_channels = config.get('dilation_channels', 32)
        self.end_channels = config.get('end_channels', 512)
        self.kernel_size = config.get('kernel_size', 2)
        self.num_blocks = config.get('num_blocks', 4)
        # 'days' in origin repo
        self.normalization = config.get('normalization', 'batch')
        self.embedding_dims = config.get('embedding_dims', 40)
        self.order = config.get('order', 2)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.normal = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.filter_convs_a = nn.ModuleList()
        self.gate_convs_a = nn.ModuleList()
        self.residual_convs_a = nn.ModuleList()
        self.skip_convs_a = nn.ModuleList()
        self.normal_a = nn.ModuleList()
        self.gconv_a = nn.ModuleList()

        self.gconv_a2p = nn.ModuleList()

        self.start_conv_a = nn.Conv2d(in_channels=1,
                                      out_channels=self.residual_channels,
                                      kernel_size=(1, 1))

        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))

        self.nodevec_p1 = nn.Parameter(torch.randn(self.time_slots, self.embedding_dims).to(self.device),
                                       requires_grad=True).to(self.device)
        self.nodevec_p2 = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dims).to(self.device),
                                       requires_grad=True).to(self.device)
        self.nodevec_p3 = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dims).to(self.device),
                                       requires_grad=True).to(self.device)
        self.nodevec_pk = nn.Parameter(
            torch.randn(self.embedding_dims, self.embedding_dims, self.embedding_dims).to(self.device),
            requires_grad=True).to(self.device)
        self.nodevec_a1 = nn.Parameter(torch.randn(self.time_slots, self.embedding_dims).to(self.device),
                                       requires_grad=True).to(self.device)
        self.nodevec_a2 = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dims).to(self.device),
                                       requires_grad=True).to(self.device)
        self.nodevec_a3 = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dims).to(self.device),
                                       requires_grad=True).to(self.device)
        self.nodevec_ak = nn.Parameter(
            torch.randn(self.embedding_dims, self.embedding_dims, self.embedding_dims).to(self.device),
            requires_grad=True).to(self.device)
        self.nodevec_a2p1 = nn.Parameter(torch.randn(self.time_slots, self.embedding_dims).to(self.device),
                                         requires_grad=True).to(self.device)
        self.nodevec_a2p2 = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dims).to(self.device),
                                         requires_grad=True).to(self.device)
        self.nodevec_a2p3 = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dims).to(self.device),
                                         requires_grad=True).to(self.device)
        self.nodevec_a2pk = nn.Parameter(
            torch.randn(self.embedding_dims, self.embedding_dims, self.embedding_dims).to(self.device),
            requires_grad=True).to(self.device)

        receptive_field = 1
        skip_channels = 8
        self.supports_len = 1
        for b in range(self.num_blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.num_layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                   out_channels=self.dilation_channels,
                                                   kernel_size=(1, self.kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=self.residual_channels,
                                                 out_channels=self.dilation_channels,
                                                 kernel_size=(1, self.kernel_size), dilation=new_dilation))

                self.residual_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))

                self.filter_convs_a.append(nn.Conv2d(in_channels=self.residual_channels,
                                                     out_channels=self.dilation_channels,
                                                     kernel_size=(1, self.kernel_size), dilation=new_dilation))

                self.gate_convs_a.append(nn.Conv1d(in_channels=self.residual_channels,
                                                   out_channels=self.dilation_channels,
                                                   kernel_size=(1, self.kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs_a.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                       out_channels=self.residual_channels,
                                                       kernel_size=(1, 1)))
                if self.normalization == "batch":
                    self.normal.append(nn.BatchNorm2d(self.residual_channels))
                    self.normal_a.append(nn.BatchNorm2d(self.residual_channels))
                elif self.normalization == "layer":
                    self.normal.append(
                        nn.LayerNorm([self.residual_channels, self.num_nodes, 13 - receptive_field - new_dilation + 1]))
                    self.normal_a.append(
                        nn.LayerNorm([self.residual_channels, self.num_nodes, 13 - receptive_field - new_dilation + 1]))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(
                    gcn(self.dilation_channels, self.residual_channels, self.dropout, support_len=self.supports_len,
                        order=self.order))
                self.gconv_a.append(
                    gcn(self.dilation_channels, self.residual_channels, self.dropout, support_len=self.supports_len,
                        order=self.order))
                self.gconv_a2p.append(
                    gcn(self.dilation_channels, self.residual_channels, self.dropout, support_len=self.supports_len,
                        order=self.order))

        self.relu = nn.ReLU(inplace=True)

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels * (12 + 10 + 9 + 7 + 6 + 4 + 3 + 1),
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.output_window,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding)
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
        adp = torch.einsum('ck, abk->abc', target_embedding, adp)
        adp = F.softmax(F.relu(adp), dim=2)
        return adp

    def predict(self, batch):
        """
        input:(B,T,N,F)-> (B, F, N, T)
        其中F包含两个特征，第0个是主特征，第1个是辅助特征。因此本模型只适合PEMSD4和8数据集
        论文中分别使用speed/flow作为主/辅助特征。使用其他特征，需要修改raw_data/dataset_name/config.json文件中的"data_col"属性。
        """
        inputs = batch['X']
        inputs = inputs.permute(0, 3, 2, 1)
        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            xo = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            xo = inputs
        # xo[:,[0]] means primary feature
        x = self.start_conv(xo[:, [0]])
        # xo[:,[1]] means auxiliary feature, witch can be set in the raw_data/dataset_name/config.json file
        x_a = self.start_conv_a(xo[:, [1]])
        idx = batch['idx']
        skip = 0

        # dynamic graph construction
        adp = self.dgconstruct(self.nodevec_p1[idx], self.nodevec_p2, self.nodevec_p3, self.nodevec_pk)
        adp_a = self.dgconstruct(self.nodevec_a1[idx], self.nodevec_a2, self.nodevec_a3, self.nodevec_ak)
        adp_a2p = self.dgconstruct(self.nodevec_a2p1[idx], self.nodevec_a2p2, self.nodevec_a2p3, self.nodevec_a2pk)

        new_supports = [adp]
        new_supports_a = [adp_a]
        new_supports_a2p = [adp_a2p]

        for i in range(self.num_blocks * self.num_layers):
            # tcn for primary part
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # tcn for auxiliary part
            residual_a = x_a
            filter_a = self.filter_convs_a[i](residual_a)
            filter_a = torch.tanh(filter_a)
            gate_a = self.gate_convs_a[i](residual_a)
            gate_a = torch.sigmoid(gate_a)
            x_a = filter_a * gate_a

            # skip connection
            s = x
            s = self.skip_convs[i](s)
            if isinstance(skip, int):  # B F N T
                skip = s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]).contiguous()
            else:
                skip = torch.cat([s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]), skip], dim=1).contiguous()

            # dynamic graph convolutions
            x = self.gconv[i](x, new_supports)
            x_a = self.gconv_a[i](x_a, new_supports_a)

            # multi-faceted fusion module
            x_a2p = self.gconv_a2p[i](x_a, new_supports_a2p)
            x = x_a2p + x

            # residual and normalization
            x_a = x_a + residual_a[:, :, :, -x_a.size(3):]
            x = x + residual[:, :, :, -x.size(3):]
            x = self.normal[i](x)
            x_a = self.normal_a[i](x_a)

        # output layer
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true)
