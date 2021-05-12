from logging import getLogger
import torch
import torch.nn as nn
from trafficdl.model import loss
from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel


class SpatialViewConv(nn.Module):
    def __init__(self, inp_channel, oup_channel, kernel_size, stride=1, padding=0):
        super(SpatialViewConv, self).__init__()
        self.inp_channel = inp_channel
        self.oup_channel = oup_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels=inp_channel, out_channels=oup_channel,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch = nn.BatchNorm2d(oup_channel)
        self.relu = nn.ReLU()

    def forward(self, inp):
        return self.relu(self.batch(self.conv(inp)))


class TemporalView(nn.Module):
    def __init__(self, fc_oup_dim, lstm_oup_dim, output_dim):
        super(TemporalView, self).__init__()
        self.lstm = nn.LSTM(fc_oup_dim, lstm_oup_dim)
        self.fc = nn.Linear(in_features=lstm_oup_dim, out_features=output_dim)

    def forward(self, inp):
        # inp = [T, B, fc_oup_dim]
        lstm_res, (h, c) = self.lstm(inp)
        # lstm_res = [T, B, lstm_oup_dim]
        # h/c = [1, B, lstm_oup_dim]
        return self.fc(h[0])  # [B, output_dim]


class DMVSTNet(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._scaler = self.data_feature.get('scaler')  # 用于数据归一化
        self.adj_mx = self.data_feature.get('adj_mx', 1)  # 邻接矩阵
        self.num_nodes = self.data_feature.get('num_nodes', 1)  # 网格个数
        self.feature_dim = self.data_feature.get('feature_dim', 1)  # 输入维度
        self.output_dim = self.data_feature.get('output_dim', 1)  # 输出维度
        self.len_row = self.data_feature.get('len_row', 1)  # 网格行数
        self.len_column = self.data_feature.get('len_column', 1)  # 网格列数
        self._logger = getLogger()

        self.device = config.get('device', torch.device('cpu'))
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.local_image_size = config.get('local_image_size', 5)
        self.padding_size = self.local_image_size // 2
        self.cnn_hidden_dim_first = config.get('cnn_hidden_dim_first', 32)
        self.fc_oup_dim = config.get('fc_oup_dim', 64)
        self.lstm_oup_dim = config.get('lstm_oup_dim', 512)

        self.padding = nn.ZeroPad2d((self.padding_size, self.padding_size, self.padding_size, self.padding_size))

        # 三层Local CNN
        self.local_conv1 = SpatialViewConv(inp_channel=self.feature_dim, oup_channel=self.cnn_hidden_dim_first,
                                           kernel_size=3, stride=1, padding=1)
        self.local_conv2 = SpatialViewConv(inp_channel=self.cnn_hidden_dim_first, oup_channel=self.cnn_hidden_dim_first,
                                           kernel_size=3, stride=1, padding=1)
        self.local_conv3 = SpatialViewConv(inp_channel=self.cnn_hidden_dim_first, oup_channel=self.cnn_hidden_dim_first,
                                           kernel_size=3, stride=1, padding=1)

        # 全连接降维
        self.fc1 = nn.Linear(in_features=self.cnn_hidden_dim_first * self.local_image_size * self.local_image_size,
                             out_features=self.fc_oup_dim)

        # TemporalView
        self.temporalLayers = TemporalView(self.fc_oup_dim, self.lstm_oup_dim, self.output_dim)

    def spatial_forward(self, grid_batch):
        x1 = self.local_conv1(grid_batch)
        x2 = self.local_conv2(x1)
        x3 = self.local_conv3(x2)
        x4 = self.fc1(torch.flatten(x3, start_dim=1))
        return x4

    def forward(self, batch):
        # input转换为卷积运算的格式 (B, input_window, len_row, len_col, feature_dim)
        x = batch['X'].permute(0, 1, 4, 2, 3)  # (B, input_window, feature_dim, len_row, len_col)
        batch_size = x.shape[0]
        x = x.reshape((batch_size * self.input_window, self.feature_dim, self.len_row, self.len_column))

        # 对输入进行0填充
        x_padding = self.padding(x)
        # 构造输出
        oup = torch.zeros((batch_size, 1, self.len_row, self.len_column, self.output_dim)).to(self.device)
        # 对每个grid进行预测
        for i in range(self.padding_size, self.len_row - self.padding_size):
            for j in range(self.padding_size, self.len_column - self.padding_size):
                spatial_res = self.spatial_forward(
                    x_padding[:, :, i - self.padding_size:i + self.padding_size + 1,
                              j - self.padding_size: j + self.padding_size + 1])
                # print('spatial_res', spatial_res.shape)  # (T*B, fc_oup_dim)
                seq_res = spatial_res.reshape((self.input_window, batch_size, self.fc_oup_dim))
                # print('seq_res', seq_res.shape)  # (T, B, fc_oup_dim)
                temporal_res = self.temporalLayers(seq_res)
                # print('temporal_res', temporal_res.shape)  # (B, output_dim)
                oup[:, :, i, j, :] = temporal_res.reshape(batch_size, 1, self.output_dim)
        return oup

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
