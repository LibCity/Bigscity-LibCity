import math
import numpy as np
from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from trafficdl.model import loss
from trafficdl.model.abstract_model import AbstractModel


class align(nn.Module):
    def __init__(self, c_in, c_out):
        super(align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)  # filter=(1,1)

    def forward(self, x):  # x: (batch_size, feature_dim(c_in), input_length, num_nodes)
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x  # return: (batch_size, c_out, input_length-1+1, num_nodes-1+1)


class temporal_conv_layer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(temporal_conv_layer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        """

        :param x: (batch_size, feature_dim(c_in), input_length, num_nodes)
        :return: (batch_size, c_out, input_length-kt+1, num_nodes)
        """
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)  # (batch_size, out_dim, input_length-kt+1, num_nodes-1+1)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)


class spatio_conv_layer(nn.Module):
    def __init__(self, ks, c, Lk):
        super(spatio_conv_layer, self).__init__()
        self.Lk = Lk
        self.theta = nn.Parameter(torch.FloatTensor(c, c, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        # Lk: (Ks, num_nodes, num_nodes)
        # x:  (batch_size, c_in, input_length, num_nodes)
        # x_c: (batch_size, c_in, input_length, Ks, num_nodes)
        # theta: (c_in, c_out, Ks)
        # g_gc: (batch_size, c_out, input_length, num_nodes)
        x_c = torch.einsum("knm,bitm->bitkn", self.Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
        return torch.relu(x_gc + x)


class st_conv_block(nn.Module):
    def __init__(self, ks, kt, n, c, p, Lk):
        super(st_conv_block, self).__init__()
        self.tconv1 = temporal_conv_layer(kt, c[0], c[1], "GLU")
        self.sconv = spatio_conv_layer(ks, c[1], Lk)
        self.tconv2 = temporal_conv_layer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

    def forward(self, x):  # x: (batch_size, feature_dim, input_length, num_nodes)
        x_t1 = self.tconv1(x)    # (batch_size, c[1], input_length-kt+1, num_nodes)
        x_s = self.sconv(x_t1)   # (batch_size, c[1], input_length-kt+1, num_nodes)
        x_t2 = self.tconv2(x_s)  # (batch_size, c[2], input_length-kt+1-kt+1, num_nodes)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)


class fully_conv_layer(nn.Module):
    def __init__(self, c, out_dim):
        super(fully_conv_layer, self).__init__()
        self.conv = nn.Conv2d(c, out_dim, 1)  # c,self.output_dim,1

    def forward(self, x):
        return self.conv(x)


class output_layer(nn.Module):
    def __init__(self, c, T, n, out_dim):
        super(output_layer, self).__init__()
        self.tconv1 = temporal_conv_layer(T, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = temporal_conv_layer(1, c, c, "sigmoid")
        self.fc = fully_conv_layer(c, out_dim)

    def forward(self, x):
        # (batch_size, input_dim(c), T, num_nodes)
        x_t1 = self.tconv1(x)
        # (batch_size, input_dim(c), 1, num_nodes)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # (batch_size, input_dim(c), 1, num_nodes)
        x_t2 = self.tconv2(x_ln)
        # (batch_size, input_dim(c), 1, num_nodes)
        return self.fc(x_t2)
        # (batch_size, 1, 1, num_nodes)


class STGCN(AbstractModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.data_feature = data_feature
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature['feature_dim']
        self.Ks = config['Ks']
        self.Kt = config['Kt']
        self.blocks = config['blocks']
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.output_dim = config.get('output_dim', 1)
        self.drop_prob = config['dropout']
        self.blocks[0][0] = self.feature_dim
        if self.input_window - 4 * (self.Kt - 1) <= 0:
            raise ValueError('Input_window must bigger than 4*(Kt-1) for 2 st_conv_block'
                             ' have 4 kt-kernel convolutional layer.')
        # 计算GCN邻接矩阵的切比雪夫估计
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')
        adj_mx = data_feature['adj_mx']  # ndarray
        adj_mx = self._scaled_laplacian(adj_mx)
        self.Lk = self._cheb_poly(adj_mx, self.Ks)
        self._logger.info('cheb_poly_Lk shape: ' + str(self.Lk.shape))
        if config['gpu']:
            self.Lk = torch.FloatTensor(self.Lk).cuda()
        else:
            self.Lk = torch.FloatTensor(self.Lk)
        # 模型结构
        self.st_conv1 = st_conv_block(self.Ks, self.Kt, self.num_nodes, self.blocks[0], self.drop_prob, self.Lk)
        self.st_conv2 = st_conv_block(self.Ks, self.Kt, self.num_nodes, self.blocks[1], self.drop_prob, self.Lk)
        self.output = output_layer(self.blocks[1][2], self.input_window - 4 * (self.Kt - 1), self.num_nodes, self.output_dim)

    def forward(self, batch):
        x = batch['X']  # (batch_size, input_length, num_nodes, feature_dim)
        x = x.permute(0, 3, 1, 2)  # (batch_size, feature_dim, input_length, num_nodes)
        x_st1 = self.st_conv1(x)   # (batch_size, c[2](64), input_length-kt+1-kt+1, num_nodes)
        x_st2 = self.st_conv2(x_st1)  # (batch_size, c[2](128), input_length-kt+1-kt+1-kt+1-kt+1, num_nodes)
        outputs = self.output(x_st2)  # (batch_size, 1(feature_dim), 1(input_length), num_nodes)
        outputs = outputs.permute(0, 2, 3, 1)  # (batch_size, 1(output_length), num_nodes, 1(feature_dim))
        return outputs

    def get_data_feature(self):
        return self.data_feature

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true, 0)

    def predict(self, batch):  # y_的feature_dim可能小于x_的！！！！！！！？？？？
        x = batch['X']  # (batch_size, input_length, num_nodes, feature_dim)
        y = batch['y']  # (batch_size, output_length, num_nodes, feature_dim)
        output_length = y.shape[1]
        y_preds = []
        x_ = x.clone()  # copy!!
        for i in range(output_length):
            batch_tmp = {'X': x_}
            y_ = self.forward(batch_tmp)  # (batch_size, 1(output_length), num_nodes, 1(feature_dim))
            y_preds.append(y_.clone())
            if y_.shape[3] < x_.shape[3]:
                y_ = torch.cat([y_, y[:, i:i+1, :, self.output_dim:]], dim=3)
            x_ = torch.cat([x_[:, 1:, :, :], y_], dim=1)
        y_preds = torch.cat(y_preds, dim=1)  # concat at time_length, y_preds.shape=y.shape
        return y_preds

    def _scaled_laplacian(self, A):
        n = A.shape[0]
        d = np.sum(A, axis=1)
        L = np.diag(d) - A
        for i in range(n):
            for j in range(n):
                if d[i] > 0 and d[j] > 0:
                    L[i, j] /= np.sqrt(d[i] * d[j])
        lam = np.linalg.eigvals(L).max().real
        return 2 * L / lam - np.eye(n)

    def _cheb_poly(self, L, Ks):
        n = L.shape[0]
        LL = [np.eye(n), L[:]]
        for i in range(2, Ks):
            LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
        return np.asarray(LL)
