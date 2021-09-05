from logging import getLogger
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse.linalg import eigs
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def scaled_laplacian(w):
    w = w.astype(float)
    n = np.shape(w)[0]
    d = []
    # simple graph, W_{i,i} = 0
    lap = -w
    # get degree matrix d and Laplacian matrix L
    for i in range(n):
        d.append(np.sum(w[i, :]))
        lap[i, i] = d[i]
    # symmetric normalized Laplacian L
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                lap[i, j] = lap[i, j] / np.sqrt(d[i] * d[j])
    lambda_max = eigs(lap, k=1, which='LR')[0][0].real
    # lambda_max \approx 2.0
    # we can replace this sentence by setting lambda_max = 2
    return 2 * lap / lambda_max - np.identity(n)


def cheb_poly(lap, ks):
    n = lap.shape[0]
    lap_list = [np.eye(n), lap[:]]
    for i in range(2, ks):
        lap_list.append(np.matmul(2 * lap, lap_list[-1]) - lap_list[-2])
    # lap_list: (Ks, n*n), Lk (n, Ks*n)
    return np.concatenate(lap_list, axis=-1)


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1, stride=1, padding=0)  # filter=(1,1)

    def forward(self, x):  # x: (batch_size, feature_dim(c_in), input_length, num_nodes)
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x  # return: (batch_size, c_out, input_length-1+1, num_nodes-1+1)


class ConvST(nn.Module):
    def __init__(self, supports, kt, ks, dim_in, dim_out, device):
        super(ConvST, self).__init__()
        self.supports = supports
        self.kt = kt
        self.ks = ks
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.device = device
        self.align = Align(c_in=dim_in, c_out=dim_out)
        self.weights = nn.Parameter(torch.FloatTensor(
            2 * self.dim_out, self.ks * self.kt * self.dim_in).to(self.device))
        self.biases = nn.Parameter(torch.zeros(2 * self.dim_out).to(self.device))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x):
        """

        Args:
            x: torch.tensor, shape=[B, dim_in, T, num_nodes]

        Returns:
            torch.tensor: shape=[B, dim_out, T, num_nodes]

        """
        batch_size, len_time, num_nodes = x.shape[0], x.shape[2], x.shape[3]
        assert x.shape[1] == self.dim_in
        res_input = self.align(x)  # (B, dim_out, T, num_nodes)
        padding = torch.zeros(batch_size, self.dim_in, self.kt - 1, num_nodes).to(self.device)
        # extract spatial-temporal relationships at the same time
        x = torch.cat((x, padding), dim=2)
        # inputs.shape = [B, dim_in, len_time+kt-1, N]
        x = torch.stack([x[:, :, i:i + self.kt, :] for i in range(0, len_time)], dim=2)
        # inputs.shape = [B, dim_in, len_time, kt, N]
        x = torch.reshape(x, (-1, num_nodes, self.kt * self.dim_in))
        # inputs.shape = [B*len_time, N, kt*dim_in]
        conv_out = self.graph_conv(x, self.supports, self.kt * self.dim_in, 2 * self.dim_out)
        # conv_out: [B*len_time, N, 2*dim_out]
        conv_out = torch.reshape(conv_out, [-1, 2 * self.dim_out, len_time, num_nodes])
        # conv_out: [B, 2*dim_out, len_time, N]
        out = (conv_out[:, :self.dim_out, :, :] + res_input) * torch.sigmoid(conv_out[:, self.dim_out:, :, :])
        return out  # [B, dim_out, len_time, N]

    def graph_conv(self, inputs, supports, dim_in, dim_out):
        """

        Args:
            inputs: a tensor of shape [batch, num_nodes, dim_in]
            supports: [num_nodes, num_nodes*ks], calculate the chebyshev polynomials in advance to save time
            dim_in:
            dim_out:

        Returns:
            torch.tensor: shape = [batch, num_nodes, dim_out]

        """
        num_nodes = inputs.shape[1]
        assert num_nodes == supports.shape[0]
        assert dim_in == inputs.shape[2]
        # [batch, num_nodes, dim_in] -> [batch, dim_in, num_nodes] -> [batch * dim_in, num_nodes]
        x_new = torch.reshape(inputs.permute(0, 2, 1), (-1, num_nodes))
        # [batch * dim_in, num_nodes] * [num_nodes, num_nodes*ks]
        #       -> [batch * dim_in, num_nodes*ks] -> [batch, dim_in, ks, num_nodes]
        x_new = torch.reshape(torch.matmul(x_new, supports), (-1, dim_in, self.ks, num_nodes))
        # [batch, dim_in, ks, num_nodes] -> [batch, num_nodes, dim_in, ks]
        x_new = x_new.permute(0, 3, 1, 2)
        # [batch, num_nodes, dim_in, ks] -> [batch*num_nodes, dim_in*ks]
        x_new = torch.reshape(x_new, (-1, self.ks * dim_in))
        outputs = F.linear(x_new, self.weights, self.biases)  # [batch*num_nodes, dim_out]
        outputs = torch.reshape(outputs, [-1, num_nodes, dim_out])  # [batch, num_nodes, dim_out]
        return outputs


class AttentionT(nn.Module):
    def __init__(self, device, len_time, num_nodes, d_out, ext_dim):
        super(AttentionT, self).__init__()
        self.device = device
        self.len_time = len_time
        self.num_nodes = num_nodes
        self.d_out = d_out
        self.ext_dim = ext_dim
        self.weight1 = nn.Parameter(torch.FloatTensor(self.len_time, self.num_nodes * self.d_out, 1).to(self.device))
        self.weight2 = nn.Parameter(torch.FloatTensor(self.ext_dim, self.len_time).to(self.device))
        self.bias = nn.Parameter(torch.zeros(self.len_time).to(self.device))
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)

    def forward(self, query, x):
        # query  # [B, ext_dim]
        # temporal attention: x.shape = [B, d_out, T, N]
        x_in = torch.reshape(x, (-1, self.num_nodes * self.d_out, self.len_time))
        # x_in.shape = [B, N*d_out, T]
        x = x_in.permute(2, 0, 1)
        # x.shape = [T, B, N*d_out]

        score = torch.reshape(torch.matmul(x, self.weight1), (-1, self.len_time)) + self.bias
        score = score + torch.matmul(query, self.weight2)
        score = torch.softmax(torch.tanh(score), dim=1)
        # score.shape = [B, T]
        x = torch.matmul(x_in, torch.unsqueeze(score, dim=-1))
        # x.shape = [B, N*d_out, 1]
        x = x.permute(0, 2, 1).reshape((-1, 1, self.num_nodes, self.d_out)).permute(0, 3, 1, 2)
        # x = torch.reshape(x, (-1, d_out, 1, N))
        # x.shape = [B, d_out, 1, N]
        return x


class AttentionC(nn.Module):
    def __init__(self, device, num_nodes, d_out, ext_dim):
        super(AttentionC, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.d_out = d_out
        self.ext_dim = ext_dim
        self.weight1 = nn.Parameter(torch.FloatTensor(self.d_out, self.num_nodes, 1).to(self.device))
        self.weight2 = nn.Parameter(torch.FloatTensor(self.ext_dim, self.d_out).to(self.device))
        self.bias = nn.Parameter(torch.zeros(self.d_out).to(self.device))
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)

    def forward(self, query, x):
        # query  # [B, ext_dim]
        # channel attention: x.shape = [B, d_out, 1, N]
        x_in = torch.reshape(x, (-1, self.num_nodes, self.d_out))
        # x_in.shape = [B, N, d_out]
        x = x_in.permute(2, 0, 1)
        # x.shape = [d_out, B, N]
        score = torch.reshape(torch.matmul(x, self.weight1), (-1, self.d_out)) + self.bias
        score = score + torch.matmul(query, self.weight2)
        score = torch.softmax(torch.tanh(score), dim=1)
        # score.shape = [B, d_out]
        x = torch.matmul(x_in, torch.unsqueeze(score, dim=-1)).permute(0, 2, 1)
        # x.shape = [B, 1, N] (1->dim)
        x = torch.unsqueeze(x, dim=2)  # [B, 1(dim), 1(T), N]
        return x


class STG2Seq(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.adj_mx = self.data_feature.get('adj_mx')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 2)
        self.output_dim = self.data_feature.get('output_dim', 2)
        self.ext_dim = self.data_feature.get('ext_dim', 1)
        # self.len_row = self.data_feature.get('len_row', 32)
        # self.len_column = self.data_feature.get('len_column', 32)
        self._scaler = self.data_feature.get('scaler')
        self._logger = getLogger()

        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.window = config.get('window', 3)
        self.dim_out = config.get('dim_out', 32)
        self.ks = config.get('ks', 3)
        self.device = config.get('device', torch.device('cpu'))
        self.supports = torch.tensor(cheb_poly(scaled_laplacian(self.adj_mx), self.ks),
                                     dtype=torch.float32).to(self.device)

        self.long_term_layer = nn.Sequential(
            ConvST(self.supports, kt=3, ks=self.ks, dim_in=self.output_dim, dim_out=self.dim_out, device=self.device),
            nn.BatchNorm2d(self.dim_out),
            ConvST(self.supports, kt=3, ks=self.ks, dim_in=self.dim_out, dim_out=self.dim_out, device=self.device),
            nn.BatchNorm2d(self.dim_out),
            ConvST(self.supports, kt=3, ks=self.ks, dim_in=self.dim_out, dim_out=self.dim_out, device=self.device),
            nn.BatchNorm2d(self.dim_out),
            ConvST(self.supports, kt=3, ks=self.ks, dim_in=self.dim_out, dim_out=self.dim_out, device=self.device),
            nn.BatchNorm2d(self.dim_out),
            ConvST(self.supports, kt=3, ks=self.ks, dim_in=self.dim_out, dim_out=self.dim_out, device=self.device),
            nn.BatchNorm2d(self.dim_out),
            ConvST(self.supports, kt=2, ks=self.ks, dim_in=self.dim_out, dim_out=self.dim_out, device=self.device),
            nn.BatchNorm2d(self.dim_out),
        )

        self.short_term_gcn = nn.Sequential(
            ConvST(self.supports, kt=3, ks=self.ks, dim_in=self.output_dim, dim_out=self.dim_out, device=self.device),
            nn.BatchNorm2d(self.dim_out),
            ConvST(self.supports, kt=3, ks=self.ks, dim_in=self.dim_out, dim_out=self.dim_out, device=self.device),
            nn.BatchNorm2d(self.dim_out),
            ConvST(self.supports, kt=3, ks=self.ks, dim_in=self.dim_out, dim_out=self.dim_out, device=self.device),
            nn.BatchNorm2d(self.dim_out),
        )

        self.attention_t = AttentionT(self.device, self.input_window + self.window,
                                      self.num_nodes, self.dim_out, self.ext_dim)
        self.attention_c_1 = AttentionC(self.device, self.num_nodes, self.dim_out, self.ext_dim)
        self.attention_c_2 = AttentionC(self.device, self.num_nodes, self.dim_out, self.ext_dim)

    def forward(self, batch):
        inputs = batch['X'][:, :, :, :self.output_dim].contiguous()  # (B, input_window, N, output_dim)
        inputs = inputs.permute(0, 3, 1, 2)  # (B, output_dim, input_window, N)
        # input_ext = batch['X'][:, :, 0, self.output_dim:].contiguous()  # (B, input_window, ext_dim)
        batch_size, input_dim, len_time, num_nodes = inputs.shape
        assert num_nodes == self.num_nodes
        assert len_time == self.input_window
        assert input_dim == self.output_dim

        labels = batch['y'][:, :, :, :self.output_dim].contiguous()  # (B, output_window, N, output_dim)
        labels = labels.permute(0, 3, 1, 2)  # (B, output_dim, output_window, N)
        labels_ext = batch['y'][:, :, 0, self.output_dim:].contiguous()  # (B, output_window, ext_dim)

        long_output = self.long_term_layer(inputs)  # (B, dim_out, input_window, N)
        preds = []

        if self.training:
            label_padding = inputs[:, :, -self.window:, :]  # (B, feature_dim, window, N)
            padded_labels = torch.cat((label_padding, labels), dim=2)  # (B, feature_dim, window+output_window, N)
            padded_labels = torch.stack([padded_labels[:, :, i:i + self.window, :]
                                         for i in range(0, self.output_window)], dim=2)
            # (B, feature_dim, output_window, window, N)
            for i in range(0, self.output_window):
                s_inputs = padded_labels[:, :, i, :, :]  # (B, feature_dim, window, N)
                ext_input = labels_ext[:, i, :]  # (B, ext_dim)
                short_output = self.short_term_gcn(s_inputs)  # (B, dim_out, window, N)
                ls_inputs = torch.cat((short_output, long_output), dim=2)
                # (B, dim_out, input_window + window, N)
                ls_inputs = self.attention_t(ext_input, ls_inputs)
                if self.output_dim == 1:
                    pred = self.attention_c_1(ext_input, ls_inputs)
                elif self.output_dim == 2:
                    pred = torch.cat((self.attention_c_1(ext_input, ls_inputs),
                                      self.attention_c_2(ext_input, ls_inputs)), dim=1)
                else:
                    raise ValueError('Error Set output_dim!')
                # pred: (B, output_dim, 1, N)
                label_padding = torch.cat((label_padding[:, :, 1:, :], pred), dim=2)
                preds.append(pred)
        else:
            label_padding = inputs[:, :, -self.window:, :]  # (B, feature_dim, window, N)
            for i in range(0, self.output_window):
                s_inputs = label_padding
                ext_input = labels_ext[:, i, :]  # (B, ext_dim)
                short_output = self.short_term_gcn(s_inputs)  # (B, dim_out, window, N)
                ls_inputs = torch.cat((short_output, long_output), dim=2)
                # (B, dim_out, input_window + window, N)
                ls_inputs = self.attention_t(ext_input, ls_inputs)
                if self.output_dim == 1:
                    pred = self.attention_c_1(ext_input, ls_inputs)
                elif self.output_dim == 2:
                    pred = torch.cat((self.attention_c_1(ext_input, ls_inputs),
                                      self.attention_c_2(ext_input, ls_inputs)), dim=1)
                else:
                    raise ValueError('Error Set output_dim!')
                # pred: (B, output_dim, 1, N)
                label_padding = torch.cat((label_padding[:, :, 1:, :], pred), dim=2)
                preds.append(pred)
        return torch.cat(preds, dim=2).permute(0, 2, 3, 1)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true)

    def predict(self, batch):
        return self.forward(batch)
