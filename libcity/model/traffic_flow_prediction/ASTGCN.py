import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
from scipy.sparse.linalg import eigs


def scaled_laplacian(weight):
    """
    compute ~L (scaled laplacian matrix)
    L = D - A
    ~L = 2L/lambda - I

    Args:
        weight(np.ndarray): shape is (N, N), N is the num of vertices

    Returns:
        np.ndarray: ~L, shape (N, N)
    """
    assert weight.shape[0] == weight.shape[1]
    n = weight.shape[0]
    diag = np.diag(np.sum(weight, axis=1))
    lap = diag - weight
    for i in range(n):
        for j in range(n):
            if diag[i, i] > 0 and diag[j, j] > 0:
                lap[i, j] /= np.sqrt(diag[i, i] * diag[j, j])
    lambda_max = eigs(lap, k=1, which='LR')[0].real
    return (2 * lap) / lambda_max - np.identity(weight.shape[0])


def cheb_polynomial(l_tilde, k):
    """
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Args:
        l_tilde(np.ndarray): scaled Laplacian, shape (N, N)
        k(int): the maximum order of chebyshev polynomials

    Returns:
        list(np.ndarray): cheb_polynomials, length: K, from T_0 to T_{K-1}
    """
    num = l_tilde.shape[0]
    cheb_polynomials = [np.identity(num), l_tilde.copy()]
    for i in range(2, k):
        cheb_polynomials.append(np.matmul(2 * l_tilde, cheb_polynomials[i - 1]) - cheb_polynomials[i - 2])
    return cheb_polynomials


class SpatialAttentionLayer(nn.Module):
    """
    compute spatial attention scores
    """

    def __init__(self, device, in_channels, num_of_vertices, num_of_timesteps):
        super(SpatialAttentionLayer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(device))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(device))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(device))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(device))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(device))

    def forward(self, x):
        """
        Args:
            x(torch.tensor): (B, N, F_in, T)

        Returns:
            torch.tensor: (B,N,N)
        """
        # x * W1 --> (B,N,F,T)(T)->(B,N,F)
        # x * W1 * W2 --> (B,N,F)(F,T)->(B,N,T)
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)
        # (W3 * x) ^ T --> (F)(B,N,F,T)->(B,N,T)-->(B,T,N)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)
        # x = lhs * rhs --> (B,N,T)(B,T,N) -> (B, N, N)
        product = torch.matmul(lhs, rhs)
        # S = Vs * sig(x + bias) --> (N,N)(B,N,N)->(B,N,N)
        s = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        # softmax (B,N,N)
        s_normalized = F.softmax(s, dim=1)
        return s_normalized


class ChebConvWithSAt(nn.Module):
    """
    K-order chebyshev graph convolution
    """

    def __init__(self, k, cheb_polynomials, in_channels, out_channels):
        """
        Args:
            k(int): K-order
            cheb_polynomials: cheb_polynomials
            in_channels(int): num of channels in the input sequence
            out_channels(int): num of channels in the output sequence
        """
        super(ChebConvWithSAt, self).__init__()
        self.K = k
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).
                                                    to(self.DEVICE)) for _ in range(k)])

    def forward(self, x, spatial_attention):
        """
        Chebyshev graph convolution operation

        Args:
            x: (batch_size, N, F_in, T)
            spatial_attention: (batch_size, N, N)

        Returns:
            torch.tensor: (batch_size, N, F_out, T)
        """
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                t_k = self.cheb_polynomials[k]  # (N,N)

                t_k_with_at = t_k.mul(spatial_attention)   # (N,N)*(B,N,N) = (B,N,N) .mul->element-wise的乘法

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = t_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (B, N, N)(B, N, F_in) = (B, N, F_in)

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class TemporalAttentionLayer(nn.Module):
    def __init__(self, device, in_channels, num_of_vertices, num_of_timesteps):
        super(TemporalAttentionLayer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(device))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(device))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(device))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(device))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(device))

    def forward(self, x):
        """
        Args:
            x: (batch_size, N, F_in, T)

        Returns:
            torch.tensor: (B, T, T)
        """

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        e = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        e_normalized = F.softmax(e, dim=1)

        return e_normalized


class ASTGCNBlock(nn.Module):
    def __init__(self, device, in_channels, k, nb_chev_filter, nb_time_filter,
                 time_strides, cheb_polynomials, num_of_vertices, num_of_timesteps):
        super(ASTGCNBlock, self).__init__()
        self.TAt = TemporalAttentionLayer(device, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = SpatialAttentionLayer(device, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = ChebConvWithSAt(k, cheb_polynomials, in_channels, nb_chev_filter)
        # 时间卷积: 输入时间长度 = num_of_timesteps = time_strides * output_window
        # 输入必须是输出output_window的固定倍数！
        # ker=3, pad=2, stride=time_strides
        # 输出时间长度 = (time_strides * output_window + 2 * pad - ker) / time_strides + 1 = output_window
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3),
                                   stride=(1, time_strides), padding=(0, 1))
        # 时间维度上卷积: 输入时间长度 = num_of_timesteps = time_strides * output_window
        # ker=1, stride=time_strides
        # 输出时间长度 = (time_strides * output_window - ker) / time_strides + 1 = output_window
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)  # 需要将channel放到最后一个维度上

    def forward(self, x):
        """
        Args:
            x: (batch_size, N, F_in, T)

        Returns:
            torch.tensor: (batch_size, N, nb_time_filter, output_window)
        """
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # TAt
        temporal_at = self.TAt(x)  # (B, T, T)

        x_tat = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_at)\
            .reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        # 结合时间注意力：(B, N*F_in, T) * (B, T, T) -> (B, N*F_in, T) -> (B, N, F_in, T)

        # SAt
        spatial_at = self.SAt(x_tat)  # (B, N, N)

        # 结合空间注意力的图卷积 cheb gcn
        spatial_gcn = self.cheb_conv_SAt(x, spatial_at)  # (B, N, F_out, T), F_out = nb_chev_filter

        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))
        # (B, N, F_out, T) -> (B, F_out, N, T) 用(1,3)的卷积核去做->(B, F_out', N, T') F_out'=nb_time_filter

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
        # (B, N, F_in, T) -> (B, F_in, N, T) 用(1,1)的卷积核去做->(B, F_out', N, T') F_out'=nb_time_filter

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (B, F_out', N, T') -> (B, T', N, F_out') -ln -> (B, T', N, F_out') -> (B, N, F_out', T')

        return x_residual


class FusionLayer(nn.Module):
    # Matrix-based fusion
    def __init__(self, n, h, w, device):
        super(FusionLayer, self).__init__()
        # define the trainable parameter
        self.weights = nn.Parameter(torch.FloatTensor(1, n, h, w).to(device))

    def forward(self, x):
        # assuming x is of size B-n-h-w
        x = x * self.weights  # element-wise multiplication
        return x


class ASTGCNSubmodule(nn.Module):
    def __init__(self, device, nb_block, in_channels, k, nb_chev_filter, nb_time_filter,
                 time_strides, cheb_polynomials, output_window, output_dim, num_of_vertices):
        super(ASTGCNSubmodule, self).__init__()

        self.BlockList = nn.ModuleList([ASTGCNBlock(device, in_channels, k, nb_chev_filter,
                                                    nb_time_filter, time_strides, cheb_polynomials,
                                                    num_of_vertices, time_strides * output_window)])

        self.BlockList.extend([ASTGCNBlock(device, nb_time_filter, k, nb_chev_filter,
                                           nb_time_filter, 1, cheb_polynomials,
                                           num_of_vertices, output_window)
                               for _ in range(nb_block-1)])

        self.final_conv = nn.Conv2d(output_window, output_window,
                                    kernel_size=(1, nb_time_filter - output_dim + 1))

        self.fusionlayer = FusionLayer(output_window, num_of_vertices, output_dim, device)

    def forward(self, x):
        """
        Args:
            x: (B, T_in, N_nodes, F_in)

        Returns:
            torch.tensor: (B, T_out, N_nodes, out_dim)
        """
        x = x.permute(0, 2, 3, 1)  # (B, N, F_in(feature_dim), T_in)
        for block in self.BlockList:
            x = block(x)  # 每个时空块的输出维度是nb_time_filter
        # (B, N, F_out(nb_time_filter), T_out(output_window))
        # 将nb_time_filter变成output_dim
        output = self.final_conv(x.permute(0, 3, 1, 2))
        # (B, N, F_out, T_out) --> (B, T_out, N, F_out) --> conv<1,F_out-out_dim+1> --> (B, T_out, N, out_dim)
        output = self.fusionlayer(output)
        return output


class ASTGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.len_period = self.data_feature.get('len_period', 0)
        self.len_trend = self.data_feature.get('len_trend', 0)
        self.len_closeness = self.data_feature.get('len_closeness', 0)
        if self.len_period == 0 and self.len_trend == 0 and self.len_closeness == 0:
            raise ValueError('Num of days/weeks/hours are all zero! Set at least one of them not zero!')
        self.output_dim = self.data_feature.get('output_dim', 1)

        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        self.nb_block = config.get('nb_block', 2)
        self.K = config.get('K', 3)
        self.nb_chev_filter = config.get('nb_chev_filter', 64)
        self.nb_time_filter = config.get('nb_time_filter', 64)

        adj_mx = self.data_feature.get('adj_mx')
        l_tilde = scaled_laplacian(adj_mx)
        self.cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(self.device)
                                 for i in cheb_polynomial(l_tilde, self.K)]
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        if self.len_closeness > 0:
            self.hours_ASTGCN_submodule = \
                ASTGCNSubmodule(self.device, self.nb_block, self.feature_dim,
                                self.K, self.nb_chev_filter, self.nb_time_filter,
                                self.len_closeness // self.output_window, self.cheb_polynomials,
                                self.output_window, self.output_dim, self.num_nodes)
        if self.len_period > 0:
            self.days_ASTGCN_submodule = \
                ASTGCNSubmodule(self.device, self.nb_block, self.feature_dim,
                                self.K, self.nb_chev_filter, self.nb_time_filter,
                                self.len_period // self.output_window, self.cheb_polynomials,
                                self.output_window, self.output_dim, self.num_nodes)
        if self.len_trend > 0:
            self.weeks_ASTGCN_submodule = \
                ASTGCNSubmodule(self.device, self.nb_block, self.feature_dim,
                                self.K, self.nb_chev_filter, self.nb_time_filter,
                                self.len_trend // self.output_window, self.cheb_polynomials,
                                self.output_window, self.output_dim, self.num_nodes)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, batch):
        x = batch['X']  # (B, Tw+Td+Th, N_nodes, F_in)
        # 时间维度(第1维)上的顺序是CPT，即
        # [0, len_closeness) -- input1
        # [len_closeness, len_closeness+len_period) -- input2
        # [len_closeness+len_period, len_closeness+len_period+len_trend) -- input3
        output = 0
        if self.len_closeness > 0:
            begin_index = 0
            end_index = begin_index + self.len_closeness
            output_hours = self.hours_ASTGCN_submodule(x[:, begin_index:end_index, :, :])
            output += output_hours
        if self.len_period > 0:
            begin_index = self.len_closeness
            end_index = begin_index + self.len_period
            output_days = self.days_ASTGCN_submodule(x[:, begin_index:end_index, :, :])
            output += output_days
        if self.len_trend > 0:
            begin_index = self.len_closeness + self.len_period
            end_index = begin_index + self.len_trend
            output_weeks = self.weeks_ASTGCN_submodule(x[:, begin_index:end_index, :, :])
            output += output_weeks
        return output  # (B, Tp, N_nodes, F_out)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true)

    def predict(self, batch):
        return self.forward(batch)
