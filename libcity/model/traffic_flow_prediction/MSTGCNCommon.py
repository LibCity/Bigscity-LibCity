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


class ChebConv(nn.Module):
    """
    K-order chebyshev graph convolution
    """
    def __init__(self, k, cheb_polynomials, in_channels, out_channels):
        """
        Args:
            k(int):
            cheb_polynomials:
            in_channels(int): num of channels in the input sequence
            out_channels(int): num of channels in the output sequence
        """
        super(ChebConv, self).__init__()
        self.K = k
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels)
                                                    .to(self.DEVICE)) for _ in range(k)])

    def forward(self, x):
        """
        Chebyshev graph convolution operation

        Args:
            x: (batch_size, N, F_in, T)

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

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(t_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class MSTGCNBlock(nn.Module):
    def __init__(self, in_channels, k, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials):
        super(MSTGCNBlock, self).__init__()
        self.ChebConv = ChebConv(k, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3),
                                   stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x):
        """
        Args:
            x: (batch_size, N, F_in, T)

        Returns:
            torch.tensor: (batch_size, N, nb_time_filter, output_window)
        """
        # cheb gcn
        spatial_gcn = self.ChebConv(x)  # (b,N,F,T)

        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (b,F,N,T)

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,F,N,T)

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)  # (b,N,F,T)

        return x_residual


class MSTGCNSubmodule(nn.Module):
    def __init__(self, device, nb_block, in_channels, k, nb_chev_filter, nb_time_filter,
                 input_window, cheb_polynomials, output_window, output_dim, num_of_vertices):
        super(MSTGCNSubmodule, self).__init__()

        self.BlockList = nn.ModuleList([
            MSTGCNBlock(in_channels, k, nb_chev_filter, nb_time_filter,
                        input_window // output_window, cheb_polynomials)])

        self.BlockList.extend([
            MSTGCNBlock(nb_time_filter, k, nb_chev_filter, nb_time_filter, 1, cheb_polynomials)
            for _ in range(nb_block-1)])

        self.final_conv = nn.Conv2d(output_window, output_window,
                                    kernel_size=(1, nb_time_filter - output_dim + 1))

    def forward(self, x):
        """
        Args:
            x: (B, T_in, N_nodes, F_in)

        Returns:
            torch.tensor: (B, T_out, N_nodes, out_dim)
        """
        x = x.permute(0, 2, 3, 1)  # (B, N, F_in(feature_dim), T_in)
        for block in self.BlockList:
            x = block(x)
        output = self.final_conv(x.permute(0, 3, 1, 2))
        return output


# 适配最一般的TrafficStateGridDataset和TrafficStatePointDataset
class MSTGCNCommon(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)

        self.input_window = config.get('input_window', 1)
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

        self.MSTGCN_submodule = \
            MSTGCNSubmodule(self.device, self.nb_block, self.feature_dim,
                            self.K, self.nb_chev_filter, self.nb_time_filter,
                            self.input_window, self.cheb_polynomials,
                            self.output_window, self.output_dim, self.num_nodes)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, batch):
        x = batch['X'].to(self.device)  # (B, T, N_nodes, F_in)
        output = self.MSTGCN_submodule(x)
        return output  # (B, T', N_nodes, F_out)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true)

    def predict(self, batch):
        return self.forward(batch)
