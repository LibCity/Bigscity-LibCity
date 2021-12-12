import torch
import torch.nn as nn
import numpy as np
from scipy.sparse.linalg import eigs
from Output.mlp import FC


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


class GatedFusion(nn.Module):

    def __init__(self, dim, bn, bn_decay, device):
        super(GatedFusion, self).__init__()
        self.D = dim
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.HS_fc = FC(input_dims=self.D, units=self.D, activations=None,
                        bn=self.bn, bn_decay=self.bn_decay, device=self.device, use_bias=False)
        self.HT_fc = FC(input_dims=self.D, units=self.D, activations=None,
                        bn=self.bn, bn_decay=self.bn_decay, device=self.device, use_bias=True)
        self.output_fc = FC(input_dims=self.D, units=[self.D, self.D], activations=[nn.ReLU, None],
                            bn=self.bn, bn_decay=self.bn_decay, device=self.device)

    def forward(self, HS, HT):
        """
        gated fusion

        Args:
            HS:     (batch_size, num_step, num_nodes, D)
            HT:     (batch_size, num_step, num_nodes, D)

        Returns:
            tensor: (batch_size, num_step, num_nodes, D)

        """
        XS = self.HS_fc(HS)
        XT = self.HT_fc(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.multiply(z, HS), torch.multiply(1 - z, HT))
        H = self.output_fc(H)
        return H
