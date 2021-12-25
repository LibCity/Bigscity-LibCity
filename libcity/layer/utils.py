import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.sparse.linalg import eigs
from Output.mlp import FC
import scipy.sparse as sp
from scipy.sparse import linalg

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
        '''
        gated fusion
        HS:     (batch_size, num_step, num_nodes, D)
        HT:     (batch_size, num_step, num_nodes, D)
        return: (batch_size, num_step, num_nodes, D)
        '''
        XS = self.HS_fc(HS)
        XT = self.HT_fc(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.multiply(z, HS), torch.multiply(1 - z, HT))
        H = self.output_fc(H)
        return H

def calculate_normalized_laplacian(adj):
    """
    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2

    Args:
        adj: adj matrix

    Returns:
        np.ndarray: L
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32)


def calculate_cheb_poly(lap, ks):
    """
    k-order Chebyshev polynomials : T0(L)~Tk(L)
    T0(L)=I/1 T1(L)=L Tk(L)=2LTk-1(L)-Tk-2(L)

    Args:
        lap: scaled laplacian matrix
        ks: k-order

    Returns:
        np.ndarray: T0(L)~Tk(L)
    """
    n = lap.shape[0]
    lap_list = [np.eye(n), lap[:]]
    for i in range(2, ks):
        lap_list.append(np.matmul(2 * lap, lap_list[-1]) - lap_list[-2])
    if ks == 0:
        raise ValueError('Ks must bigger than 0!')
    if ks == 1:
        return np.asarray(lap_list[0:1])  # 1*n*n
    else:
        return np.asarray(lap_list)       # Ks*n*n


def calculate_first_approx(weight):
    '''
    1st-order approximation function.
    :param W: weighted adjacency matrix of G. Not laplacian matrix.
    :return: np.ndarray
    '''
    # TODO: 如果W对角线本来就是全1？
    n = weight.shape[0]
    adj = weight + np.identity(n)
    d = np.sum(adj, axis=1)
    # sinvd = np.sqrt(np.mat(np.diag(d)).I)
    # return np.array(sinvd * A * sinvd)
    sinvd = np.sqrt(np.linalg.inv(np.diag(d)))
    lap = np.matmul(np.matmul(sinvd, adj), sinvd)  # n*n
    lap = np.expand_dims(lap, axis=0)              # 1*n*n
    return lap


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
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

# spatial matrixs based on shortest path algorithm
def get_spatial_matrix(adj_mx):
    h, w = adj_mx.shape
    inf = float("inf")

    S_near = np.zeros((h, w))
    S_middle = np.zeros((h, w))
    S_distant = np.zeros((h, w))

    i = 0
    for row in adj_mx:
        L_min = np.min(row)
        np.place(row, row == inf, [-1])
        L_max = np.max(row)
        eta = (L_max-L_min)/3
        S_near[i] = np.logical_and(row >= L_min, row < L_min + eta)
        S_middle[i] = np.logical_and(row >= L_min + eta, row < L_min + 2 * eta)
        S_distant[i] = np.logical_and(row >= L_min + 2*eta, row < L_max)
        i = i + 1

    S_near = S_near.astype(np.float32)
    S_middle = S_middle.astype(np.float32)
    S_distant = S_distant.astype(np.float32)
    return torch.tensor(S_near), torch.tensor(S_middle), torch.tensor(S_distant)
