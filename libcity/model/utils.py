import scipy.sparse as sp
from scipy.sparse import linalg
import numpy as np
import torch


# def build_sparse_matrix(device, lap):
#     lap = lap.tocoo()
#     indices = np.column_stack((lap.row, lap.col))
#     # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
#     indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
#     lap = torch.sparse_coo_tensor(indices.T, lap.data, lap.shape, device=device)
#     return lap.to(torch.float32)


def build_sparse_matrix(device, lap):
    """
    构建稀疏矩阵(tensor)

    Args:
        device:
        lap: 拉普拉斯

    Returns:

    """
    shape = lap.shape
    i = torch.LongTensor(np.vstack((lap.row, lap.col)).astype(int))
    v = torch.FloatTensor(lap.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)


def get_cheb_polynomial(l_tilde, k):
    """
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Args:
        l_tilde(scipy.sparse.coo.coo_matrix): scaled Laplacian, shape (N, N)
        k(int): the maximum order of chebyshev polynomials

    Returns:
        list(np.ndarray): cheb_polynomials, length: K, from T_0 to T_{K-1}
    """
    l_tilde = sp.coo_matrix(l_tilde)
    num = l_tilde.shape[0]
    cheb_polynomials = [sp.eye(num).tocoo(), l_tilde.copy()]
    for i in range(2, k + 1):
        cheb_i = (2 * l_tilde).dot(cheb_polynomials[i - 1]) - cheb_polynomials[i - 2]
        cheb_polynomials.append(cheb_i.tocoo())
    return cheb_polynomials


def get_supports_matrix(adj_mx, filter_type='laplacian', undirected=True):
    """
    选择不同类别的拉普拉斯

    Args:
        undirected:
        adj_mx:
        filter_type:

    Returns:

    """
    supports = []
    if filter_type == "laplacian":
        supports.append(calculate_scaled_laplacian(adj_mx, lambda_max=None, undirected=undirected))
    elif filter_type == "random_walk":
        supports.append(calculate_random_walk_matrix(adj_mx).T)
    elif filter_type == "dual_random_walk":
        supports.append(calculate_random_walk_matrix(adj_mx).T)
        supports.append(calculate_random_walk_matrix(adj_mx.T).T)
    else:
        supports.append(calculate_scaled_laplacian(adj_mx))
    return supports


def calculate_normalized_laplacian(adj):
    """
    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    对称归一化的拉普拉斯

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
    """
    L = D^-1 * A
    随机游走拉普拉斯

    Args:
        adj_mx: adj matrix

    Returns:
        np.ndarray: L
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    """
    计算近似后的拉普莱斯矩阵~L

    Args:
        adj_mx:
        lambda_max:
        undirected:

    Returns:
        ~L = 2 * L / lambda_max - I
    """
    adj_mx = sp.coo_matrix(adj_mx)
    if undirected:
        bigger = adj_mx > adj_mx.T
        smaller = adj_mx < adj_mx.T
        notequall = adj_mx != adj_mx.T
        adj_mx = adj_mx - adj_mx.multiply(notequall) + adj_mx.multiply(bigger) + adj_mx.T.multiply(smaller)
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32).tocoo()
