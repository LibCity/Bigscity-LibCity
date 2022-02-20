import numpy as np
import torch
from torch import nn

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class AdjProcessor:
    def __init__(self, kernel_type: str, K: int):
        self.kernel_type = kernel_type
        # chebyshev (Defferard NIPS'16)/localpool (Kipf ICLR'17)/random_walk_diffusion (Li ICLR'18)
        self.K = K if self.kernel_type != 'localpool' else 1
        # max_chebyshev_polynomial_order (Defferard NIPS'16)/max_diffusion_step (Li ICLR'18)

    def process(self, adj: torch.Tensor):
        kernel_list = list()

        if self.kernel_type in ['localpool', 'chebyshev']:  # spectral
            adj_norm = self.symmetric_normalize(adj)
            if self.kernel_type == 'localpool':
                localpool = torch.eye(adj_norm.shape[0]) + adj_norm  # same as add self-loop first
                kernel_list.append(localpool)

            else:  # chebyshev
                laplacian_norm = torch.eye(adj_norm.shape[0]) - adj_norm
                laplacian_rescaled = self.rescale_laplacian(laplacian_norm)
                kernel_list = self.compute_chebyshev_polynomials(laplacian_rescaled, kernel_list)

        elif self.kernel_type == 'random_walk_diffusion':  # spatial
            # diffuse k steps on transition matrix P
            P_forward = self.random_walk_normalize(adj)
            kernel_list = self.compute_chebyshev_polynomials(P_forward.T, kernel_list)

        elif self.kernel_type == 'dual_random_walk_diffusion':
            # diffuse k steps bidirectionally on transition matrix P
            P_forward = self.random_walk_normalize(adj)
            P_backward = self.random_walk_normalize(adj.T)
            forward_series, backward_series = [], []
            forward_series = self.compute_chebyshev_polynomials(P_forward.T, forward_series)
            backward_series = self.compute_chebyshev_polynomials(P_backward.T, backward_series)
            kernel_list += forward_series + backward_series[1:]  # 0-order Chebyshev polynomial is same: I

        else:
            raise ValueError(
                'Invalid kernel_type. Must be one of [chebyshev, localpool, random_walk_diffusion, dual_random_walk_diffusion].')

        kernels = torch.stack(kernel_list, dim=0)
        return kernels

    @staticmethod
    def random_walk_normalize(A):  # asymmetric
        d_inv = torch.pow(A.sum(dim=1), -1)  # OD matrix Ai,j sum on j (axis=1)
        d_inv[torch.isinf(d_inv)] = 0.
        D = torch.diag(d_inv)
        P = torch.mm(D, A)
        return P

    @staticmethod
    def symmetric_normalize(A):
        D = torch.diag(torch.pow(A.sum(dim=1), -0.5))
        A_norm = torch.mm(torch.mm(D, A), D)
        return A_norm

    @staticmethod
    def rescale_laplacian(L):
        # rescale laplacian to arccos range [-1,1] for input to Chebyshev polynomials of the first kind
        try:
            lambda_ = torch.eig(L)[0][:, 0]  # get the real parts of eigenvalues
            lambda_max = lambda_.max()  # get the largest eigenvalue
        except:
            print("Eigen_value calculation didn't converge, using max_eigen_val=2 instead.")
            lambda_max = 2
        L_rescaled = (2 / lambda_max) * L - torch.eye(L.shape[0])
        return L_rescaled

    def compute_chebyshev_polynomials(self, x, T_k):
        # compute Chebyshev polynomials up to order k. Return a list of matrices.
        # print(f"Computing Chebyshev polynomials up to order {self.K}.")
        for k in range(self.K + 1):
            if k == 0:
                T_k.append(torch.eye(x.shape[0]))
            elif k == 1:
                T_k.append(x)
            else:
                T_k.append(2 * torch.mm(x, T_k[k - 1]) - T_k[k - 2])
        return T_k


class ODconv(nn.Module):  # Origin-Destination Convolution
    def __init__(self, K: int, input_dim: int, hidden_dim: int, use_bias=True, activation=None):
        super(ODconv, self).__init__()
        self.K = K
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.activation = activation() if activation is not None else None
        self.init_params()

    def init_params(self, b_init=0.0):
        self.W = nn.Parameter(torch.empty(self.input_dim * (self.K ** 2), self.hidden_dim), requires_grad=True)
        nn.init.xavier_normal_(self.W)
        if self.use_bias:
            self.b = nn.Parameter(torch.empty(self.hidden_dim), requires_grad=True)
            nn.init.constant_(self.b, val=b_init)
        return

    def forward(self, X: torch.Tensor, G: torch.Tensor or tuple):
        feat_set = list()
        if type(G) == tuple:  # dynamic: O-D graph pair
            assert len(G) == 2
            if (len(G[0].shape) == 2) & (
                    len(G[1].shape) == 2):  # adaptive graph input: (N, N) -> preprocess based on Chebyshev polynomials
                T_k = [[torch.eye(G[i].shape[0]).to(X.device), G[i]] for i in range(2)]  # initialize for O and D
                T_k_pair = []
                for i in range(2):
                    for k in range(2, self.K):
                        T_k[i].append(2 * torch.mm(G[i], T_k[i][-1]) - T_k[i][-2])
                    T_k_pair.append(torch.stack(T_k[i], dim=0))
                G = tuple(T_k_pair)  # adaptive dynamic graph input: ((K, N, N), (K, N, N))

                for o in range(self.K):
                    for d in range(self.K):
                        mode_1_prod = torch.einsum('bncl,nm->bmcl', X, G[0][o, :, :])
                        mode_2_prod = torch.einsum('bmcl,cd->bmdl', mode_1_prod, G[1][d, :, :])
                        feat_set.append(mode_2_prod)
            else:
                raise NotImplementedError
        else:  # static
            assert self.K == G.shape[-3]
            for o in range(self.K):
                for d in range(self.K):
                    mode_1_prod = torch.einsum('bncl,nm->bmcl', X, G[o, :, :])
                    mode_2_prod = torch.einsum('bmcl,cd->bmdl', mode_1_prod, G[d, :, :])
                    feat_set.append(mode_2_prod)

        _2D_feat = torch.cat(feat_set, dim=-1)
        mode_3_prod = torch.einsum('bmdk,kh->bmdh', _2D_feat, self.W)

        if self.use_bias:
            mode_3_prod += self.b
        H = self.activation(mode_3_prod) if self.activation is not None else mode_3_prod
        return H


class ODCRUcell(nn.Module):  # Origin-Destination Convolutional Recurrent Unit
    def __init__(self, num_nodes: int, K: int, input_dim: int, hidden_dim: int, use_bias=True, activation=None):
        super(ODCRUcell, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.gates = ODconv(K, input_dim + hidden_dim, hidden_dim * 2, use_bias, activation)
        self.candi = ODconv(K, input_dim + hidden_dim, hidden_dim, use_bias, activation)

    def init_hidden(self, batch_size: int):
        weight = next(self.parameters()).data
        hidden = (weight.new_zeros(batch_size, self.num_nodes, self.num_nodes, self.hidden_dim))
        return hidden

    def forward(self, G: torch.Tensor, Xt: torch.Tensor, Ht_1: torch.Tensor):
        assert len(Xt.shape) == len(Ht_1.shape) == 4, 'ODCRU cell must take in 4D tensor as input [Xt, Ht-1]'

        XH = torch.cat([Xt, Ht_1], dim=-1)
        XH_conv = self.gates(X=XH, G=G)

        u, r = torch.split(XH_conv, self.hidden_dim, dim=-1)
        update = torch.sigmoid(u)
        reset = torch.sigmoid(r)

        candi = torch.cat([Xt, reset * Ht_1], dim=-1)
        candi_conv = torch.tanh(self.candi(X=candi, G=G))

        Ht = (1.0 - update) * Ht_1 + update * candi_conv
        return Ht


class ODCRUencoder(nn.Module):
    def __init__(self, num_nodes: int, K: int, input_dim: int, hidden_dim: int, num_layers: int, use_bias=True,
                 activation=None, return_all_layers=True):
        super(ODCRUencoder, self).__init__()
        self.hidden_dim = self._extend_for_multilayers(hidden_dim, num_layers)
        self.num_layers = num_layers
        self.return_all_layers = return_all_layers
        assert len(self.hidden_dim) == self.num_layers, 'Input [hidden, layer] length must be consistent'

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else self.hidden_dim[i - 1]
            self.cell_list.append(
                ODCRUcell(num_nodes, K, cur_input_dim, self.hidden_dim[i], use_bias=use_bias, activation=activation))

    def forward(self, G_list: list, X_seq: torch.Tensor, H0_l=None):
        assert len(X_seq.shape) == 5, 'ODCRU encoder must take in 5D tensor as input X_seq'
        assert self.num_layers == len(G_list), '#layers must be consistent with length of input G_list'

        batch_size, seq_len, _, _, _ = X_seq.shape
        if H0_l is None:
            H0_l = self._init_hidden(batch_size)

        out_seq_lst = list()  # layerwise output seq
        Ht_lst = list()  # layerwise last state
        in_seq_l = X_seq  # current input seq

        for l in range(self.num_layers):
            Ht = H0_l[l]
            out_seq_l = list()
            for t in range(seq_len):
                if (type(G_list[l]) is not torch.Tensor) & (type(G_list[l]) != tuple):  # if DGC
                    G = G_list[l](x_t=X_seq[:, t, :, :, :])  # query O-D graph pair
                else:
                    G = G_list[l]
                Ht = self.cell_list[l](G=G, Xt=in_seq_l[:, t, ...], Ht_1=Ht)
                out_seq_l.append(Ht)

            out_seq_l = torch.stack(out_seq_l, dim=1)  # (B, T, N, C, h)
            in_seq_l = out_seq_l  # update input seq

            out_seq_lst.append(out_seq_l)
            Ht_lst.append(Ht)

        if not self.return_all_layers:
            out_seq_lst = out_seq_lst[-1:]
            Ht_lst = Ht_lst[-1:]
        return out_seq_lst, Ht_lst

    def _init_hidden(self, batch_size):
        H0_l = []
        for i in range(self.num_layers):
            H0_l.append(self.cell_list[i].init_hidden(batch_size))
        return H0_l

    @staticmethod
    def _extend_for_multilayers(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ODCRUdecoder(nn.Module):
    def __init__(self, num_nodes: int, K: int, output_dim: int, hidden_dim: int, out_horizon: int, num_layers: int,
                 use_bias=True, activation=None):
        super(ODCRUdecoder, self).__init__()
        self.out_horizon = out_horizon  # output steps
        self.hidden_dim = self._extend_for_multilayers(hidden_dim, num_layers)
        self.num_layers = num_layers
        assert len(self.hidden_dim) == self.num_layers, 'Input [hidden, layer] length must be consistent'

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = output_dim if i == 0 else self.hidden_dim[i - 1]
            self.cell_list.append(
                ODCRUcell(num_nodes, K, cur_input_dim, self.hidden_dim[i], use_bias=use_bias, activation=activation))

    def forward(self, G_list: list, Xt: torch.Tensor, H0_l: list):
        assert len(Xt.shape) == 4, 'ODCRU decoder must take in 4D tensor as input Xt'
        assert self.num_layers == len(G_list), '#layers must be consistent with length of input G_list'

        Ht_lst = list()  # layerwise hidden state
        Xin_l = Xt

        for l in range(self.num_layers):
            if type(G_list[l]) is not torch.Tensor:  # if DGC
                G = G_list[l](x_t=Xt[:, :, :, :])  # query O-D graph pair
            else:
                G = G_list[l]
            Ht_l = self.cell_list[l](G=G, Xt=Xin_l, Ht_1=H0_l[l])
            Ht_lst.append(Ht_l)
            Xin_l = Ht_l  # update input for next layer

        return Ht_l, Ht_lst

    @staticmethod
    def _extend_for_multilayers(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class DynGraphConstructor(nn.Module):  # parameterize a dynamic graph pair
    def __init__(self, num_nodes: int):
        super(DynGraphConstructor, self).__init__()
        self.W_o = nn.Parameter(torch.empty(num_nodes, num_nodes),
                                requires_grad=True)  # weight of destination dynamics for origins
        nn.init.xavier_normal_(self.W_o)
        self.W_d = nn.Parameter(torch.empty(num_nodes, num_nodes),
                                requires_grad=True)  # weight of origin dynamics for destinations
        nn.init.xavier_normal_(self.W_d)

    def forward(self, x_t: torch.Tensor):  # query O-D graph pair
        assert len(x_t.shape) == 4  # (batch, N, N, hidden)

        O = torch.softmax(torch.relu(torch.einsum('bpdh,dd,bqdh->pq', x_t, self.W_o.to(x_t.device), x_t)), dim=1)
        D = torch.softmax(torch.relu(torch.einsum('boeh,oo,bofh->ef', x_t, self.W_d.to(x_t.device), x_t)), dim=1)
        return (O, D)


def get_support_K(kernel_type, cheby_order):
    if kernel_type == 'localpool':
        assert cheby_order == 1
        K = 1
    elif (kernel_type == 'chebyshev') | (kernel_type == 'random_walk_diffusion'):
        K = cheby_order + 1
    elif kernel_type == 'dual_random_walk_diffusion':
        K = cheby_order * 2 + 1
    else:
        raise ValueError('Invalid kernel_type. Must be one of '
                         '[chebyshev, localpool, random_walk_diffusion, dual_random_walk_diffusion].')
    return K


def preprocess_adj(adj_mtx: np.array, kernel_type, cheby_order):
    adj_preprocessor = AdjProcessor(kernel_type, cheby_order)
    adj = torch.from_numpy(adj_mtx).float()
    adj = adj_preprocessor.process(adj)
    return adj  # G: (support_K, N, N)


class ODCRN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # initialize static graphs and K values
        self.K = get_support_K(config['kernel_type'], config['cheby_order'])
        self.G = preprocess_adj(adj_mtx=data_feature['adj_mx'],
                                kernel_type=config['kernel_type'],
                                cheby_order=config['cheby_order'])

        self.device = config.get('device', 'cpu')

        self.num_layers = config['num_layers']
        self.hidden_dim = config['hidden_dim']
        self.output_window = config['output_window']
        self._scaler = self.data_feature.get('scaler')
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.use_DGC = config['use_DGC']

        self.num_nodes = data_feature['num_nodes']

        if self.use_DGC:
            self.DGC = DynGraphConstructor(self.num_nodes)
        # encoder-decoder framework
        self.encoder = ODCRUencoder(self.num_nodes, self.K, self.input_dim, self.hidden_dim, self.num_layers,
                                    use_bias=True,
                                    activation=None,
                                    return_all_layers=True)
        self.decoder = ODCRUdecoder(self.num_nodes, self.K, self.input_dim, self.hidden_dim, self.output_window,
                                    self.num_layers,
                                    use_bias=True,
                                    activation=None)
        self.linear = nn.Linear(in_features=self.hidden_dim, out_features=self.input_dim, bias=True)

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
        x = batch['X']
        return self.forward(x)

    def forward(self, X_seq: torch.Tensor):

        # encoding
        _, Ht_lst = self.encoder(G_list=[self.DGC, self.G], X_seq=X_seq, H0_l=None) if self.use_DGC else self.encoder(
            G_list=[self.G, self.G], X_seq=X_seq, H0_l=None)

        # initiate decoder input
        deco_input = torch.zeros(X_seq.shape[:1] + X_seq.shape[2:], device=X_seq.device)
        # decoding
        outputs = list()
        for t in range(self.decoder.out_horizon):
            Ht_l, Ht_lst = self.decoder(G_list=[self.DGC, self.G], Xt=deco_input,
                                        H0_l=Ht_lst) if self.use_DGC else self.decoder(G_list=[self.G, self.G],
                                                                                       Xt=deco_input,
                                                                                       H0_l=Ht_lst)
            output = self.linear(Ht_l)
            deco_input = output  # update decoder input
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        return outputs
