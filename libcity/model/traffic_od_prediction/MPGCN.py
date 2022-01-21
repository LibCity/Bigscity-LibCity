import numpy
import torch
from torch import nn
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
import numpy as np
from torch import nn
import torch



class GCN(nn.Module):
    def __init__(self, K:int, input_dim:int, hidden_dim:int, bias=True, activation=nn.ReLU):
        super().__init__()
        self.K = K
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.activation = activation() if activation is not None else None
        self.init_params(n_supports=K)

    def init_params(self, n_supports:int, b_init=0):
        self.W = nn.Parameter(torch.empty(n_supports*self.input_dim, self.hidden_dim), requires_grad=True)
        nn.init.xavier_normal_(self.W)      # sampled from a normal distribution N(0, std^2), also known as Glorot initialization
        if self.bias:
            self.b = nn.Parameter(torch.empty(self.hidden_dim), requires_grad=True)
            nn.init.constant_(self.b, val=b_init)

    def forward(self, G:torch.Tensor, x:torch.Tensor):
        '''
        Batch-wise graph convolution operation on given n support adj matrices
        :param G: support adj matrices - torch.Tensor (K, n_nodes, n_nodes)
        :param x: graph feature/signal - torch.Tensor (batch_size, n_nodes, input_dim)
        :return: hidden representation - torch.Tensor (batch_size, n_nodes, hidden_dim)
        '''
        assert self.K == G.shape[0]

        support_list = list()
        for k in range(self.K):
            support = torch.einsum('ij,bjp->bip', [G[k,:,:], x])
            support_list.append(support)
        support_cat = torch.cat(support_list, dim=-1)

        output = torch.einsum('bip,pq->biq', [support_cat, self.W])
        if self.bias:
            output += self.b
        output = self.activation(output) if self.activation is not None else output
        return output

    def __repr__(self):
        return self.__class__.__name__ + f'({self.K} * input {self.input_dim} -> hidden {self.hidden_dim})'



class Adj_Processor():
    def __init__(self, kernel_type:str, K:int):
        self.kernel_type = kernel_type
        # chebyshev (Defferard NIPS'16)/localpool (Kipf ICLR'17)/random_walk_diffusion (Li ICLR'18)
        self.K = K if self.kernel_type != 'localpool' else 1
        # max_chebyshev_polynomial_order (Defferard NIPS'16)/max_diffusion_step (Li ICLR'18)

    def process(self, flow:torch.Tensor):
        '''
        Generate adjacency matrices
        :param flow: batch flow stat - (batch_size, Origin, Destination) torch.Tensor
        :return: processed adj matrices - (batch_size, K_supports, O, D) torch.Tensor
        '''
        batch_list = list()

        for b in range(flow.shape[0]):
            adj = flow[b, :, :]
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
                raise ValueError('Invalid kernel_type. Must be one of [chebyshev, localpool, random_walk_diffusion, dual_random_walk_diffusion].')

            # #print(f"Minibatch {b}: {self.kernel_type} kernel has {len(kernel_list)} support kernels.")
            kernels = torch.stack(kernel_list, dim=0)
            batch_list.append(kernels)
        batch_adj = torch.stack(batch_list, dim=0)
        return batch_adj

    @staticmethod
    def random_walk_normalize(A):   # asymmetric
        d_inv = torch.pow(A.sum(dim=1), -1)   # OD matrix Ai,j sum on j (axis=1)
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
            lambda_ = torch.eig(L)[0][:,0]      # get the real parts of eigenvalues
            lambda_max = lambda_.max()      # get the largest eigenvalue
        except:
            #print("Eigen_value calculation didn't converge, using max_eigen_val=2 instead.")
            lambda_max = 2
        L_rescaled = (2 / lambda_max) * L - torch.eye(L.shape[0])
        return L_rescaled

    def compute_chebyshev_polynomials(self, x, T_k):
        # compute Chebyshev polynomials up to order k. Return a list of matrices.
        # #print(f"Computing Chebyshev polynomials up to order {self.K}.")
        for k in range(self.K + 1):
            if k == 0:
                T_k.append(torch.eye(x.shape[0]))
            elif k == 1:
                T_k.append(x)
            else:
                T_k.append(2 * torch.mm(x, T_k[k-1]) - T_k[k-2])
        return T_k




class MPGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super(MPGCN,self).__init__(config,data_feature)
        self.M = 2
        self.config = config
        self.num_nodes = self.data_feature.get('num_nodes')
        self.device = config.get('device', torch.device('cpu'))
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.p_interval = config.get('p_interval', 1)
        self.adj_mx = self.data_feature.get('adj_mx')
        self.lstm_hidden_dim = config.get('lstm_hidden_dim', 32)
        self.gcn_hidden_dim = config.get('lstm_hidden_dim', 32)
        self.lstm_num_layers = 1
        self.gcn_num_layers = 3
        self.inputdim = 1
        self.K = self.get_support_K(config.get('kernel_type'),config.get('cheby_order'))
        # initiate a branch of (LSTM, 2DGCN, FC) for each graph input
        self.branch_models = nn.ModuleList()
        self.criterion = self.get_loss()
        self.O_D_G = self.data_feature.get('O_D_G')
        self.D_D_G = self.data_feature.get('D_D_G')
        for m in range(self.M):
            branch = nn.ModuleDict()
            branch['temporal'] = nn.LSTM(input_size=self.inputdim, hidden_size=self.lstm_hidden_dim, num_layers=self.lstm_num_layers, batch_first=True)
            branch['spatial'] = nn.ModuleList()
            for n in range(self.gcn_num_layers):
                cur_input_dim = self.lstm_hidden_dim if n == 0 else self.gcn_hidden_dim
                branch['spatial'].append(BDGCN(K=self.K, input_dim=cur_input_dim, hidden_dim=self.gcn_hidden_dim, use_bias=True, activation=nn.ReLU))
            branch['fc'] = nn.Sequential(
                nn.Linear(in_features=self.gcn_hidden_dim, out_features=1, bias=True),
                nn.ReLU())
            self.branch_models.append(branch)

    @staticmethod
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
    def preprocess_dynamic_graph(self, dyn_G:torch.Tensor):
        # reuse adj_preprocessor initialized in preprocessing static graphs, otherwise needed to initiate one each batch
        return self.adj_preprocessor.process(dyn_G).to(self.config.get('GPU'))         # (batch, K, N, N)

    def preprocess_adj(self, adj_mtx:np.array, kernel_type, cheby_order):
        self.adj_preprocessor = Adj_Processor(kernel_type, cheby_order)
        b_adj = torch.from_numpy(adj_mtx).float().unsqueeze(dim=0)      # batch_size=1
        adj = self.adj_preprocessor.process(b_adj)
        return adj.squeeze(dim=0).to(self.config.get('GPU'))       # G: (support_K, N, N)

    def init_hidden_list(self, batch_size: int):  # for LSTM initialization
        hidden_list = list()
        for m in range(self.M):
            weight = next(self.parameters()).data
            hidden = (weight.new_zeros(self.lstm_num_layers, batch_size * (self.num_nodes ** 2), self.lstm_hidden_dim),
                      weight.new_zeros(self.lstm_num_layers, batch_size * (self.num_nodes ** 2), self.lstm_hidden_dim))
            hidden_list.append(hidden)
        return hidden_list

    def forward(self, x_seq_in: torch.Tensor, G_list: list):
        '''
        :param x_seq_in: (batch, seq, O, D, 1)
        :param G_list: static graph (K, N, N); dynamic OD graph tuple ((batch, K, N, N), (batch, K, N, N))
        :return:
        '''
        x_seq_list = []
        for i in range(x_seq_in.shape[1]):
            x_seq_list.append(x_seq_in[:,i,:,:,:])
        result_list = []
        for j in range(x_seq_in.shape[1]):
            x_seq = x_seq_in[:,j:12+j,:,:,:]
            assert (len(x_seq.shape) == 5) & (self.num_nodes == x_seq.shape[2] == x_seq.shape[3])
            assert len(G_list) == self.M
            batch_size, seq_len, _, _, i = x_seq.shape
            hidden_list = self.init_hidden_list(batch_size)

            lstm_in = x_seq.permute(0, 2, 3, 1, 4).reshape(batch_size * (self.num_nodes ** 2), seq_len, i)
            branch_out = []
            for m in range(self.M):
                lstm_out, hidden_list[m] = self.branch_models[m]['temporal'](lstm_in, hidden_list[m])
                gcn_in = lstm_out[:, -1, :].reshape(batch_size, self.num_nodes, self.num_nodes, self.lstm_hidden_dim)
                for n in range(self.gcn_num_layers):
                    gcn_in = self.branch_models[m]['spatial'][n](gcn_in, G_list[m])
                fc_out = self.branch_models[m]['fc'](gcn_in)
                branch_out.append(fc_out)
            # ensemble
            ensemble_out = torch.mean(torch.stack(branch_out, dim=-1), dim=-1)#one step
            result_list.append(ensemble_out)
            x_seq_list.append(ensemble_out)
            x_seq_in = torch.stack(x_seq_list,dim = 1)
        result = torch.stack(result_list,dim = 1)
        return  result



    def predict(self, batch):
        adj_G = self.preprocess_adj(self.adj_mx,self.config.get('kernel_type'),self.config.get('cheby_order'))
        dyn_OD_G = (self.preprocess_dynamic_graph(torch.from_numpy(self.O_D_G).float()),  self.preprocess_dynamic_graph(torch.from_numpy(self.D_D_G).float()))
        return self.forward(batch['X'],[adj_G,dyn_OD_G])

    def get_loss(self):
        if self.config.get('loss') == 'MSE':
            criterion = nn.MSELoss(reduction='mean')
        elif self.config.get('loss') == 'MAE':
            criterion = nn.L1Loss(reduction='mean')
        elif self.config.get('loss') == 'Huber':
            criterion = nn.SmoothL1Loss(reduction='mean')
        else:
            raise NotImplementedError('Invalid loss function.')
        return criterion

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_pred = self.predict(batch)
        return self.criterion(y_pred, y_true)

class BDGCN(nn.Module):        # 2DGCN: handling both static and dynamic graph input
    def __init__(self, K:int, input_dim:int, hidden_dim:int, use_bias=True, activation=None):
        super(BDGCN, self).__init__()
        self.K = K
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.activation = activation() if activation is not None else None
        self.init_params()

    def init_params(self, b_init=0.0):
        self.W = nn.Parameter(torch.empty(self.input_dim*(self.K**2), self.hidden_dim), requires_grad=True)
        nn.init.xavier_normal_(self.W)
        if self.use_bias:
            self.b = nn.Parameter(torch.empty(self.hidden_dim), requires_grad=True)
            nn.init.constant_(self.b, val=b_init)
        return

    def forward(self, X:torch.Tensor, G:torch.Tensor or tuple):
        feat_set = list()
        if type(G) == torch.Tensor:         # static graph input: (K, N, N)
            assert self.K == G.shape[-3]
            for o in range(self.K):
                for d in range(self.K):
                    mode_1_prod = torch.einsum('bncl,nm->bmcl', X, G[o, :, :])
                    mode_2_prod = torch.einsum('bmcl,cd->bmdl', mode_1_prod, G[d, :, :])
                    feat_set.append(mode_2_prod)

        elif type(G) == tuple:              # dynamic graph input: ((batch, K, N, N), (batch, K, N, N))
            assert (len(G) == 2) & (self.K == G[0].shape[-3] == G[1].shape[-3])
            for o in range(self.K):
                for d in range(self.K):
                    mode_1_prod = torch.einsum('bncl,bnm->bmcl', X, G[0][:, o, :, :])
                    mode_2_prod = torch.einsum('bmcl,bcd->bmdl', mode_1_prod, G[1][:, d, :, :])
                    feat_set.append(mode_2_prod)
        else:
            raise NotImplementedError

        _2D_feat = torch.cat(feat_set, dim=-1)
        mode_3_prod = torch.einsum('bmdk,kh->bmdh', _2D_feat, self.W)

        if self.use_bias:
            mode_3_prod += self.b
        H = self.activation(mode_3_prod) if self.activation is not None else mode_3_prod
        return H
