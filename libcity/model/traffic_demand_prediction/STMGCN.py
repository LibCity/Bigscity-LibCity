from torch import nn
import torch
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
from libcity.model import utils


class GCNNet(nn.Module):
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
        nn.init.xavier_normal_(self.W)
        if self.bias:
            self.b = nn.Parameter(torch.empty(self.hidden_dim), requires_grad=True)
            nn.init.constant_(self.b, val=b_init)

    def forward(self, A:torch.Tensor, x:torch.Tensor):
        '''
        Batch-wise graph convolution operation on given list of support adj matrices
        :param A: support adj matrices - torch.Tensor (K, n_nodes, n_nodes)
        :param x: graph feature/signal - torch.Tensor (batch_size, n_nodes, input_dim)
        :return: hidden representation - torch.Tensor (batch_size, n_nodes, hidden_dim)
        '''
        assert self.K == A.shape[0]

        support_list = list()
        for k in range(self.K):
            support = torch.einsum('ij,bjp->bip', [A[k,:,:], x])
            support_list.append(support)
        support_cat = torch.cat(support_list, dim=-1)

        output = torch.einsum('bip,pq->biq', [support_cat, self.W])
        if self.bias:
            output += self.b
        output = self.activation(output) if self.activation is not None else output
        return output

    def __repr__(self):
        return self.__class__.__name__ + f'({self.K} * input {self.input_dim} -> hidden {self.hidden_dim})'
class Adj_Preprocessor(object):
    def __init__(self, kernel_type:str, K:int):
        self.kernel_type = kernel_type
        # chebyshev (Defferard NIPS'16)/localpool (Kipf ICLR'17)/random_walk_diffusion (Li ICLR'18)
        self.K = K if self.kernel_type != 'localpool' else 1
        # max_chebyshev_polynomial_order (Defferard NIPS'16)/max_diffusion_step (Li ICLR'18)

    def process(self, adj:torch.Tensor):
        '''
        Generate adjacency matrices
        :param adj: input adj matrix - (N, N) torch.Tensor
        :return: processed adj matrix - (K_supports, N, N) torch.Tensor
        '''
        kernel_list = list()

        if self.kernel_type in ['localpool', 'chebyshev']:  # spectral
            adj_norm = self.symmetric_normalize(adj)
            # adj_norm = self.random_walk_normalize(adj)     # for asymmetric normalization
            if self.kernel_type == 'localpool':
                localpool = torch.eye(adj_norm.shape[0]) + adj_norm  # same as add self-loop first
                kernel_list.append(localpool)

            else:  # chebyshev
                laplacian_norm = torch.eye(adj_norm.shape[0]) - adj_norm
                rescaled_laplacian = self.rescale_laplacian(laplacian_norm)
                kernel_list = self.compute_chebyshev_polynomials(rescaled_laplacian, kernel_list)

        elif self.kernel_type == 'random_walk_diffusion':  # spatial

            # diffuse k steps on transition matrix P
            P_forward = self.random_walk_normalize(adj)
            kernel_list = self.compute_chebyshev_polynomials(P_forward.T, kernel_list)
            '''
            # diffuse k steps bidirectionally on transition matrix P
            P_forward = self.random_walk_normalize(adj)
            P_backward = self.random_walk_normalize(adj.T)
            forward_series, backward_series = [], []
            forward_series = self.compute_chebyshev_polynomials(P_forward.T, forward_series)
            backward_series = self.compute_chebyshev_polynomials(P_backward.T, backward_series)
            kernel_list += forward_series + backward_series[1:]  # 0-order Chebyshev polynomial is same: I
            '''
        else:
            raise ValueError('Invalid kernel_type. Must be one of [chebyshev, localpool, random_walk_diffusion].')

        # print(f"Minibatch {b}: {self.kernel_type} kernel has {len(kernel_list)} support kernels.")
        kernels = torch.stack(kernel_list, dim=0)

        return kernels

    @staticmethod
    def random_walk_normalize(A):   # asymmetric
        d_inv = torch.pow(A.sum(dim=1), -1)   # OD matrix Ai,j sum on j (axis=1)
        d_inv[torch.isinf(d_inv)] = 0.
        D = torch.diag(d_inv)
        A_norm = torch.mm(D, A)
        return A_norm

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
            print("Eigen_value calculation didn't converge, using max_eigen_val=2 instead.")
            lambda_max = 2
        L_rescale = (2 / lambda_max) * L - torch.eye(L.shape[0])
        return L_rescale

    def compute_chebyshev_polynomials(self, x, T_k):
        # compute Chebyshev polynomials up to order k. Return a list of matrices.
        # print(f"Computing Chebyshev polynomials up to order {self.K}.")
        for k in range(self.K + 1):
            if k == 0:
                T_k.append(torch.eye(x.shape[0]))
            elif k == 1:
                T_k.append(x)
            else:
                T_k.append(2 * torch.mm(x, T_k[k-1]) - T_k[k-2])
        return T_k



class CG_LSTM_Net(nn.Module):
    def __init__(self, seq_len:int, n_nodes:int, input_dim:int,
                 lstm_hidden_dim: int, lstm_num_layers: int,
                 K:int, gconv_use_bias:bool, gconv_activation=nn.ReLU):
        super().__init__()
        self.seq_len = seq_len
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers

        self.gconv_temporal_feats = GCNNet(K=K, input_dim=seq_len, hidden_dim=seq_len,
                                           bias=gconv_use_bias, activation=gconv_activation)
        self.fc = nn.Linear(in_features=seq_len, out_features=seq_len, bias=True)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim,
                            num_layers=lstm_num_layers, batch_first=True)



    def forward(self, adj:torch.Tensor, obs_seq:torch.Tensor, hidden:tuple):
        '''
        Context Gated LSTM:
            1. temporal obs_seq as feature for region, convolve neighbors on adj
            2. global pool -> FC -> FC -> temporal obs weights
            3. re-weighted obs_seq -> global-shared LSTM
        :param adj: support adj matrices for collecting neighbors - torch.Tensor (K, n_nodes, n_nodes)
        :param obs_seq: observation sequence - torch.Tensor (batch_size, seq_len, n_nodes, n_feats)
        :param hidden: tuple of hidden states (h, c) - torch.Tensor (n_layers, batch_size*n_nodes, hidden_dim) x2
        :return:
        '''
        # TODO
        obs_seq = obs_seq[..., None]
        batch_size = obs_seq.shape[0]
        x_seq = obs_seq.sum(dim=-1)     # sum up feature dimension: default 1

        # channel-wise attention on timestep
        x_seq = x_seq.permute(0, 2, 1)
        x_seq_gconv = self.gconv_temporal_feats(A=adj, x=x_seq)
        x_hat = torch.add(x_seq, x_seq_gconv)       # eq. 6
        z_t = x_hat.sum(dim=1)/x_hat.shape[1]       # eq. 7
        s = torch.sigmoid(self.fc(torch.relu(self.fc(z_t))))    # eq. 8
        obs_seq_reweighted = torch.einsum('btnf,bt->btnf', [obs_seq, s])      # eq. 9

        # global-shared LSTM
        shared_seq = obs_seq_reweighted.permute(0, 2, 1, 3).reshape(batch_size*self.n_nodes, self.seq_len, self.input_dim)
        x, hidden = self.lstm(shared_seq, hidden)

        output = x[:, -1, :].reshape(batch_size, self.n_nodes, self.lstm_hidden_dim)
        return output, hidden

    def init_hidden(self, batch_size:int):
        weight = next(self.parameters()).data
        hidden = (weight.new_zeros(self.lstm_num_layers, batch_size * self.n_nodes, self.lstm_hidden_dim),
                  weight.new_zeros(self.lstm_num_layers, batch_size * self.n_nodes, self.lstm_hidden_dim))
        return hidden



class ST_MGCN_Net(nn.Module):
    def __init__(self, M:int, seq_len:int, n_nodes:int, input_dim:int, lstm_hidden_dim:int, lstm_num_layers:int,
                 gcn_hidden_dim:int, sta_kernel_config:dict, gconv_use_bias:bool, gconv_activation=nn.ReLU):
        super().__init__()
        self.M = M
        self.sta_K = self.get_support_K(sta_kernel_config)

        # initiate one pair of CG_LSTM & GCN for each adj input
        self.rnn_list, self.gcn_list = nn.ModuleList(), nn.ModuleList()
        for m in range(self.M):
            cglstm = CG_LSTM_Net(seq_len=seq_len, n_nodes=n_nodes, input_dim=input_dim,
                                 lstm_hidden_dim=lstm_hidden_dim, lstm_num_layers=lstm_num_layers,
                                 K=self.sta_K, gconv_use_bias=gconv_use_bias, gconv_activation=gconv_activation)
            self.rnn_list.append(cglstm)
            gcn = GCNNet(K=self.sta_K, input_dim=lstm_hidden_dim, hidden_dim=gcn_hidden_dim,
                         bias=gconv_use_bias, activation=gconv_activation)
            self.gcn_list.append(gcn)
        self.fc = nn.Linear(in_features=gcn_hidden_dim, out_features=input_dim, bias=True)

    @staticmethod
    def get_support_K(config:dict):
        if config['kernel_type'] == 'localpool':
            assert config['K'] == 1
            K = 1
        elif config['kernel_type'] == 'chebyshev':
            K = config['K'] + 1
        elif config['kernel_type'] == 'random_walk_diffusion':
            K = config['K'] * 2 + 1
        else:
            raise ValueError('Invalid kernel_type. Must be one of [chebyshev, localpool, random_walk_diffusion].')
        return K

    def init_hidden_list(self, batch_size:int):
        hidden_list = list()
        for m in range(self.M):
            hidden = self.rnn_list[m].init_hidden(batch_size)
            hidden_list.append(hidden)
        return hidden_list

    def forward(self, obs_seq:torch.Tensor, sta_adj_list:list):
        '''
        On each graph do CG_LSTM + GCN -> sum -> fc output
        :param obs_seq: observation sequence - torch.Tensor (batch_size, seq_len, n_nodes, n_feats)
        :param sta_adj_list: [(K_supports, N, N)] * M_sta
        :return: y_pred (t+1) - torch.Tensor (batch_size, n_nodes, n_feats)
        '''
        assert len(sta_adj_list) == self.M
        batch_size = obs_seq.shape[0]
        hidden_list = self.init_hidden_list(batch_size)

        feat_list = list()
        for m in range(self.M):
            cg_rnn_out, hidden_list[m] = self.rnn_list[m](sta_adj_list[m], obs_seq, hidden_list[m])
            gcn_out = self.gcn_list[m](sta_adj_list[m], cg_rnn_out)
            feat_list.append(gcn_out)
        feat_fusion = torch.sum(torch.stack(feat_list, dim=-1), dim=-1)     # aggregation

        output = self.fc(feat_fusion)
        return output[:,:,0]



class STMGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        M_adj,obs_len, n_nodes, num_input_features, lstm_hidden_dim, lstm_hidden_layers, gcn_hidden_dim, sta_kernel_config, gconv_use_bias, gconv_activation, num_output_classes = self.get_input(config, data_feature)

        model = ST_MGCN_Net(M=M_adj, seq_len=sum(obs_len), n_nodes=30, input_dim=1, lstm_hidden_dim=64, lstm_num_layers=3,
                               gcn_hidden_dim=64, sta_kernel_config=sta_kernel_config, gconv_use_bias=True, gconv_activation=nn.ReLU)
        self.device = config.get('device')
        # model = model.to(self.device)


    def get_input(self,config,data_feature):
        M_adj = data_feature.get('M_adj',3)
        obs_len = tuple(3,1,1)
        n_nodes = data_feature.get('n_nodes',30)
        num_input_features = data_feature.get('feature_dim', 1)
        lstm_hidden_dim = data_feature.get("lstm_hidden_dim",64)
        lstm_hidden_layers = data_feature.get("lstm_hidden_layers", 3)
        gcn_hidden_dim = data_feature.get("gcn_hidden_dim",64)
        sta_kernel_config = config.get("sta_kernel_config",{'kernel_type':'chebyshev', 'K':2})
        gconv_use_bias = config.get("gconv_use_bias",True)
        gconv_activation = config.get("gconv_activation",nn.ReLU)
        num_output_classes = config.get('output_dim', 8)

        return M_adj,obs_len,n_nodes,num_input_features,lstm_hidden_dim,lstm_hidden_layers,gcn_hidden_dim,sta_kernel_config,gconv_use_bias,gconv_activation,num_output_classes

    def predict(self, batch):
        pass

    def calculate_loss(self, batch):
        pass