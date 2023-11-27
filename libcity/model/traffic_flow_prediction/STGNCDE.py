import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
from libcity.model.controldiffeq import NaturalCubicSpline, cdeint_gde_dev

class FinalTanh_f(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_f, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        
        self.linears = nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        # z: torch.Size([64, 207, 32])
        # self.linear_out(z): torch.Size([64, 207, 64])
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)    
        z = z.tanh()
        return z

class VectorField_g(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k, embed_dim,
                    g_type):
        super(VectorField_g, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        
        # self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        #                                    for _ in range(num_hidden_layers - 1))

        #FIXME:
        # self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        
        self.g_type = g_type
        if self.g_type == 'agc':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))


    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        if self.g_type == 'agc':
            z = self.agc(z)
        else:
            raise ValueError('Check g_type argument')
        # for linear in self.linears:
        #     z = linear(x_gconv)
        #     z = z.relu()

        #FIXME:
        # z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.hidden_channels)
        z = z.tanh()
        return z #torch.Size([64, 307, 64, 1])

    def agc(self, z):
        """
        Adaptive Graph Convolution
        - Node Adaptive Parameter Learning
        - Data Adaptive Graph Generation
        """
        node_num = self.node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        # laplacian=False
        laplacian=False
        if laplacian == True:
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            support_set = [supports, -torch.eye(node_num).to(supports.device)]
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            # support_set = [-supports]
        else:
            support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(self.node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, z)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        z = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return z

class STG_NCDE(AbstractTrafficStateModel):
    # def __init__(self, args, func_f, func_g, input_channels, hidden_channels, output_channels, initial, device, atol, rtol, solver):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

         # section 1: data_feature
        self._scaler = self.data_feature.get('scaler')
        self.num_node = self.data_feature.get('num_nodes', 1)
        self.input_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)

        self._logger = getLogger()
        # section 2: model config 
        # self.input_window = config.get('input_window', 1)
        # self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))

        # self.hidden_size = config.get('hidden_size', 64)
        # self.num_layers = config.get('num_layers', 1)
        # self.dropout = config.get('dropout', 0)
        # section 3: model structure
        # self.rnn = nn.LSTM(input_size=self.num_nodes * self.feature_dim, hidden_size=self.hidden_size,
        #                    num_layers=self.num_layers, dropout=self.dropout)
        # self.fc = nn.Linear(self.hidden_size, self.num_nodes * self.output_dim)


        # self.num_node = args.num_nodes
        # self.input_dim = input_channels
        self.hidden_dim = config.get('hidden_dim', 1)
        self.hid_hid_dim = config.get('hid_hid_dim', 1)
        # self.output_dim = output_channels
        self.horizon = config.get('horizon', 1)
        self.num_layers = config.get('num_layers', 1)

        self.default_graph = config.get('default_graph', True)
        self.embed_dim = config.get('embed_dim', 1)
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)
        
        self.func_f = FinalTanh_f(input_channels=self.input_dim, hidden_channels=self.hidden_dim,
                                        hidden_hidden_channels=self.hid_hid_dim,
                                        num_hidden_layers=self.num_layers)
        self.func_g = VectorField_g(input_channels=self.input_dim, hidden_channels=self.hidden_dim,
                                        hidden_hidden_channels=self.hid_hid_dim,
                                        num_hidden_layers=self.num_layers, num_nodes=self.num_node, cheb_k=2, embed_dim=self.embed_dim,
                                        g_type=config.get('g_type', 'agc'))
        self.solver = 'rk4'
        self.atol = 1e-9
        self.rtol = 1e-7

        #predictor
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.init_type = 'fc'
        if self.init_type == 'fc':
            self.initial_h = torch.nn.Linear(self.input_dim, self.hidden_dim)
            self.initial_z = torch.nn.Linear(self.input_dim, self.hidden_dim)
        elif self.init_type == 'conv':
            self.start_conv_h = nn.Conv2d(in_channels=self.input_dim,
                                            out_channels=self.hidden_dim,
                                            kernel_size=(1,1))
            self.start_conv_z = nn.Conv2d(in_channels=self.input_dim,
                                            out_channels=self.hidden_dim,
                                            kernel_size=(1,1))

    def printDevice(self, a):
        self._logger.info(str(a)+":"+str(a.device))
    
    def predict(self, batch):
        times = torch.linspace(0,11,12)
        *coeffs, target = batch
        times = times.to(self.device)
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        spline = NaturalCubicSpline(times, coeffs)
        if self.init_type == 'fc':
            h0 = self.initial_h(spline.evaluate(times[0]))
            z0 = self.initial_z(spline.evaluate(times[0]))
        elif self.init_type == 'conv':
            h0 = self.start_conv_h(spline.evaluate(times[0]).transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze()
            z0 = self.start_conv_z(spline.evaluate(times[0]).transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze()
        z_t = cdeint_gde_dev(dX_dt=spline.derivative, #dh_dt
                                   h0=h0,
                                   z0=z0,
                                   func_f=self.func_f,
                                   func_g=self.func_g,
                                   t=times,
                                   method=self.solver,
                                   atol=self.atol,
                                   rtol=self.rtol)

        # init_state = self.encoder.init_hidden(source.shape[0])
        # output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        z_T = z_t[-1:,...].transpose(0,1)

        #CNN based predictor
        output = self.end_conv(z_T)                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output.to(self.device)
    
    def calculate_loss(self, batch):
        *coeffs, y_true = batch  # ground-truth value
        y_predicted = self.predict(batch)  # prediction results
        # denormalization the value
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        loss = torch.nn.L1Loss().to(self.device)
        
        return loss(y_predicted, y_true)
        # return loss.masked_mae_loss(y_pred=y_predicted, y_true=y_true)