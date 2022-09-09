import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from libcity.model import loss
import math
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class FilterLinear(nn.Module):
    def __init__(self, device, input_dim, output_dim, in_features, out_features, filter_square_matrix, bias=True):
        """
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        """
        super(FilterLinear, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features

        self.num_nodes = filter_square_matrix.shape[0]
        self.filter_square_matrix = Variable(filter_square_matrix.repeat(output_dim, input_dim).to(device),
                                             requires_grad=False)

        self.weight = Parameter(torch.Tensor(out_features, in_features).to(device))  # [out_features, in_features]
        if bias:
            self.bias = Parameter(torch.Tensor(out_features).to(device))  # [out_features]
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')'


class TGCLSTM(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super(TGCLSTM, self).__init__(config, data_feature)
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.input_dim = self.data_feature.get('feature_dim', 1)
        self.in_features = self.input_dim * self.num_nodes
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.out_features = self.output_dim * self.num_nodes

        self.K = config.get('K_hop_numbers', 3)
        self.back_length = config.get('back_length', 3)
        self.dataset_class = config.get('dataset_class', 'TrafficSpeedDataset')
        self.device = config.get('device', torch.device('cpu'))
        self._scaler = self.data_feature.get('scaler')

        self.A_list = []  # Adjacency Matrix List
        adj_mx = data_feature['adj_mx']
        if self.dataset_class == "TGCLSTMDataset":
            adj_mx[adj_mx > 1e-4] = 1
            adj_mx[adj_mx <= 1e-4] = 0
        adj = torch.FloatTensor(adj_mx).to(self.device)
        adj_temp = torch.eye(self.num_nodes, self.num_nodes, device=self.device)
        for i in range(self.K):
            adj_temp = torch.matmul(adj_temp, adj)
            if config.get('Clamp_A', True):
                # confine elements of A
                adj_temp = torch.clamp(adj_temp, max=1.)
            if self.dataset_class == "TGCLSTMDataset":
                self.A_list.append(
                    torch.mul(adj_temp, torch.Tensor(data_feature['FFR'][self.back_length])
                              .to(self.device)))
            else:
                self.A_list.append(adj_temp)

        # a length adjustable Module List for hosting all graph convolutions
        self.gc_list = nn.ModuleList([FilterLinear(self.device, self.input_dim, self.output_dim, self.in_features,
                                                   self.out_features, self.A_list[i], bias=False) for i in
                                      range(self.K)])

        hidden_size = self.out_features
        input_size = self.out_features * self.K

        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)

        # initialize the neighbor weight for the cell state
        self.Neighbor_weight = Parameter(torch.FloatTensor(self.out_features, self.out_features).to(self.device))
        stdv = 1. / math.sqrt(self.out_features)
        self.Neighbor_weight.data.uniform_(-stdv, stdv)

    def step(self, step_input, hidden_state, cell_state):
        x = step_input  # [batch_size, in_features]

        gc = self.gc_list[0](x)  # [batch_size, out_features]
        for i in range(1, self.K):
            gc = torch.cat((gc, self.gc_list[i](x)), 1)  # [batch_size, out_features * K]

        combined = torch.cat((gc, hidden_state), 1)  # [batch_size, out_features * (K+1)]
        # fl: nn.linear(out_features * (K+1), out_features)
        f = torch.sigmoid(self.fl(combined))
        i = torch.sigmoid(self.il(combined))
        o = torch.sigmoid(self.ol(combined))
        c_ = torch.tanh(self.Cl(combined))

        nc = torch.matmul(cell_state, torch.mul(
            Variable(self.A_list[-1].repeat(self.output_dim, self.output_dim), requires_grad=False).to(self.device),
            self.Neighbor_weight))

        cell_state = f * nc + i * c_  # [batch_size, out_features]
        hidden_state = o * torch.tanh(cell_state)  # [batch_size, out_features]

        return hidden_state, cell_state, gc

    def bi_torch(self, a):
        a[a < 0] = 0
        a[a > 0] = 1
        return a

    def forward(self, batch):
        inputs = batch['X']  # [batch_size,  input_window, num_nodes, input_dim]
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        hidden_state, cell_state = self.init_hidden(batch_size)  # [batch_size, out_features]

        outputs = None

        for i in range(time_step):
            step_input = torch.squeeze(torch.transpose(inputs[:, i:i + 1, :, :], 2, 3)).reshape(batch_size, -1)
            hidden_state, cell_state, gc = self.step(step_input, hidden_state, cell_state)
            # gc: [batch_size, out_features * K]
            if outputs is None:
                outputs = hidden_state.unsqueeze(1)  # [batch_size, 1, out_features]
            else:
                outputs = torch.cat((outputs, hidden_state.unsqueeze(1)), 1)  # [batch_size, input_window, out_features]
        output = torch.transpose(torch.squeeze(outputs[:, -1, :]).reshape(batch_size, self.output_dim, self.num_nodes),
                                 1, 2).unsqueeze(1)
        return output  # [batch_size, 1, num_nodes, out_dim]

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true)

    def predict(self, batch):
        x = batch['X']
        y = batch['y']
        output_length = y.shape[1]
        y_preds = []
        x_ = x.clone()
        for i in range(output_length):
            batch_tmp = {'X': x_}
            y_ = self.forward(batch_tmp)
            y_preds.append(y_.clone())
            if y_.shape[3] < x_.shape[3]:
                y_ = torch.cat([y_, y[:, i:i + 1, :, self.output_dim:]], dim=3)
            x_ = torch.cat([x_[:, 1:, :, :], y_], dim=1)
        y_preds = torch.cat(y_preds, dim=1)  # [batch_size, output_window, batch_size, output_dim]
        return y_preds

    def init_hidden(self, batch_size):
        hidden_state = Variable(torch.zeros(batch_size, self.out_features).to(self.device))
        cell_state = Variable(torch.zeros(batch_size, self.out_features).to(self.device))
        return hidden_state, cell_state
