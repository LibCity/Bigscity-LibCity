from trafficdl.model.abstract_model import AbstractModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from trafficdl.model import loss
import math


class FilterLinear(nn.Module):
    def __init__(self, device, in_features, out_features, filter_square_matrix, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.filter_square_matrix = Variable(filter_square_matrix.to(device), requires_grad=False)

        self.weight = Parameter(torch.Tensor(out_features, in_features).to(device))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features).to(device))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.filter_square_matrix.matmul(self.weight), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')'


class TGCLSTM(AbstractModel):
    def __init__(self, config, data_feature):
        '''
        Args:
            K: K-hop graph
            A: adjacency matrix
            FFR: free-flow reachability matrix
            feature_size: the dimension of features
            Clamp_A: Boolean value, clamping all elements of A between 0. to 1.
        '''
        super(TGCLSTM, self).__init__(config, data_feature)
        self.gpu = config.get('gpu', True)
        self.data_feature = data_feature
        self.feature_size = data_feature['adj_mx'].shape[0]
        self.hidden_size = self.feature_size
        self.K = config.get('K_hop_numbers', 3)
        self.dataset_class = config.get('dataset_class', 'TrafficSpeedDataset')
        self.scaler_type = config.get('scaler', 'standard')
        self.device = config.get('device', torch.device('cpu'))

        self.A_list = []  # Adjacency Matrix List
        adj_mx = data_feature['adj_mx']

        adj_mx[adj_mx > 1e-4] = 1
        adj_mx[adj_mx <= 1e-4] = 0

        A = torch.FloatTensor(adj_mx).to(self.device)

        A_temp = torch.eye(self.feature_size, self.feature_size, device=self.device)
        for i in range(self.K):
            A_temp = torch.matmul(A_temp, A)
            if config.get('Clamp_A', True):
                # confine elements of A
                A_temp = torch.clamp(A_temp, max=1.)
            if self.dataset_class == "TGCLSTMDataset":
                self.A_list.append(
                    torch.mul(A_temp,
                              torch.Tensor(data_feature['FFR'][config.get('back_length', 3)]).to(self.device)))
            else:
                self.A_list.append(A_temp)

        # a length adjustable Module List for hosting all graph convolutions
        self.gc_list = nn.ModuleList(
            [FilterLinear(self.device, self.feature_size, self.feature_size, self.A_list[i], bias=False) for i in
             range(self.K)])

        hidden_size = self.feature_size
        input_size = self.feature_size * self.K

        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)

        # initialize the neighbor weight for the cell state
        self.Neighbor_weight = Parameter(torch.FloatTensor(self.feature_size).to(self.device))
        stdv = 1. / math.sqrt(self.feature_size)
        self.Neighbor_weight.data.uniform_(-stdv, stdv)

        self.output_last = config.get('output_last', True)
        self._scaler = self.data_feature.get('scaler')
        self.output_dim = config.get('output_dim', 1)

    def step(self, input, Hidden_State, Cell_State):
        x = input

        gc = self.gc_list[0](x)
        for i in range(1, self.K):
            gc = torch.cat((gc, self.gc_list[i](x)), 1)

        combined = torch.cat((gc, Hidden_State), 1)
        f = torch.sigmoid(self.fl(combined))
        i = torch.sigmoid(self.il(combined))
        o = torch.sigmoid(self.ol(combined))
        C = torch.tanh(self.Cl(combined))

        NC = torch.mul(Cell_State,
                       torch.mv(Variable(self.A_list[-1], requires_grad=False).to(self.device), self.Neighbor_weight))
        Cell_State = f * NC + i * C
        Hidden_State = o * torch.tanh(Cell_State)

        return Hidden_State, Cell_State, gc

    def Bi_torch(self, a):
        a[a < 0] = 0
        a[a > 0] = 1
        return a

    def forward(self, batch):
        inputs = batch['X']
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)

        outputs = None

        for i in range(time_step):
            Hidden_State, Cell_State, gc = self.step(torch.squeeze(inputs[:, i:i + 1, :]), Hidden_State, Cell_State)

            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)
        if self.output_last:
            return outputs[:, -1, :].unsqueeze(1).unsqueeze(-1)
        else:
            return outputs.unsqueeze(-1)

    def get_data_feature(self):
        return self.data_feature

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
            if y_.shape[3] < x_.shape[3]:  # y_的feature_dim可能小于x_的
                y_ = torch.cat([y_, y[:, i:i + 1, :, self.output_dim:]], dim=3)
            x_ = torch.cat([x_[:, 1:, :, :], y_], dim=1)
        y_preds = torch.cat(y_preds, dim=1)
        return y_preds

    def initHidden(self, batch_size):
        Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).to(self.device))
        Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).to(self.device))
        return Hidden_State, Cell_State

    def reinitHidden(self, batch_size, Hidden_State_data, Cell_State_data):
        Hidden_State = Variable(Hidden_State_data.to(self.device), requires_grad=True)
        Cell_State = Variable(Cell_State_data.to(self.device), requires_grad=True)
        return Hidden_State, Cell_State
