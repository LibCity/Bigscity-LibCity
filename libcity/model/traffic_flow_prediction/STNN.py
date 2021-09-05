import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from logging import getLogger
import torch
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class MLP(nn.Module):
    def __init__(self, ninp, nhid, nout, nlayers, dropout):
        super(MLP, self).__init__()
        self.ninp = ninp
        # modules
        if nlayers == 1:
            self.module = nn.Linear(ninp, nout)
        else:
            modules = [nn.Linear(ninp, nhid), nn.ReLU(), nn.Dropout(dropout)]
            nlayers -= 1
            while nlayers > 1:
                modules += [nn.Linear(nhid, nhid), nn.ReLU(), nn.Dropout(dropout)]
                nlayers -= 1
            modules.append(nn.Linear(nhid, nout))
            self.module = nn.Sequential(*modules)

    def forward(self, input):
        return self.module(input)


class STNN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._scaler = self.data_feature.get('scaler')  # 用于数据归一化
        self.adj_mx = self.data_feature.get('adj_mx', 1)  # 邻接矩阵
        self.num_nodes = self.data_feature.get('num_nodes', 1)  # 网格个数
        self.feature_dim = self.data_feature.get('feature_dim', 1)  # 输入维度
        self.output_dim = self.data_feature.get('output_dim', 1)  # 输出维度

        self._logger = getLogger()

        self.device = config.get('device', torch.device('cpu'))

        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)

        self.mode = config.get('mode', 'refine')
        nhid = config.get("nhid", 0)
        nlayers = config.get("nlayers", 1)
        dropout_f = config.get("dropout_f", 0.1)
        dropout_d = config.get("dropout_d", 0.1)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

        relations = torch.Tensor(self.adj_mx).unsqueeze(1)
        # kernel
        self.activation = torch.tanh
        device = self.device
        if self.mode is None or self.mode == 'refine':
            self.relations = torch.cat((torch.eye(self.num_nodes).unsqueeze(1), relations), 1)
        elif self.mode == 'discover':
            self.relations = torch.cat((torch.eye(self.num_nodes).unsqueeze(1),
                                        torch.ones(self.num_nodes, 1, self.num_nodes).to(device)), 1)
        self.nr = self.relations.size(1)
        # modules
        self.drop = nn.Dropout(dropout_f)
        self.factors = nn.Parameter(torch.Tensor(self.input_window, self.num_nodes, self.feature_dim))
        self.sigmo = nn.Sigmoid()
        self.ffunc = nn.Linear(self.input_window * self.num_nodes * self.feature_dim, 2 * self.input_window * self.num_nodes * self.feature_dim)

        self.dynamic = MLP(self.feature_dim * self.nr, nhid, self.feature_dim, nlayers, dropout_d)
        self.decoder = nn.Linear(self.feature_dim * self.input_window, self.output_window * self.output_dim, bias=False)
        if self.mode == 'refine':
            self.relations.data = self.relations.data.ceil().clamp(0, 1).bool()
            self.rel_weights = nn.Parameter(torch.Tensor(self.relations.sum().item() - self.num_nodes))
        elif self.mode == 'discover':
            self.rel_weights = nn.Parameter(torch.Tensor(self.num_nodes, 1, self.num_nodes))
        # init
        self._init_weights()

    def _init_weights(self):
        self.factors.data.uniform_(-0.1, 0.1)
        if self.mode == 'refine':
            self.rel_weights.data.fill_(0.5)
        elif self.mode == 'discover':
            self.rel_weights.data.fill_(1 / self.num_nodes)

    def get_relations(self):
        if self.mode is None:
            return self.relations
        else:
            weights = F.hardtanh(self.rel_weights, 0, 1)
            if self.mode == 'refine':
                intra = self.rel_weights.new(self.num_nodes, self.num_nodes).copy_(self.relations[:, 0]).unsqueeze(1)
                inter = self.rel_weights.new_zeros(self.num_nodes, self.nr - 1, self.num_nodes)
                inter.masked_scatter_(self.relations[:, 1:], weights)
            if self.mode == 'discover':
                intra = self.relations[:, 0].unsqueeze(1)
                inter = weights
            return torch.cat((intra, inter), 1)

    def forward(self, batch):
        x = torch.Tensor(batch['X'])  # shape = (batch_size, input_length, ..., feature_dim)
        x_size = x.shape
        nowrel = self.get_relations()
        nowrel_size = nowrel.shape
        nowrel = nowrel.repeat(self.input_window, 1, 1).expand(
            x_size[0], nowrel_size[0] * self.input_window, nowrel_size[1], nowrel_size[2])  # 64-12*41-2-41

        nowrel = nowrel.contiguous().view(
            x_size[0] * self.input_window * self.num_nodes, nowrel_size[1], nowrel_size[2])  # 64*12*4-2-41
        z_inf = x.repeat(1, self.num_nodes, 1, 1).view(
            x_size[0] * self.input_window * self.num_nodes, self.num_nodes, self.feature_dim)  # 64-12*41-41-1
        z_context = nowrel.matmul(z_inf)  # 64*12*41-2-1
        z_gen = self.dynamic(z_context.view(-1, self.nr * self.feature_dim))

        return self.activation(z_gen.view(x.shape))

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        res = loss.masked_mae_torch(y_predicted, y_true)
        return res

    def predict(self, batch):
        x = torch.Tensor(batch['X'])  # shape = (batch_size, input_length, ..., feature_dim)
        x_size = x.shape

        # step one:Xt to Zt
        x_stepone = self.ffunc(x.view(x_size[0], self.input_window * self.num_nodes * self.feature_dim))
        x_steptwo = self.sigmo(x_stepone.view(x_size[0], self.input_window, self.num_nodes, self.feature_dim * 2))
        z_inf = self.drop(
            self.factors[(x_steptwo[:, :, :, 0] * 11).ceil().long(), (x_steptwo[:, :, :, 1] * 40).ceil().long()])

        batch['X'] = z_inf.view(x_size)
        # step two:Zt to Zt+12
        for i in range(self.output_window):
            z_next = self.forward(batch)
            batch['X'] = z_next
        z_inf = batch['X']
        # step three: Zt+12 to Y
        x_rec = self.decoder(z_inf.view(-1, self.feature_dim * self.input_window))
        return x_rec.view((-1, self.output_window, self.num_nodes, self.output_dim))
