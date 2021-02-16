import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from logging import getLogger
from trafficdl.model.abstract_model import AbstractModel
from trafficdl.model import loss
import sys

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str, gpu: bool):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type
        self._device = torch.device("cuda:0" if gpu else "cpu")

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=self._device))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=self._device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class TGCNCell(nn.Module):
    def __init__(self, num_units, adj_mx, num_nodes, gpu, input_dim = 1):
        """
        :param num_units:
        :param adj_mx:
        :param num_nodes:
        """
        #----------------------初始化参数---------------------------#
        super().__init__()
        self.num_units = num_units
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self._device = torch.device("cuda" if gpu else "cpu")
        self.act = torch.tanh

        # 这里提前构建好拉普拉斯, 但是不确定是否需要 _build_函数
        support = calculate_normalized_laplacian(adj_mx)
        self.normalized_adj = self._build_sparse_matrix(support, self._device)

        self._gconv_params = LayerParams(self, 'gconv', gpu)

        self.init_params()

    def init_params(self,bias_start = 0.0):
        input_size = self.input_dim + self.num_units
        weight_0 = torch.nn.Parameter(torch.empty((input_size, 2*self.num_units) , device=self._device))
        bias_0 = torch.nn.Parameter(torch.empty(2*self.num_units, device=self._device))
        weight_1 = torch.nn.Parameter(torch.empty((input_size, self.num_units), device=self._device))
        bias_1 = torch.nn.Parameter(torch.empty(self.num_units, device=self._device))

        torch.nn.init.xavier_normal_(weight_0)
        torch.nn.init.xavier_normal_(weight_1)
        torch.nn.init.constant_(bias_0, bias_start)
        torch.nn.init.constant_(bias_1, bias_start)

        self.register_parameter(name= 'weights_0',param= weight_0)
        self.register_parameter(name= 'weights_1',param= weight_1)
        self.register_parameter(name= 'bias_0', param= bias_0)
        self.register_parameter(name= 'bias_1', param= bias_1)

        self.weigts = {weight_0.shape: weight_0, weight_1.shape : weight_1}
        self.biases = {bias_0.shape : bias_0, bias_1.shape : bias_1}


    @staticmethod
    def _build_sparse_matrix(L, device):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)
        return L

    def forward(self, inputs, state):
        """ Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs:  # shape `(batch, self.num_nodes * self.dim)`
        :param state:  # shape `(batch, self.num_nodes * self.gru_units)`
                       default= None
        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * gru_units)`.
        """
        output_size = 2 * self.num_units
        value = torch.sigmoid(self._gc(inputs, state, output_size, bias_start=1.0))   # (batch_size, self.num_nodes, output_size)
        r,u = torch.split(tensor=value, split_size_or_sections=self.num_units, dim=-1)
        r = torch.reshape(r, (-1, self.num_nodes * self.num_units))    # (batch_size, self.num_nodes * self.gru_units)
        u = torch.reshape(u, (-1, self.num_nodes * self.num_units))


        c = self.act(self._gc(inputs, r * state, self.num_units))
        c = c.reshape(shape= (-1, self.num_nodes * self.num_units))
        new_state = u * state + (1.0 - u) * c
        return new_state


    def _gc(self, inputs, state, output_size, bias_start=0.0):
        """
        :param inputs:  # shape `(batch, self.num_nodes * self.dim)`
        :param state:  # shape `(batch, self.num_nodes * self.gru_units)`
                       default= None
        :return shape `(B, num_nodes , output_size)`.
        """

        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self.num_nodes, -1))    #  (batch, self.num_nodes, self.dim)
        state  = torch.reshape(state, (batch_size, self.num_nodes, -1))     #  (batch, self.num_nodes, self.gru_units)
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.shape[2]

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)   # (num_nodes, dim+gru_units, batch)
        x0 = x0.reshape(shape=(self.num_nodes, -1))


        x1 = torch.sparse.mm(self.normalized_adj.float(), x0.float())    # A * X

        x1 = x1.reshape(shape= (self.num_nodes, input_size, batch_size))
        x1 = x1.permute(2, 0, 1)                    #(batch_size, self.num_nodes, input_size)
        x1 = x1.reshape(shape= (-1, input_size))    #(batch_size * self.num_nodes, input_size)

        #weights = self._gconv_params.get_weights((input_size, output_size))
        weights = self.weigts[(input_size, output_size)]
        x1 = torch.matmul(x1, weights)             #(batch_size * self.num_nodes, output_size)

        #biases = self._gconv_params.get_biases(output_size, bias_start)
        biases = self.biases[(output_size,)]
        x1 += biases

        x1 = x1.reshape(shape=(batch_size, self.num_nodes, output_size))
        return x1


#!!!!!!!!!!!!!!!这里将来要改成AbstractModel
class TGCN(AbstractModel):
    def __init__(self, config, data_feature):

        self.data_feature = data_feature
        self.adj_mx = self.data_feature.get('adj_mx')
        self.num_nodes = self.data_feature.get('num_nodes',1)
        config['num_nodes'] = self.num_nodes
        self.input_dim = self.data_feature.get('feature_dim',1)
        self.output_dim = config.get('output_dim', 1)
        self.gru_units = int(config.get('rnn_units', 64))
        self.lam = config.get('lambda',0.0015)

        #super().__init__(config, data_feature)
        super().__init__(config,data_feature)



        #虽然不知道啥用但是放上总是好的
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.device = torch.device("cuda" if config.get('gpu', True) and torch.cuda.is_available() else "cpu")
        self._logger = getLogger()

        # -------------------构造模型-----------------------------
        self.tgcn_model = TGCNCell(self.gru_units, self.adj_mx, self.num_nodes, config['gpu'], self.input_dim)
        self.output_model = nn.Linear(self.gru_units, self.output_window)

    def forward(self,batch):
        """
        :param inputs: shape (batch_size, input_window, num_nodes, input_dim)
        :param labels: shape (batch_size, output_window, num_nodes, output_dim)
        :return: output: (batch_size, self.output_window, self.num_nodes, self.output_dim)
        """
        inputs = batch['X'] if torch.is_tensor(batch['X']) == True else torch.tensor(batch['X'])
        labels = batch['y'] if torch.is_tensor(batch['y']) == True else torch.tensor(batch['y'])
# 这里修改过!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        batch_size, input_window, num_nodes, input_dim = inputs.shape


        inputs = inputs.permute(1, 0, 2, 3)  # (input_window, batch_size, num_nodes, input_dim)
        inputs = inputs.view(self.input_window, batch_size, num_nodes * input_dim).to(self.device)

        state = torch.zeros(batch_size, self.num_nodes * self.gru_units).to(self.device)
        for t in range(input_window):
            state = self.tgcn_model(inputs[t], state)


        state = state.view(batch_size, self.num_nodes, self.gru_units)  # (batch_size, self.num_nodes, self.gru_units)
        output = self.output_model(state)       # (batch_size, self.num_nodes, self.output_window)
        output = output.unsqueeze(-1)             # (batch_size, self.num_nodes, self.output_window, 1)
        output = output.permute(0, 2, 1, 3).to(self.device)
        return output

    def get_data_feature(self):
        return self.data_feature

    def calculate_loss(self, batch):
        lam = self.lam
        Lreg = sum((torch.norm(param) ** 2 / 2) for param in self.parameters()    )

        labels = batch['y'] if torch.is_tensor(batch['y']) == True else torch.tensor(batch['y'])
        y_pred = self.predict(batch)
        y_pred = torch.reshape(y_pred[..., :self.output_dim], [-1, self.num_nodes])
        y_true = torch.reshape(labels[..., :self.output_dim], [-1, self.num_nodes])
        loss = torch.mean(torch.norm(y_true - y_pred) ** 2 / 2) + lam * Lreg
        #loss /= y_pred.numel()
        return loss

    def predict(self, batch):
        return self.forward(batch)
