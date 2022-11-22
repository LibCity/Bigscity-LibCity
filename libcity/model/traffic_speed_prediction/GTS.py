from logging import getLogger
import torch
import numpy as np
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from torch.nn import functional as F
import torch.nn as nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cosine_similarity_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def sample_gumbel(device, shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(device, logits, temperature, eps=1e-10):
    sample = sample_gumbel(device, logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(device, logits, temperature, hard=False, eps=1e-10):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y_soft = gumbel_softmax_sample(device, logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


class GCONV(nn.Module):
    def __init__(self, num_nodes, max_diffusion_step, device, input_dim, hid_dim, output_dim, bias_start=0.0):
        super().__init__()
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._device = device
        self._num_matrices = self._max_diffusion_step + 1  # Ks
        self._output_dim = output_dim
        input_size = input_dim + hid_dim
        shape = (input_size * self._num_matrices, self._output_dim)
        self.weight = torch.nn.Parameter(torch.empty(*shape, device=self._device))
        self.biases = torch.nn.Parameter(torch.empty(self._output_dim, device=self._device))
        torch.nn.init.xavier_normal_(self.weight)
        torch.nn.init.constant_(self.biases, bias_start)

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, inputs, state, adj_mx):
        # 对X(t)和H(t-1)做图卷积，并加偏置bias
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        # (batch_size, num_nodes, total_arg_size(input_dim+state_dim))
        input_size = inputs_and_state.size(2)  # =total_arg_size

        x = inputs_and_state
        # T0=I x0=T0*x=x
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)  # (1, num_nodes, total_arg_size * batch_size)

        # 3阶[T0,T1,T2]Chebyshev多项式近似g(theta)
        # 把图卷积公式中的~L替换成了随机游走拉普拉斯D^(-1)*W
        if self._max_diffusion_step == 0:
            pass
        else:
            # T1=L x1=T1*x=L*x
            x1 = torch.sparse.mm(adj_mx, x0)  # supports: n*n; x0: n*(total_arg_size * batch_size)
            x = self._concat(x, x1)  # (2, num_nodes, total_arg_size * batch_size)
            for k in range(2, self._max_diffusion_step + 1):
                # T2=2LT1-T0=2L^2-1 x2=T2*x=2L^2x-x=2L*x1-x0...
                # T3=2LT2-T1=2L(2L^2-1)-L x3=2L*x2-x1...
                x2 = 2 * torch.sparse.mm(adj_mx, x1) - x0
                x = self._concat(x, x2)  # (3, num_nodes, total_arg_size * batch_size)
                x1, x0 = x2, x1  # 循环
        # x.shape (Ks, num_nodes, total_arg_size * batch_size)
        # Ks = len(supports) * self._max_diffusion_step + 1
        x = torch.reshape(x, shape=[self._num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, num_matrices)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * self._num_matrices])
        x = torch.matmul(x, self.weight)  # (batch_size * self._num_nodes, self._output_dim)
        x += self.biases
        # Reshape res back to 2D: (batch_size * num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * self._output_dim])


class FC(nn.Module):
    def __init__(self, num_nodes, device, input_dim, hid_dim, output_dim, bias_start=0.0):
        super().__init__()
        self._num_nodes = num_nodes
        self._device = device
        self._output_dim = output_dim
        input_size = input_dim + hid_dim
        shape = (input_size, self._output_dim)
        self.weight = torch.nn.Parameter(torch.empty(*shape, device=self._device))
        self.biases = torch.nn.Parameter(torch.empty(self._output_dim, device=self._device))
        torch.nn.init.xavier_normal_(self.weight)
        torch.nn.init.constant_(self.biases, bias_start)

    def forward(self, inputs, state):
        batch_size = inputs.shape[0]
        # Reshape input and state to (batch_size * self._num_nodes, input_dim/state_dim)
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        # (batch_size * self._num_nodes, input_size(input_dim+state_dim))
        value = torch.sigmoid(torch.matmul(inputs_and_state, self.weight))
        # (batch_size * self._num_nodes, self._output_dim)
        value += self.biases
        # Reshape res back to 2D: (batch_size * num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(value, [batch_size, self._num_nodes * self._output_dim])


class DCGRUCell(nn.Module):
    def __init__(self, input_dim, num_units, max_diffusion_step, num_nodes, device, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True):
        """
        Args:
            input_dim:
            num_units:
            max_diffusion_step:
            num_nodes:
            device:
            nonlinearity:
            filter_type: "laplacian", "random_walk", "dual_random_walk"
            use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._device = device
        # self._adj_mx = self._calculate_random_walk_matrix(adj_mx).t()
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru
        self.device = device

        if self._use_gc_for_ru:
            self._fn = GCONV(self._num_nodes, self._max_diffusion_step, self._device,
                             input_dim=input_dim, hid_dim=self._num_units, output_dim=2*self._num_units,
                             bias_start=1.0)
        else:
            self._fn = FC(self._num_nodes, self._device, input_dim=input_dim,
                          hid_dim=self._num_units, output_dim=2*self._num_units, bias_start=1.0)
        self._gconv = GCONV(self._num_nodes, self._max_diffusion_step, self._device,
                            input_dim=input_dim, hid_dim=self._num_units, output_dim=self._num_units,
                            bias_start=0.0)

    @staticmethod
    def _build_sparse_matrix(lap, device):
        lap = lap.tocoo()
        indices = np.column_stack((lap.row, lap.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        lap = torch.sparse_coo_tensor(indices.T, lap.data, lap.shape, device=device)
        return lap

    def _calculate_random_walk_matrix(self, adj_mx):
        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[0])).to(self._device)
        d = torch.sum(adj_mx, 1)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(self._device), d_inv)
        d_mat_inv = torch.diag(d_inv)
        random_walk_mx = torch.mm(d_mat_inv, adj_mx)
        return random_walk_mx

    def forward(self, inputs, hx, adj):
        """
        Gated recurrent unit (GRU) with Graph Convolution.
        Args:
            inputs: (B, num_nodes * input_dim)
            hx: (B, num_nodes * rnn_units)
        Returns:
            torch.tensor: shape (B, num_nodes * rnn_units)
        """
        adj_mx = self._calculate_random_walk_matrix(adj).t()

        output_size = 2 * self._num_units
        value = torch.sigmoid(self._fn(inputs, hx, adj_mx))  # (batch_size, num_nodes * output_size)
        value = torch.reshape(value, (-1, self._num_nodes, output_size))    # (batch_size, num_nodes, output_size)

        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))  # (batch_size, num_nodes * _num_units)
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))  # (batch_size, num_nodes * _num_units)

        c = self._gconv(inputs, r * hx, adj_mx)  # (batch_size, num_nodes * _num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state  # (batch_size, num_nodes * _num_units)


class Seq2SeqAttrs:
    def __init__(self, config, data_feature):
        self.max_diffusion_step = int(config.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(config.get('cl_decay_steps', 1000))
        self.filter_type = config.get('filter_type', 'laplacian')
        self.num_nodes = int(data_feature.get('num_nodes', 1))
        # print(f"num nodes is {self.num_nodes}")
        self.num_rnn_layers = int(config.get('num_rnn_layers', 1))
        self.rnn_units = int(config.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.input_dim = int(data_feature.get('feature_dim'))
        self.device = config.get('device', torch.device('cpu'))


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, config, data_feature, device):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config, data_feature)
        self.device = device
        self.seq_len = int(config.get('input_window', 1))  # for the encoder
        self.dcgru_layers = nn.ModuleList()
        self.dcgru_layers.append(DCGRUCell(self.input_dim, self.rnn_units, self.max_diffusion_step,
                                           self.num_nodes, self.device, filter_type=self.filter_type))
        for i in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(DCGRUCell(self.rnn_units, self.rnn_units, self.max_diffusion_step,
                                               self.num_nodes, self.device, filter_type=self.filter_type))

    def forward(self, inputs, adj, hidden_state=None):
        """
        Encoder forward pass.
        Args:
            inputs: shape (batch_size, self.num_nodes * self.input_dim)
            hidden_state: (num_layers, batch_size, self.hidden_state_size),
                optional, zeros if not provided, hidden_state_size = num_nodes * rnn_units
        Returns:
            tuple: tuple contains:
                output: shape (batch_size, self.hidden_state_size) \n
                hidden_state: shape (num_layers, batch_size, self.hidden_state_size) \n
                (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size), device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            # next_hidden_state: (batch_size, self.num_nodes * self.rnn_units)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state  # 循环
        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, config, data_feature, device):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config, data_feature)
        self.device = device
        self.output_dim = config.get('output_dim', 1)
        self.horizon = int(config.get('output_window', 1))
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList()
        self.dcgru_layers.append(DCGRUCell(self.output_dim, self.rnn_units, self.max_diffusion_step,
                                           self.num_nodes, self.device, filter_type=self.filter_type))
        for i in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(DCGRUCell(self.rnn_units, self.rnn_units, self.max_diffusion_step,
                                               self.num_nodes, self.device, filter_type=self.filter_type))

    def forward(self, inputs, adj, hidden_state=None):
        """
        Decoder forward pass.
        Args:
            inputs:  shape (batch_size, self.num_nodes * self.output_dim)
            hidden_state: (num_layers, batch_size, self.hidden_state_size),
                optional, zeros if not provided, hidden_state_size = num_nodes * rnn_units
        Returns:
            tuple: tuple contains:
                output: shape (batch_size, self.num_nodes * self.output_dim) \n
                hidden_state: shape (num_layers, batch_size, self.hidden_state_size) \n
                (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            # next_hidden_state: (batch_size, self.num_nodes * self.rnn_units)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)
        return output, torch.stack(hidden_states)


class GTS(AbstractTrafficStateModel, Seq2SeqAttrs):
    def __init__(self, config, data_feature):
        """
        构造模型
        :param config: 源于各种配置的配置字典
        :param data_feature: 从数据集Dataset类的`get_data_feature()`接口返回的必要的数据相关的特征
        """

        super().__init__(config, data_feature)
        self.config = config
        self.device = config.get('device', torch.device('cpu'))
        self.adj_mx = torch.Tensor(data_feature.get('adj_mx')).to(self.device)
        Seq2SeqAttrs.__init__(self, self.config, data_feature)

        self.seq_len = int(config.get('input_window', 1))  # for the encoder
        self.horizon = int(config.get('output_window', 1))  # for the decoder

        self.encoder_model = EncoderModel(self.config, data_feature, self.device)
        self.decoder_model = DecoderModel(self.config, data_feature, self.device)
        self._logger = getLogger()

        # 此处 adj_mx 作用是在训练自动图结构推断时起到参考作用
        self.adj_mx = torch.Tensor(data_feature.get('adj_mx')).to(self.device)
        # print(f"ADJMX={self.adj_mx}")
        self.cl_decay_steps = config.get('cl_decay_steps', 1000)
        self.use_curriculum_learning = config.get('use_curriculum_learning', True)
        self.temperature = config.get('temperature', 0.5)
        self.epoch_use_regularization = config.get('epoch_use_regularization', 50)

        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.num_batches = self.data_feature.get('num_batches', 1)
        self._scaler = self.data_feature.get('scaler')
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.ext_dim = self.data_feature.get('ext_dim', 1)
        train_feas = self.data_feature.get('train_data')  # (num_samples, num_nodes)

        self.node_feas = torch.Tensor(train_feas).to(self.device)

        self.kernal_size = config.get('kernal_size', 10)
        self.dim_fc = (self.node_feas.shape[0] - 2 * self.kernal_size + 2) * 16
        self.embedding_dim = config.get('embedding_dim', 100)
        self.conv1 = torch.nn.Conv1d(1, 8, self.kernal_size, stride=1)
        self.conv2 = torch.nn.Conv1d(8, 16, self.kernal_size, stride=1)
        self.hidden_drop = torch.nn.Dropout(0.2)
        # print(f"FC shape={self.dim_fc}, {self.embedding_dim}")
        self.fc = torch.nn.Linear(self.dim_fc, self.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.fc_out = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.fc_cat = nn.Linear(self.embedding_dim, 2)

        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
            return labels_onehot

        # Generate off-diagonal interaction graph
        off_diag = np.ones([self.num_nodes, self.num_nodes])
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(self.device)
        self.rel_send = torch.FloatTensor(rel_send).to(self.device)

        self.input_dim = self.feature_dim
        # print(f"feature_dim = {self.input_dim}")

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, adj):
        """
        Encoder forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], adj, encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, adj, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim), device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = \
                self.decoder_model(decoder_input, adj, decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def _prepare_data_x(self, x):
        x = x.float()
        x = x.permute(1, 0, 2, 3)
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        return x

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(self.device), y.to(self.device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = x.float()
        y = y.float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].contiguous().view(
            self.horizon, batch_size, self.num_nodes * self.output_dim)
        return x, y

    def forward(self, batch, batches_seen=None):
        batch_size = batch['X'].size(0)
        if batch['y'] is not None:
            inputs, labels = self._prepare_data(batch['X'], batch['y'])
            # print(f"y = {batch['y'].shape}")
            # print(f"labels = {labels.shape}")
        else:
            inputs = self._prepare_data_x(batch['X'])
            labels = None

        # 图结构的推断过程
        x = self.node_feas.transpose(1, 0).view(self.num_nodes, 1, -1)  # [207, 1, 24000]
        x = self.conv1(x)  # [207, 8, 23991]
        x = F.relu(x)
        x = self.bn1(x)
        # x = self.hidden_drop(x)
        x = self.conv2(x)  # [207, 16, 23982]
        x = F.relu(x)
        x = self.bn2(x)
        x = x.view(self.num_nodes, -1)  # [207, 383712]
        x = self.fc(x)
        x = F.relu(x)
        x = self.bn3(x)

        receivers = torch.matmul(self.rel_rec, x)
        senders = torch.matmul(self.rel_send, x)
        x = torch.cat([senders, receivers], dim=1)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)

        adj = gumbel_softmax(self.device, x, temperature=self.temperature, hard=True)
        adj = adj[:, 0].clone().reshape(self.num_nodes, -1)
        mask = torch.eye(self.num_nodes, self.num_nodes).bool().to(self.device)
        adj.masked_fill_(mask, 0)

        encoder_hidden_state = self.encoder(inputs, adj)
        self._logger.debug("Encoder complete, starting decoder")
        outputs = self.decoder(encoder_hidden_state, adj, labels, batches_seen=batches_seen)
        self._logger.debug("Decoder complete")

        # print(f"shape of output = {outputs.shape}")
        orig_out = outputs.view(self.horizon, batch_size, self.num_nodes, self.output_dim).permute(1, 0, 2, 3)
        return orig_out, x[:, 0].clone().reshape(self.num_nodes, -1)

    def calculate_loss(self, batch, batches_seen=None):
        y_true = batch['y']
        epoch = batches_seen // self.num_batches
        self._logger.debug(f"EPOCH = {epoch}, bep={batches_seen}")
        y_predicted, mid_output = self.forward(batch, batches_seen)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        # 根据训练轮数，选择性地加入正则项
        loss_1 = loss.masked_mae_torch(y_predicted, y_true)
        if epoch < self.epoch_use_regularization:
            pred = torch.sigmoid(mid_output.view(mid_output.shape[0] * mid_output.shape[1]))
            # print(f"shape = {mid_output.shape}")
            # print(f"aview = {self.adj_mx.view(mid_output.shape[0] * mid_output.shape[1])}")
            true_label = self.adj_mx.view(mid_output.shape[0] * mid_output.shape[1]).to(self.device)
            compute_loss = torch.nn.BCELoss()
            loss_g = compute_loss(pred, true_label)
            self._logger.debug(f"loss_g = {loss_g}, loss_1 = {loss_1}")
            loss_t = loss_1 + loss_g
            return loss_t
        else:
            self._logger.debug(f"loss_1 = {loss_1}")
            return loss_1

    def predict(self, batch, batches_seen=None):
        return self.forward(batch, batches_seen)[0]
