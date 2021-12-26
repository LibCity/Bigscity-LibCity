import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math


# fixme cell应该这样添加么，要不要学习FOST写一个 base cell

class GCGRUCell(nn.Module):
    def __init__(self, config, gate, update):
        super(GCGRUCell, self).__init__()
        """
            config['gate'] and config['update'] = { adj, num_nodes, dim_in, dim_out, ... }
            gate output_dim = 2 * update output_dim
        """
        self.gate = gate(config['gate'])
        self.update = update(config['update'])

    def forward(self, x, state):
        """

        u/r = sigmoid(W_u/W_r(f(A，[X,state])) + b)
        c = tanh( W_c(f(A,[X,r * state])) +b)
        out = u*state + (1-u) * c

        Args:
            x: B, num_nodes, input_dim
            state: B, num_nodes, hidden_dim
        Returns:
            tensor: B, num_nodes, hidden_dim
        """
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        u_r = torch.sigmoid(self.gate(input_and_state))
        u, r = torch.split(u_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, u * state), dim=-1)
        hc = torch.tanh(self.update(candidate))
        h = r * state + (1 - r) * hc
        return h


class GCLSTMCell(nn.Module):
    def __init__(self, config, gcn):
        super(GCLSTMCell, self).__init__()
        self.input_size = config.get('input_size')
        self.hidden_size = config.get('hidden_size')
        self.bias = config.get('bias', True)
        self.gcn = gcn(config['gcn'])
        self.x2h = nn.Linear(self.input_size, 4 * self.hidden_size, bias=self.bias)
        self.h2h = nn.Linear(self.hidden_size, 4 * self.hidden_size, bias=self.bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        """
        f/i/o = sigmoid( W(f(x)) + U(hidden) +bias)
        c = tanh( W(f(x)) + U(hidden) +bias)
        C_hot = f * context + i * c
        h_hot = o * tanh(c_hot)

        Args:
            x: shape (batch,num_nodes,input_dim)
            hidden: shape (2,batch,num_nodes,input_dim)

        Returns:
            tuple:(hy tensor,
                    cy tensor)
        """
        hx, cx = hidden

        x = self.gcn(x)

        x = x.view(-1, x.size(-1))  # (batch*num_nodes,input_size)

        gates = self.x2h(x) + self.h2h(hx)  # (batch*num_nodes,4*hidden)

        gates = gates.squeeze()

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)  # (batch*num_nodes,hidden)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        outgate = F.sigmoid(outgate)
        cellgate = F.tanh(cellgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)

        hy = torch.mul(outgate, F.tanh(cy))
        return (hy, cy)


class CARACell(nn.Module):

    def hard_sigmoid(self, x):
        x = torch.tensor(x / 6 + 0.5)
        x = F.threshold(-x, -1, -1)
        x = F.threshold(-x, 0, 0)
        return x

    def __init__(self, output_dim, input_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 **kwargs):
        super(CARACell, self).__init__()
        self.output_dim = output_dim
        self.init = init
        self.inner_init = inner_init
        self.activation = self.hard_sigmoid
        self.inner_activation = nn.Tanh()
        self.build(input_dim)

    def add_weight(self, shape, initializer):
        ts = torch.zeros(shape)
        if initializer == 'glorot_uniform':
            ts = nn.init.xavier_normal_(ts)
        elif initializer == 'orthogonal':
            ts = nn.init.orthogonal_(ts)

        return nn.Parameter(ts)

    def build(self, input_shape):
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape

        self.W_z = self.add_weight((self.input_dim, self.output_dim),
                                   initializer=self.init)
        self.U_z = self.add_weight((self.output_dim, self.output_dim),
                                   initializer=self.init)
        self.b_z = self.add_weight((self.output_dim,),
                                   initializer='zero')
        self.W_r = self.add_weight((self.input_dim, self.output_dim),
                                   initializer=self.init)
        self.U_r = self.add_weight((self.output_dim, self.output_dim),
                                   initializer=self.init)
        self.b_r = self.add_weight((self.output_dim,),
                                   initializer='zero')
        self.W_h = self.add_weight((self.input_dim, self.output_dim),
                                   initializer=self.init)
        self.U_h = self.add_weight((self.output_dim, self.output_dim),
                                   initializer=self.init)
        self.b_h = self.add_weight((self.output_dim,),
                                   initializer='zero')

        self.A_h = self.add_weight((self.output_dim, self.output_dim),
                                   initializer=self.init)
        self.A_u = self.add_weight((self.output_dim, self.output_dim),
                                   initializer=self.init)

        self.b_a_h = self.add_weight((self.output_dim,),
                                     initializer='zero')
        self.b_a_u = self.add_weight((self.output_dim,),
                                     initializer='zero')

        self.W_t = self.add_weight((self.input_dim, self.output_dim),
                                   initializer=self.init)
        self.U_t = self.add_weight((1, self.output_dim),
                                   initializer=self.init)
        self.b_t = self.add_weight((self.output_dim,),
                                   initializer='zero')

        self.W_g = self.add_weight((self.input_dim, self.output_dim),
                                   initializer=self.init)
        self.U_g = self.add_weight((1, self.output_dim),
                                   initializer=self.init)
        self.b_g = self.add_weight((self.output_dim,),
                                   initializer='zero')

    def preprocess_input(self, x):
        return x

    def forward(self, x, u,t,g,states):
        """
        用于多批次同一时间
        states为上一次多批次统一时间数据
        Args:
            # phi_t
            u = [:, self.output_dim]
            # delta_t
            t = [:, 1]
            # delta_g
            g = [:, self.output_dim]
            # phi_v
            x = [:, self.output_dim]
            states = [:,self.output_dim]
        Returns:

        """

        h_tm1 = states


        t = self.inner_activation(torch.matmul(t, self.U_t))
        g = self.inner_activation(torch.matmul(g, self.U_g))
        #       Time-based gate
        t1 = self.inner_activation(torch.matmul(x, self.W_t) + t + self.b_t)
        #       Geo-based gate
        g1 = self.inner_activation(torch.matmul(x, self.W_g) + g + self.b_g)

        #       Contextual Attention Gate
        a = self.inner_activation(
            torch.matmul(h_tm1, self.A_h) + torch.matmul(u, self.A_u) + self.b_a_h + self.b_a_u)

        x_z = torch.matmul(x, self.W_z) + self.b_z
        x_r = torch.matmul(x, self.W_r) + self.b_r
        x_h = torch.matmul(x, self.W_h) + self.b_h

        u_z_ = torch.matmul((1 - a) * u, self.W_z) + self.b_z
        u_r_ = torch.matmul((1 - a) * u, self.W_r) + self.b_r
        u_h_ = torch.matmul((1 - a) * u, self.W_h) + self.b_h

        u_z = torch.matmul(a * u, self.W_z) + self.b_z
        u_r = torch.matmul(a * u, self.W_r) + self.b_r
        u_h = torch.matmul(a * u, self.W_h) + self.b_h

        #       update gate
        z = self.inner_activation(x_z + torch.matmul(h_tm1, self.U_z) + u_z)
        #       reset gate
        r = self.inner_activation(x_r + torch.matmul(h_tm1, self.U_r) + u_r)
        #       hidden state
        hh = self.activation(x_h + torch.matmul(r * t1 * g1 * h_tm1, self.U_h) + u_h)

        h = z * h_tm1 + (1 - z) * hh
        h = (1 + u_z_ + u_r_ + u_h_) * h
        return h
        # return h


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class STLSTMCell(nn.Module):
    """
    A Spatial-Temporal Long Short Term Memory (ST-LSTM) cell.
    Kong D, Wu F. HST-LSTM: A Hierarchical Spatial-Temporal Long-Short Term Memory Network
    for Location Prediction[C]//IJCAI. 2018: 2341-2347.
    Examples:
        >>> st_lstm = STLSTMCell(10, 20)
        >>> input_l = torch.randn(6, 3, 10)
        >>> input_s = torch.randn(6, 3, 10)
        >>> input_q = torch.randn(6, 3, 10)
        >>> hc = (torch.randn(3, 20), torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        >>>     hc = st_lstm(input_l[i], input_s[i], input_q[i], hc)
        >>>     output.append(hc[0])
    """

    def __init__(self, input_size, hidden_size, bias=True):
        """
        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`. Default: ``True``
        """
        super(STLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.w_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.w_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.w_s = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.w_q = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        if bias:
            self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('b_ih', None)
            self.register_parameter('b_hh', None)

        self.reset_parameters()

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        # type: (Tensor, Tensor, str) -> None
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def st_lstm_cell_cal(self, input_l, input_s, input_q, hidden, cell, w_ih, w_hh, w_s, w_q, b_ih, b_hh):
        """
        Proceed calculation of one step of STLSTM.
        :param input_l: input of location embedding, shape (batch_size, input_size)
        :param input_s: input of spatial embedding, shape (batch_size, input_size)
        :param input_q: input of temporal embedding, shape (batch_size, input_size)
        :param hidden: hidden state from previous step, shape (batch_size, hidden_size)
        :param cell: cell state from previous step, shape (batch_size, hidden_size)
        :param w_ih: chunk of weights for process input tensor, shape (4 * hidden_size, input_size)
        :param w_hh: chunk of weights for process hidden state tensor, shape (4 * hidden_size, hidden_size)
        :param w_s: chunk of weights for process input of spatial embedding, shape (3 * hidden_size, input_size)
        :param w_q: chunk of weights for process input of temporal embedding, shape (3 * hidden_size, input_size)
        :param b_ih: chunk of biases for process input tensor, shape (4 * hidden_size)
        :param b_hh: chunk of biases for process hidden state tensor, shape (4 * hidden_size)
        :return: hidden state and cell state of this step.
        """
        # Shape (batch_size, 4 * hidden_size)
        gates = torch.mm(input_l, w_ih.t()) + torch.mm(hidden, w_hh.t()) + b_ih + b_hh
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

        ifo_gates = torch.cat((in_gate, forget_gate, out_gate), 1)  # shape (batch_size, 3 * hidden_size)
        ifo_gates += torch.mm(input_s, w_s.t()) + torch.mm(input_q, w_q.t())
        in_gate, forget_gate, out_gate = ifo_gates.chunk(3, 1)

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        next_cell = (forget_gate * cell) + (in_gate * cell_gate)
        next_hidden = out_gate * torch.tanh(cell_gate)

        return next_hidden, next_cell

    def forward(self, input_l, input_s, input_q, hc=None):
        """
        Proceed one step forward propagation of ST-LSTM.
        :param input_l: input of location embedding vector, shape (batch_size, input_size)
        :param input_s: input of spatial embedding vector, shape (batch_size, input_size)
        :param input_q: input of temporal embedding vector, shape (batch_size, input_size)
        :param hc: tuple containing hidden state and cell state of previous step.
        :return: hidden state and cell state of this step.
        """
        self.check_forward_input(input_l)
        self.check_forward_input(input_s)
        self.check_forward_input(input_q)
        if hc is None:
            zeros = torch.zeros(input_l.size(0), self.hidden_size, dtype=input_l.dtype, device=input_l.device)
            hc = (zeros, zeros)
        self.check_forward_hidden(input_l, hc[0], '[0]')
        self.check_forward_hidden(input_l, hc[1], '[0]')
        self.check_forward_hidden(input_s, hc[0], '[0]')
        self.check_forward_hidden(input_s, hc[1], '[0]')
        self.check_forward_hidden(input_q, hc[0], '[0]')
        self.check_forward_hidden(input_q, hc[1], '[0]')
        return self.st_lstm_cell_cal(input_l=input_l, input_s=input_s, input_q=input_q,
                                     hidden=hc[0], cell=hc[1],
                                     w_ih=self.w_ih, w_hh=self.w_hh, w_s=self.w_s, w_q=self.w_q,
                                     b_ih=self.b_ih, b_hh=self.b_hh)
