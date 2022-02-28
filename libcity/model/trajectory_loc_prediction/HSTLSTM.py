import math
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from bisect import bisect
import numpy as np
from torch import nn

from libcity.model.abstract_model import AbstractModel


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

        self.w_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.w_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.w_s = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.w_q = Parameter(torch.Tensor(3 * hidden_size, input_size))
        if bias:
            self.b_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.b_hh = Parameter(torch.Tensor(4 * hidden_size))
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
            init.uniform_(weight, -stdv, stdv)

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


def cal_slot_distance(value, slots):
    """
    Calculate a value's distance with nearest lower bound and higher bound in slots.
    :param value: The value to be calculated.
    :param slots: values of slots, needed to be sorted.
    :return: normalized distance with lower bound and higher bound,
        and index of lower bound and higher bound.
    """
    higher_bound = bisect(slots, value)
    lower_bound = higher_bound - 1
    if higher_bound == len(slots):
        return 1., 0., lower_bound, lower_bound
    else:
        lower_value = slots[lower_bound]
        higher_value = slots[higher_bound]
        total_distance = higher_value - lower_value
        return (value - lower_value) / total_distance, (higher_value -
                                                        value) / total_distance, lower_bound, higher_bound


def cal_slot_distance_batch(batch_value, slots):
    """
    Proceed `cal_slot_distance` on a batch of data.
    :param batch_value: a batch of value, size (batch_size, step)
    :param slots: values of slots, needed to be sorted.
    :return: batch of distances and indexes. All with shape (batch_size, step).
    """
    # Lower bound distance, higher bound distance, lower bound, higher bound.
    ld, hd, l, h = [], [], [], []
    for batch in batch_value:
        ld_row, hd_row, l_row, h_row = [], [], [], []
        for step in batch:
            ld_one, hd_one, l_one, h_one = cal_slot_distance(step, slots)
            ld_row.append(ld_one)
            hd_row.append(hd_one)
            l_row.append(l_one)
            h_row.append(h_one)
        ld.append(ld_row)
        hd.append(hd_row)
        l.append(l_row)
        h.append(h_row)
    return ld, hd, l, h


def construct_slots(min_value, max_value, num_slots, type):
    """
    Construct values of slots given min value and max value.
    :param min_value: minimum value.
    :param max_value: maximum value.
    :param num_slots: number of slots to construct.
    :param type: type of slots to construct, 'linear' or 'exp'.
    :return: values of slots.
    """
    if type == 'exp':
        n = (max_value - min_value) / (math.exp(num_slots - 1) - 1)
        return [n * (math.exp(x) - 1) + min_value for x in range(num_slots)]
    elif type == 'linear':
        n = (max_value - min_value) / (num_slots - 1)
        return [n * x + min_value for x in range(num_slots)]


class STLSTM(nn.Module):
    """
    One layer, batch-first Spatial-Temporal LSTM network.
    Kong D, Wu F. HST-LSTM: A Hierarchical Spatial-Temporal Long-Short Term Memory Network
    for Location Prediction[C]//IJCAI. 2018: 2341-2347.
    Examples:
        >>> st_lstm = STLSTM(10, 20)
        >>> input_l = torch.randn(6, 3, 10)
        >>> input_s = torch.randn(6, 3, 10)
        >>> input_q = torch.randn(6, 3, 10)
        >>> hidden_out, cell_out = st_lstm(input_l, input_s, input_q)
    """
    def __init__(self, input_size, hidden_size, bias=True):
        """
        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`. Default: ``True``
        """
        super(STLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.cell = STLSTMCell(input_size, hidden_size, bias)

    def check_forward_input(self, input_l, input_s, input_q):
        if not (input_l.size(1) == input_s.size(1) == input_q.size(1)):
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def forward(self, input_l, input_s, input_q, hc=None):
        """
        Proceed forward propagation of ST-LSTM network.
        :param input_l: input of location embedding vector, shape (batch_size, step, input_size)
        :param input_s: input of spatial embedding vector, shape (batch_size, step, input_size)
        :param input_q: input of temporal embedding vector, shape (batch_size, step, input_size)
        :param hc: tuple containing initial hidden state and cell state, optional.
        :return: hidden states and cell states produced by iterate through the steps.
        """
        output_hidden, output_cell = [], []
        self.check_forward_input(input_l, input_s, input_q)
        for step in range(input_l.size(1)):
            hc = self.cell(input_l[:, step, :], input_s[:, step, :], input_q[:, step, :], hc)
            output_hidden.append(hc[0])
            output_cell.append(hc[1])
        return torch.stack(output_hidden, 1), torch.stack(output_cell, 1)


class HSTLSTM(AbstractModel):
    """
    RNN classifier using ST-LSTM as its core.
    """
    def __init__(self, config, data_feature):
        """
        """
        super(HSTLSTM, self).__init__(config, data_feature)
        self.tim_slot_max = data_feature['tim_slot_max']
        self.dis_slot_max = data_feature['dis_slot_max']
        self.tim_slot_len = min(config['tim_slot_len'], self.tim_slot_max + 1)
        self.dis_slot_len = min(config['dis_slot_len'], self.dis_slot_max + 1)
        self.tim_slots = np.linspace(0, self.tim_slot_max, self.tim_slot_len).astype(int)
        self.dis_slots = np.linspace(0, self.dis_slot_max, self.dis_slot_len).astype(int)
        self.embed_size = config["embed_size"]
        self.hidden_size = config['hidden_size']
        self.loc_size = data_feature['loc_size']
        self.device = config['device']
        # Initialization of network parameters.
        self.st_lstm = STLSTM(self.embed_size, self.hidden_size)
        # output layer
        self.linear = nn.Linear(self.hidden_size, self.loc_size)

        # Embedding matrix for every temporal and spatial slots.
        self.embed_s = nn.Embedding(self.tim_slot_len, self.embed_size)
        self.embed_s.weight.data.normal_(0, 0.1)
        self.embed_q = nn.Embedding(self.dis_slot_len, self.embed_size)
        self.embed_q.weight.data.normal_(0, 0.1)
        self.embed_l = nn.Embedding(self.loc_size, self.embed_size)

        # Initialization of network components.
        self.loss_func = nn.NLLLoss()

    def place_parameters(self, ld, hd, l, h):
        ld = torch.FloatTensor(ld).to(self.device)
        hd = torch.FloatTensor(hd).to(self.device)
        l = torch.LongTensor(l).to(self.device)
        h = torch.LongTensor(h).to(self.device)

        return ld, hd, l, h

    def cal_inter(self, ld, hd, l, h, embed):
        """
        Calculate a linear interpolation.
        :param ld: Distances to lower bound, shape (batch_size, step)
        :param hd: Distances to higher bound, shape (batch_size, step)
        :param l: Lower bound indexes, shape (batch_size, step)
        :param h: Higher bound indexes, shape (batch_size, step)
        """
        # Fetch the embed of higher and lower bound.
        # Each result shape (batch_size, step, input_size)
        l_embed = embed(l)
        h_embed = embed(h)
        return torch.stack([hd], -1) * l_embed + torch.stack([ld], -1) * h_embed

    def forward(self, batch_l, batch_t, batch_d, origin_len):
        """
        Process forward propagation of ST-LSTM classifier.
        :param batch_l: batch of input location sequences,
            size (batch_size, time_step, input_size)
        :param batch_t: batch of temporal interval value, size (batch_size, step)
        :param batch_d: batch of spatial distance value, size (batch_size, step)
        :return: prediction result of this batch, size (batch_size, output_size, step).
        """

        t_ld, t_hd, t_l, t_h = self.place_parameters(*cal_slot_distance_batch(batch_t.tolist(), self.tim_slots))
        d_ld, d_hd, d_l, d_h = self.place_parameters(*cal_slot_distance_batch(batch_d.tolist(), self.dis_slots))

        batch_s = self.cal_inter(t_ld, t_hd, t_l, t_h, self.embed_s)
        batch_q = self.cal_inter(d_ld, d_hd, d_l, d_h, self.embed_q)
        batch_l = self.embed_l(batch_l)

        hidden_out, cell_out = self.st_lstm(batch_l, batch_s, batch_q)
        # we do padding
        # 因为是补齐了的，所以需要找到真正的 out
        final_out_index = torch.tensor(origin_len) - 1
        final_out_index = final_out_index.reshape(final_out_index.shape[0], 1, -1)
        final_out_index = final_out_index.repeat(1, 1, self.hidden_size).to(self.device)
        out = torch.gather(hidden_out, 1, final_out_index).squeeze(1)  # batch_size * hidden_size
        linear_out = self.linear(out)
        return F.log_softmax(linear_out, dim=1)

    def predict(self, batch):
        """
        Predict a batch of data.
        :param batch_l: batch of input location sequences,
            size (batch_size, time_step, input_size)
        :param batch_t: batch of temporal interval value, size (batch_size, step)
        :param batch_d: batch of spatial distance value, size (batch_size, step)
        :return: batch of predicted class indices, size (batch_size).
        """
        batch_l = batch['current_loc']
        batch_t = batch['tim_interval']
        batch_d = batch['dis']
        origin_len = batch.get_origin_len('current_loc')
        return self.forward(batch_l, batch_t, batch_d, origin_len)

    def calculate_loss(self, batch):
        """
        Train model using one batch of data and return loss value.
        :param model: One instance of STLSTMClassifier.
        :param batch_l: batch of input location sequences,
            size (batch_size, time_step, input_size)
        :param batch_t: batch of temporal interval value, size (batch_size, step)
        :param batch_d: batch of spatial distance value, size (batch_size, step)
        :param batch_label: batch of label, size (batch_size)
        :return: loss value.
        """
        batch_l = batch['current_loc']
        batch_t = batch['tim_interval']
        batch_d = batch['dis']
        origin_len = batch.get_origin_len('current_loc')
        prediction = self.forward(batch_l, batch_t, batch_d, origin_len)
        batch_label = batch['target']

        return self.loss_func(prediction, batch_label)
