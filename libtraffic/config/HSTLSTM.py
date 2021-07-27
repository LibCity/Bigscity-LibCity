import math

import torch
from torch.nn import init
from torch.nn.parameter import Parameter
import math
from bisect import bisect

import nni
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm, trange

def st_lstm_cell_cal(input_l, input_s, input_q, hidden, cell, w_ih, w_hh, w_s, w_q, b_ih, b_hh):
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
    gates = torch.mm(input_l, w_ih.t()) + torch.mm(hidden, w_hh.t()) + b_ih + b_hh  # Shape (batch_size, 4 * hidden_size)
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
        return st_lstm_cell_cal(input_l=input_l, input_s=input_s, input_q=input_q,
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
        return (value - lower_value) / total_distance, \
               (higher_value - value) / total_distance, \
               lower_bound, higher_bound


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
    return np.array(ld), np.array(hd), np.array(l), np.array(h)


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
            hc = self.cell(input_l[:,step,:], input_s[:,step,:], input_q[:,step,:], hc)
            output_hidden.append(hc[0])
            output_cell.append(hc[1])
        return torch.stack(output_hidden, 1), torch.stack(output_cell, 1)

class STLSTMClassifier(nn.Module):
    """
    RNN classifier using ST-LSTM as its core.
    """
    def __init__(self, input_size, output_size, hidden_size,
                 temporal_slots, spatial_slots,
                 device, learning_rate):
        """
        :param input_size: The number of expected features in the input vectors.
        :param output_size: The number of classes in the classifier outputs.
        :param hidden_size: The number of features in the hidden state.
        :param temporal_slots: values of temporal slots.
        :param spatial_slots: values of spatial slots.
        :param device: The name of the device used for training.
        :param learning_rate: Learning rate of training.
        """
        super(STLSTMClassifier, self).__init__()
        self.temporal_slots = sorted(temporal_slots)
        self.spatial_slots = sorted(spatial_slots)

        # Initialization of network parameters.
        self.st_lstm = STLSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

        # Embedding matrix for every temporal and spatial slots.
        self.embed_s = nn.Embedding(len(temporal_slots), input_size)
        self.embed_s.weight.data.normal_(0, 0.1)
        self.embed_q = nn.Embedding(len(spatial_slots), input_size)
        self.embed_q.weight.data.normal_(0, 0.1)

        # Initialization of network components.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_func = nn.CrossEntropyLoss()

        self.device = torch.device(device)
        self.to(self.device)

    def place_parameters(self, ld, hd, l, h):
        ld = torch.from_numpy(np.array(ld)).type(torch.FloatTensor).to(self.device)
        hd = torch.from_numpy(np.array(hd)).type(torch.FloatTensor).to(self.device)
        l = torch.from_numpy(np.array(l)).type(torch.LongTensor).to(self.device)
        h = torch.from_numpy(np.array(h)).type(torch.LongTensor).to(self.device)

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

    def forward(self, batch_l, batch_t, batch_d):
        """
        Process forward propagation of ST-LSTM classifier.
        :param batch_l: batch of input location sequences,
            size (batch_size, time_step, input_size)
        :param batch_t: batch of temporal interval value, size (batch_size, step)
        :param batch_d: batch of spatial distance value, size (batch_size, step)
        :return: prediction result of this batch, size (batch_size, output_size, step).
        """
        batch_l = torch.from_numpy(np.array(batch_l)).type(torch.FloatTensor).to(self.device)

        t_ld, t_hd, t_l, t_h = self.place_parameters(*cal_slot_distance_batch(batch_t, self.temporal_slots))
        d_ld, d_hd, d_l, d_h = self.place_parameters(*cal_slot_distance_batch(batch_d, self.spatial_slots))

        batch_s = self.cal_inter(t_ld, t_hd, t_l, t_h, self.embed_s)
        batch_q = self.cal_inter(d_ld, d_hd, d_l, d_h, self.embed_q)

        hidden_out, cell_out = self.st_lstm(batch_l, batch_s, batch_q)
        linear_out = self.linear(hidden_out[:,-1,:])
        return linear_out

    def predict(self, batch_l, batch_t, batch_d):
        """
        Predict a batch of data.
        :param batch_l: batch of input location sequences,
            size (batch_size, time_step, input_size)
        :param batch_t: batch of temporal interval value, size (batch_size, step)
        :param batch_d: batch of spatial distance value, size (batch_size, step)
        :return: batch of predicted class indices, size (batch_size).
        """
        return torch.max(self.forward(batch_l, batch_t, batch_d), 1)[1].detach().cpu().numpy().squeeze()

    def calculate_loss(self, batch_l, batch_t, batch_d, batch_label):
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
        prediction = self(batch_l, batch_t, batch_d)
        batch_label = torch.from_numpy(np.array(batch_label)).type(torch.LongTensor).to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss_func(prediction, batch_label)
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()