import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math


# fixme cell应该这样添加么，要不要学习FOST写一个 base cell

class TGCNCell(nn.Module):
    def __init__(self, config, gate, update):
        super(TGCNCell, self).__init__()
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


class TGCLSTMCell(nn.Module):
    def __init__(self,config, gcn):
        super(TGCLSTMCell, self).__init__()
        self.input_size = config.get('input_size')
        self.hidden_size = config.get('hidden_size')
        self.bias = config.get('bias', True)
        self.gcn=gcn(config['gcn'])
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



