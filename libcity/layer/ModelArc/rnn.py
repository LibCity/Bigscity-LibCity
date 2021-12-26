import torch
import torch.nn as nn
from torch.autograd import Variable
from Temporal.rnn import TGCNCell
from Temporal.rnn import TGCLSTMCell


class TemporalGRU(nn.Module):
    def __init__(self, config, update, gate):
        super(TemporalGRU, self).__init__()
        # Hidden dimensions
        self.num_nodes = config['num_nodes']
        #         print("num_nodes:"+str(self.num_nodes))
        self.feature_dim = config['feature_dim']
        self.device = config.get('device')
        self.gru_cell = TGCNCell(config,gate, update)

    def forward(self, x):
        """
        Args:
            x: shape (batch,num_timesteps,num_nodes,input_dim)

        Returns:
            tensor shape=(btach,num_nodes,hidden_dim)
        """
        if self.device:
            h0 = Variable(torch.zeros(x.size(0), x.size(2), self.hidden_dim).to(self.device))
        else:
            h0 = Variable(torch.zeros(x.size(0), x.size(2), self.hidden_dim))

        outs = []
        hn = h0
        x = x.permute(1, 0, 2, 3)  # (num_timesteps, batch,num_nodes,input_dim)
        for seq in range(x.size(0)):
            hn = self.gru_cell(x[seq], hn)
            outs.append(hn)
        outs = torch.stack(outs)
        # (batch, num_timesteps, num_nodes, input_dim)
        outs = outs.permute(1, 0, 2, 3)
        return outs


class TemporalLSTM(nn.Module):
    def __init__(self, config,gcn):
        super(TemporalLSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = config.get("hidden_dim")
        self.output_dim= config.get("output_dim")
        self.lstm = TGCLSTMCell(config,gcn)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.device = config.get("device")

    def forward(self, x):
        """
        Args:
            x: shape=(batch,num_timesteps,num_nodes,input_dim)

        Returns:

        """
        if torch.cuda.is_available():
            hn = Variable(torch.zeros(x.size(0), x.size(2), self.hidden_dim).to(self.device))
        else:
            hn = Variable(torch.zeros(x.size(0), x.size(2), self.hidden_dim))

        # Initialize cell state
        if torch.cuda.is_available():
            cn = Variable(torch.zeros(x.size(0), x.size(2), self.hidden_dim).to(self.device))
        else:
            cn = Variable(torch.zeros(x.size(0), x.size(2), self.hidden_dim))
        outs = []
        x = x.permute(1, 0, 2, 3)
        for seq in range(x.size(1)):
            hn, cn = self.lstm(x[seq], (hn, cn))
            outs.append(hn)
        outs = torch.vstack(outs)
        # out.shape=(batch,num_timesteps,num_nodes,output_dim)
        outs = outs.permute(1, 0, 2, 3)
        return outs