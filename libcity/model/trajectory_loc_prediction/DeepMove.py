import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from libcity.model.abstract_model import AbstractModel


class Attn(nn.Module):
    """
    Attention Module. Heavily borrowed from Practical Pytorch
    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation
    """

    def __init__(self, method, hidden_size, device):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.device = device
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, out_state, history):
        """[summary]

        Args:
            out_state (tensor): batch_size * state_len * hidden_size
            history (tensor): batch_size * history_len * hiddden_size

        Returns:
            [tensor]: (batch_size, state_len, history_len)
        """
        if self.method == 'dot':
            history = history.permute(0, 2, 1)  # batch_size * hidden_size * history_len
            attn_energies = torch.bmm(out_state, history)
        elif self.method == 'general':
            history = self.attn(history)
            history = history.permute(0, 2, 1)
            attn_energies = torch.bmm(out_state, history)
        return F.softmax(attn_energies, dim=2)


class DeepMove(AbstractModel):
    """rnn model with long-term history attention"""

    def __init__(self, config, data_feature):
        super(DeepMove, self).__init__(config, data_feature)
        self.loc_size = data_feature['loc_size']
        self.loc_emb_size = config['loc_emb_size']
        self.tim_size = data_feature['tim_size']
        self.tim_emb_size = config['tim_emb_size']
        self.hidden_size = config['hidden_size']
        self.attn_type = config['attn_type']
        self.device = config['device']
        self.rnn_type = config['rnn_type']
        self.evaluate_method = config['evaluate_method']

        self.emb_loc = nn.Embedding(
            self.loc_size, self.loc_emb_size,
            padding_idx=data_feature['loc_pad'])
        self.emb_tim = nn.Embedding(
            self.tim_size, self.tim_emb_size,
            padding_idx=data_feature['tim_pad'])

        input_size = self.loc_emb_size + self.tim_emb_size
        self.attn = Attn(self.attn_type, self.hidden_size, self.device)
        self.fc_attn = nn.Linear(input_size, self.hidden_size)

        if self.rnn_type == 'GRU':
            self.rnn_encoder = nn.GRU(input_size, self.hidden_size, 1)
            self.rnn_decoder = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn_encoder = nn.LSTM(input_size, self.hidden_size, 1)
            self.rnn_decoder = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn_encoder = nn.RNN(input_size, self.hidden_size, 1)
            self.rnn_decoder = nn.LSTM(input_size, self.hidden_size, 1)

        self.fc_final = nn.Linear(2 * self.hidden_size, self.loc_size)
        self.dropout = nn.Dropout(p=config['dropout_p'])
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for
        consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters()
              if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters()
              if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters()
             if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, batch):
        loc = batch['current_loc']
        tim = batch['current_tim']
        history_loc = batch['history_loc']
        history_tim = batch['history_tim']
        loc_len = batch.get_origin_len('current_loc')
        history_len = batch.get_origin_len('history_loc')
        batch_size = loc.shape[0]
        h1 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        h2 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        c1 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        c2 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        # change batch * seq * input_size to seq * batch * input_size
        x = torch.cat((loc_emb, tim_emb), 2).permute(1, 0, 2)
        x = self.dropout(x)

        history_loc_emb = self.emb_loc(history_loc)
        history_tim_emb = self.emb_tim(history_tim)
        history_x = torch.cat(
            (history_loc_emb, history_tim_emb), 2).permute(1, 0, 2)
        history_x = self.dropout(history_x)

        # pack x and history_x
        pack_x = pack_padded_sequence(x, lengths=loc_len, enforce_sorted=False)
        pack_history_x = pack_padded_sequence(
            history_x, lengths=history_len, enforce_sorted=False)
        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            hidden_history, h1 = self.rnn_encoder(pack_history_x, h1)
            hidden_state, h2 = self.rnn_decoder(pack_x, h2)
        elif self.rnn_type == 'LSTM':
            hidden_history, (h1, c1) = self.rnn_encoder(
                pack_history_x, (h1, c1))
            hidden_state, (h2, c2) = self.rnn_decoder(pack_x, (h2, c2))
        # unpack
        hidden_history, hidden_history_len = pad_packed_sequence(
            hidden_history, batch_first=True)
        hidden_state, hidden_state_len = pad_packed_sequence(
            hidden_state, batch_first=True)
        # batch_size * state_len * history_len
        attn_weights = self.attn(hidden_state, hidden_history)
        # batch_size * state_len * input_size
        context = attn_weights.bmm(hidden_history)
        # batch_size * state_len * 2 x input_size
        out = torch.cat((hidden_state, context), 2)
        # 因为是补齐了的，所以需要找到真正的 out
        origin_len = batch.get_origin_len('current_loc')
        final_out_index = torch.tensor(origin_len) - 1
        final_out_index = final_out_index.reshape(final_out_index.shape[0], 1, -1)
        final_out_index = final_out_index.repeat(1, 1, 2*self.hidden_size).to(self.device)
        out = torch.gather(out, 1, final_out_index).squeeze(1)  # batch_size * (2*hidden_size)
        out = self.dropout(out)

        y = self.fc_final(out)  # batch_size * loc_size
        score = F.log_softmax(y, dim=1)
        return score

    def predict(self, batch):
        score = self.forward(batch)
        if self.evaluate_method == 'sample':
            # build pos_neg_inedx
            pos_neg_index = torch.cat((batch['target'].unsqueeze(1), batch['neg_loc']), dim=1)
            score = torch.gather(score, 1, pos_neg_index)
        return score

    def calculate_loss(self, batch):
        criterion = nn.NLLLoss().to(self.device)
        scores = self.forward(batch)
        return criterion(scores, batch['target'])
