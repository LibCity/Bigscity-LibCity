# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

# ############# rnn model with attention ####################### #
class Attn(nn.Module):
    """Attention Module. Heavily borrowed from Practical Pytorch
    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation"""

    def __init__(self, method, hidden_size, use_cuda):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, out_state, history):
        seq_len = history.size()[1]
        state_len = out_state.size()[1]
        batch_size = history.size()[0]
        
        if self.use_cuda:
            attn_energies = torch.zeros(batch_size, state_len, seq_len).cuda()
        else:
            attn_energies = torch.zeros(batch_size, state_len, seq_len)
        for i in range(state_len):
            for j in range(seq_len):
                for k in range(batch_size):
                    attn_energies[k, i, j] = self.score(out_state[k][i], history[k][j])
        return F.softmax(attn_energies, dim = 2)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output)))
            energy = self.other.dot(energy)
            return energy


# ##############long###########################
class TrajPreAttnAvgLongUser(nn.Module):
    """rnn model with long-term history attention"""

    def __init__(self, parameters):
        super(TrajPreAttnAvgLongUser, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.uid_size = parameters.uid_size
        self.uid_emb_size = parameters.uid_emb_size
        self.hidden_size = parameters.hidden_size
        self.attn_type = parameters.attn_type
        self.rnn_type = parameters.rnn_type
        self.use_cuda = parameters.use_cuda

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
        self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size
        self.attn = Attn(self.attn_type, self.hidden_size, self.use_cuda)
        self.fc_attn = nn.Linear(input_size, self.hidden_size)

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, 1)

        self.fc_final = nn.Linear(2 * self.hidden_size + self.uid_emb_size, self.loc_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, loc, tim, history_loc, history_tim, history_count, uid, target_len):
        h1 = Variable(torch.zeros(1, 1, self.hidden_size))
        c1 = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            h1 = h1.cuda()
            c1 = c1.cuda()

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)

        loc_emb_history = self.emb_loc(history_loc).squeeze(1) # 去掉维数为 1 的维度
        tim_emb_history = self.emb_tim(history_tim).squeeze(1)
        count = 0
        if self.use_cuda:
            loc_emb_history2 = Variable(torch.zeros(len(history_count), loc_emb_history.size()[-1])).cuda()
            tim_emb_history2 = Variable(torch.zeros(len(history_count), tim_emb_history.size()[-1])).cuda()
        else:
            loc_emb_history2 = Variable(torch.zeros(len(history_count), loc_emb_history.size()[-1]))
            tim_emb_history2 = Variable(torch.zeros(len(history_count), tim_emb_history.size()[-1]))
        for i, c in enumerate(history_count):
            if c == 1:
                tmp = loc_emb_history[count].unsqueeze(0) # shape: 1 * 500 
            else:
                tmp = torch.mean(loc_emb_history[count:count + c, :], dim=0, keepdim=True) # 为什么要 mean 一下呢 因为要把相邻且处于时间段的数据合到一起
            loc_emb_history2[i, :] = tmp
            tim_emb_history2[i, :] = tim_emb_history[count, :].unsqueeze(0) # 1 * 
            count += c

        history = torch.cat((loc_emb_history2, tim_emb_history2), 1)
        history = F.tanh(self.fc_attn(history))

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            out_state, h1 = self.rnn(x, h1)
        elif self.rnn_type == 'LSTM':
            out_state, (h1, c1) = self.rnn(x, (h1, c1))
        out_state = out_state.squeeze(1) # seq_len * hiden_size
        # out_state = F.selu(out_state)

        attn_weights = self.attn(out_state[-target_len:], history).unsqueeze(0) # 为什么这里要用 target len 截取一下 因为 x[0] 投入 RNN 出来的是 x_expect[1]
        context = attn_weights.bmm(history.unsqueeze(0)).squeeze(0) # 把 history 与对应的 weight 相乘 shape: target_len * hiden_size
        out = torch.cat((out_state[-target_len:], context), 1)  # no need for fc_attn

        uid_emb = self.emb_uid(uid).repeat(target_len, 1) # 补充好维数
        out = torch.cat((out, uid_emb), 1)
        out = self.dropout(out)

        y = self.fc_final(out) # shape target_len * loc_size
        score = F.log_softmax(y, 1) # 这个 score 这里最好指明 dim 不然会报警，我觉得应该是 1

        return score


class DeepMove(nn.Module):
    """rnn model with long-term history attention"""

    def __init__(self, config):
        super(DeepMove, self).__init__()
        self.loc_size = config['data_feature']['loc_size']
        self.loc_emb_size = config['loc_emb_size']
        self.tim_size = config['data_feature']['tim_size']
        self.tim_emb_size = config['tim_emb_size']
        self.hidden_size = config['hidden_size']
        self.attn_type = config['attn_type']
        self.use_cuda = config['use_cuda']
        self.rnn_type = config['rnn_type']

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size
        self.attn = Attn(self.attn_type, self.hidden_size, self.use_cuda)
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
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, batch):
        loc = batch['current_loc']
        tim = batch['current_tim']
        history_loc = batch['history_loc']
        history_tim = batch['history_tim']
        loc_len = batch.get_origin_len('current_loc')
        history_len = batch.get_origin_len('history_loc')
        batch_size = loc.shape[0]
        h1 = torch.zeros(1, batch_size, self.hidden_size)
        h2 = torch.zeros(1, batch_size, self.hidden_size)
        c1 = torch.zeros(1, batch_size, self.hidden_size)
        c2 = torch.zeros(1, batch_size, self.hidden_size)
        if self.use_cuda:
            h1 = h1.cuda()
            h2 = h2.cuda()
            c1 = c1.cuda()
            c2 = c2.cuda()

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        x = torch.cat((loc_emb, tim_emb), 2).permute(1, 0, 2) # change batch * seq * input_size to seq * batch * input_size
        x = self.dropout(x)

        history_loc_emb = self.emb_loc(history_loc)
        history_tim_emb = self.emb_tim(history_tim)
        history_x = torch.cat((history_loc_emb, history_tim_emb), 2).permute(1, 0, 2)
        history_x = self.dropout(history_x)

        # pack x and history_x
        pack_x = pack_padded_sequence(x, lengths=loc_len, enforce_sorted=False)
        pack_history_x = pack_padded_sequence(history_x, lengths=history_len, enforce_sorted=False)
        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            hidden_history, h1 = self.rnn_encoder(pack_history_x, h1)
            hidden_state, h2 = self.rnn_decoder(pack_x, h2)
        elif self.rnn_type == 'LSTM':
            hidden_history, (h1, c1) = self.rnn_encoder(pack_history_x, (h1, c1))
            hidden_state, (h2, c2) = self.rnn_decoder(pack_x, (h2, c2))
        #unpack
        hidden_history, hidden_history_len = pad_packed_sequence(hidden_history, batch_first=True)
        hidden_state, hidden_state_len = pad_packed_sequence(hidden_state, batch_first=True)
        # hidden_history = hidden_history.permute(1, 0, 2) # change history_len * batch_size * input_size to batch_size * history_len * input_size
        # hidden_state = hidden_state.permute(1, 0, 2)
        attn_weights = self.attn(hidden_state, hidden_history) # batch_size * state_len * history_len
        context = attn_weights.bmm(hidden_history) # batch_size * state_len * input_size
        out = torch.cat((hidden_state, context), 2)  # batch_size * state_len * 2 x input_size
        out = self.dropout(out)

        y = self.fc_final(out) # batch_size * state_len * loc_size
        score = F.log_softmax(y, dim=2)
        # 因为是补齐了的，所以需要找到真正的 score
        for i in range(score.shape[0]):
            if i == 0:
                true_scores = score[i][loc_len[i] - 1].reshape(1, -1)
            else:
                true_scores = torch.cat((true_scores, score[i][loc_len[i] - 1].reshape(1, -1)), 0)
        return true_scores
