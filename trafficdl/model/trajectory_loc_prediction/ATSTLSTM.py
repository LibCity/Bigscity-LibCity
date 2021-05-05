from trafficdl.model.abstract_model import AbstractModel

import torch.nn as nn
import torch
from math import sqrt
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class Attn(nn.Module):
    """ Attention 注意力机制模块, 对 LSTM 中间层输出做加权平均. """

    def __init__(self, hidden_size):
        """ 初始化.
        Args:
            hidden_size (int): 中间层输出向量的大小
        """

        super(Attn, self).__init__()
        self.sqrt_rec_size = 1. / sqrt(hidden_size)
        self.linear = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        """ 前向传播.
        Args:
            x (torch.tensor.Tensor): shape (batch, seq_len, hidden_size) 中间层输出序列
        Returns:
            (torch.tensor.Tensor): shape (batch, size)
        """
        w = self.linear(x) * self.sqrt_rec_size
        w = self.softmax(w)
        c = w.bmm(x)
        return c


class ATSTLSTM(AbstractModel):
    """ ATST_LSTM 轨迹下一跳预测模型. """

    def __init__(self, config, data_feature):
        """ 模型初始化.
        Args:
            config: useless
            data_feature: useless
        """

        super(ATSTLSTM, self).__init__(config, data_feature)
        self.hidden_size = config['hidden_size']
        self.loc_size = data_feature['loc_size']
        self.uid_size = data_feature['uid_size']
        self.device = config['device']
        # 构建网络
        self.loc_embedding = nn.Embedding(num_embeddings=self.loc_size, embedding_dim=self.hidden_size,
                                          padding_idx=data_feature['loc_pad'])
        self.user_embedding = nn.Embedding(num_embeddings=self.uid_size, embedding_dim=self.hidden_size)
        # Wv
        self.wv = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        # Wl
        self.wl = nn.Linear(in_features=1, out_features=self.hidden_size, bias=False)
        # Wt
        self.wt = nn.Linear(in_features=1, out_features=self.hidden_size, bias=False)
        # Wn
        self.wn = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        # Wp
        self.wp = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size)
        self.attn = Attn(size=self.hidden_size)

    def forward(self, batch):
        # batch_size * padded_seq_len
        loc = batch['loc']
        dis = batch['dis']
        tim = batch['tim']
        # batch_size * num_samples
        loc_neg = batch['loc_neg']
        dis_neg = batch['dis_neg']
        tim_neg = batch['tim_neg']
        # batch_size * 1
        uid = batch['uid']
        target_loc = batch['target_loc']
        target_dis = batch['target_dis']
        target_tim = batch['target_tim']
        origin_len = batch.get_origin_len('loc')
        padded_seq_len = loc.shape[1]
        # concat all input to do embedding
        total_loc = torch.cat([loc, target_loc, loc_neg], dim=1)
        total_dis = torch.cat([dis, target_dis, dis_neg], dim=1)
        total_tim = torch.cat([tim, target_tim, tim_neg], dim=1)
        # embedding
        total_loc_emb = self.loc_embedding(total_loc)  # batch_size * total_len * hidden_size
        total_emb = self.wv(total_loc_emb) + self.wl(total_dis) + self.wt(total_tim)
        # split emb
        current_emb, rest_emb = torch.split(total_emb, [padded_seq_len, total_emb.shape[1] - padded_seq_len], dim=1)
        # lstm
        pack_current_emb = pack_padded_sequence(current_emb, lengths=origin_len, enforce_sorted=False)
        lstm_out, (h_n, c_n) = self.lstm(pack_current_emb)
        lstm_out, out_len = pad_packed_sequence(lstm_out, batch_first=True)
        # attn
        attn_out = self.attn(lstm_out)  # batch_size * padded_seq_len * hidden_size
        # get the last time slot's attn_out
        last_slot_index = torch.tensor(origin_len) - 1
        last_slot_index = last_slot_index.reshape(last_slot_index.shape[0], 1, -1)
        last_slot_index = last_slot_index.repeat(1, 1, self.hidden_size).to(self.device)
        rn = torch.gather(attn_out, 1, last_slot_index).squeeze(1)  # batch_size *  hidden_size
        # get user laten vec
        pu = self.user_embedding(uid)  # batch_size * hidden_size
        # first output (wn*rn + wp * pu)
        first_part = self.wn(rn) + self.wp(pu)  # batch_size * hidden_size
        first_part = first_part.unsqueeze(2)  # batch_size * hidden_size * 1
        output = torch.bmm(rest_emb, first_part).squeeze(2)  # batch_size * (num_samples+1)
        return output

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        """ 计算模型损失（不包含正则项）
        Args:
            batch (trafficdl.data.batch): 输入
        Returns:
            (torch.tensor.Tensor): shape () 损失
        """

        score = self.predict(batch)
        score_pos, score_neg = torch.split(score, [1, score.shape[1] - 1], dim=1)
        # score_pos is batch_size * 1
        # score_neg is batch_size * num_samples
        loss = torch.sum(torch.log(torch.add(torch.exp(torch.sub(score_neg, score_pos)), 1)))
        return loss
