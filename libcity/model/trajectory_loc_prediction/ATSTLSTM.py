from libcity.model.abstract_model import AbstractModel

import torch.nn as nn
import torch
from math import sqrt
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.functional import normalize


class Attn(nn.Module):
    """ Attention 注意力机制模块, 对 LSTM 中间层输出做加权平均. """

    def __init__(self, hidden_size):
        """ 初始化.
        Args:
            hidden_size (int): 中间层输出向量的大小
        """

        super(Attn, self).__init__()
        self.sqrt_rec_size = 1. / sqrt(hidden_size)
        # context vector
        self.zu = nn.Linear(in_features=hidden_size, out_features=1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        """ 前向传播.
        Args:
            x (torch.tensor.Tensor): shape (batch, seq_len, hidden_size) 中间层输出序列
        Returns:
            (torch.tensor.Tensor): shape (batch, size)
        """
        w = self.zu(x) * self.sqrt_rec_size
        w = w.permute(0, 2, 1)
        w = self.softmax(w)  # batch_size * 1 *seq_len
        c = torch.bmm(w, x)
        return c.squeeze(1)


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
        self.attn = Attn(hidden_size=self.hidden_size)

    def forward(self, batch):
        # batch_size * padded_seq_len
        loc = batch['current_loc']
        dis = batch['current_dis']
        tim = batch['current_tim']
        # batch_size * neg_samples
        loc_neg = batch['loc_neg']
        dis_neg = batch['dis_neg']
        tim_neg = batch['tim_neg']
        # batch_size
        uid = batch['uid']
        # batch_size * 1
        target_loc = batch['target_loc'].unsqueeze(1)
        target_dis = batch['target_dis'].unsqueeze(1)
        target_tim = batch['target_tim'].unsqueeze(1)
        origin_len = batch.get_origin_len('current_loc')
        padded_seq_len = loc.shape[1]
        # concat all input to do embedding
        total_loc = torch.cat([loc, target_loc, loc_neg], dim=1)
        total_dis = torch.cat([dis, target_dis, dis_neg], dim=1).unsqueeze(2)
        total_tim = torch.cat([tim, target_tim, tim_neg], dim=1).unsqueeze(2)
        # embedding
        total_loc_emb = self.loc_embedding(total_loc)  # batch_size * total_len * hidden_size
        total_emb = self.wv(total_loc_emb) + self.wl(total_dis) + self.wt(total_tim)
        # split emb
        current_emb, rest_emb = torch.split(total_emb, [padded_seq_len, total_emb.shape[1] - padded_seq_len], dim=1)
        # lstm
        pack_current_emb = pack_padded_sequence(current_emb, lengths=origin_len, enforce_sorted=False, batch_first=True)
        lstm_out, (h_n, c_n) = self.lstm(pack_current_emb)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        # attn
        rn = self.attn(lstm_out)  # batch_size * hidden_size
        # get user laten vec
        pu = self.user_embedding(uid)  # batch_size * hidden_size
        # first output (wn*rn + wp * pu)
        first_part = self.wn(rn) + self.wp(pu)  # batch_size * hidden_size
        first_part = first_part.unsqueeze(2)  # batch_size * hidden_size * 1
        output = torch.bmm(rest_emb, first_part).squeeze(2)  # batch_size * (neg_samples+1)

        return output

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        """ 计算模型损失（不包含正则项）
        Args:
            batch (libcity.data.batch): 输入
        Returns:
            (torch.tensor.Tensor): shape () 损失
        """

        score = self.predict(batch)
        # 这里需要对 score 进行一个归一化，不然 loss 会变成 inf
        score = normalize(score, dim=1)
        score_pos, score_neg = torch.split(score, [1, score.shape[1] - 1], dim=1)
        # score_pos is batch_size * 1
        # score_neg is batch_size * neg_samples
        loss = -(score_pos - score_neg).sigmoid().log().sum()
        return loss
