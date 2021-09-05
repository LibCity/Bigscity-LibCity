import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from libcity.model.abstract_model import AbstractModel


class LSTPM(AbstractModel):
    """
    reference: https://github.com/NLPWM-WHU/LSTPM
    """

    def __init__(self, config, data_feature):
        super(LSTPM, self).__init__(config, data_feature)
        self.loc_size = data_feature['loc_size']
        self.tim_size = data_feature['tim_size']
        self.hidden_size = config['hidden_size']
        self.emb_size = config['emb_size']
        self.device = config['device']
        # todo why embeding?
        self.loc_emb = nn.Embedding(self.loc_size, self.emb_size, padding_idx=data_feature['loc_pad'])
        # self.emb_tim = nn.Embedding(self.tim_size, 10) 根本就没有用时间???这都能 soda ??
        self.lstmcell = nn.LSTM(input_size=self.emb_size,
                                hidden_size=self.hidden_size)
        self.lstmcell_history = nn.LSTM(
            input_size=self.emb_size, hidden_size=self.hidden_size)
        self.linear = nn.Linear(self.hidden_size * 2, self.loc_size, bias=False)
        self.dropout = nn.Dropout(config['dropout'])
        # self.user_dropout = nn.Dropout(user_dropout) 不是这模型对吗？？
        self.tim_sim_matrix = data_feature['tim_sim_matrix']
        # could be the same as self.lstmcell
        self.dilated_rnn = nn.LSTMCell(
            input_size=self.emb_size, hidden_size=self.hidden_size)
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.init_weights()

    def init_weights(self):
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

    def _pad_batch_of_lists_masks(self, current_session, origin_len):
        # 因为没有 pad 所以 max_len == len(l)
        padde_mask_non_local = [1.0] * (origin_len) + [0.0] * (len(current_session) - origin_len)
        padde_mask_non_local = torch.FloatTensor(padde_mask_non_local).to(self.device)
        return padde_mask_non_local

    def forward(self, batch):
        batch_size = batch['current_loc'].shape[0]
        origin_len = batch.get_origin_len('current_loc')
        current_loc = batch['current_loc']
        current_tim = batch['current_tim']
        items = self.loc_emb(current_loc).permute(1, 0, 2)  # sequence * batch_size * embedding
        current_loc = current_loc.tolist()
        # pack x and history_x
        pack_items = pack_padded_sequence(items, lengths=origin_len, enforce_sorted=False)
        h1 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        c1 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        out, (h1, c1) = self.lstmcell(pack_items, (h1, c1))
        # batch_size * sequence_length * hidden_size
        out, _ = pad_packed_sequence(out, batch_first=True)
        items = items.permute(1, 0, 2)  # sequence * batch_size * embeeding
        y_list = []
        out_hie = []  # batch_size * hidden_size
        dilated_rnn_input_index = batch['dilated_rnn_input_index']
        for ii in range(batch_size):
            current_session_input_dilated_rnn_index = dilated_rnn_input_index[ii].tolist()  # origin_cur_len
            hiddens_current = items[ii]
            dilated_lstm_outs_h = []
            dilated_lstm_outs_c = []
            for index_dilated in range(len(current_session_input_dilated_rnn_index)):
                index_dilated_explicit = current_session_input_dilated_rnn_index[index_dilated]
                hidden_current = hiddens_current[index_dilated].unsqueeze(0)
                if index_dilated == 0:
                    h = torch.zeros(1, self.hidden_size).to(self.device)
                    c = torch.zeros(1, self.hidden_size).to(self.device)
                    (h, c) = self.dilated_rnn(hidden_current, (h, c))
                    dilated_lstm_outs_h.append(h)
                    dilated_lstm_outs_c.append(c)
                else:
                    (h, c) = self.dilated_rnn(hidden_current, (
                            dilated_lstm_outs_h[index_dilated_explicit],
                            dilated_lstm_outs_c[index_dilated_explicit]))
                    dilated_lstm_outs_h.append(h)
                    dilated_lstm_outs_c.append(c)
            out_hie.append(dilated_lstm_outs_h[-1])
            current_session_timid = current_tim[ii].tolist()[origin_len[ii] - 1]  # 不包含我 pad 的那个点
            current_session_embed = out[ii]  # sequence_len * hidden_size
            # FloatTensor sequence_len * 1
            current_session_mask = self._pad_batch_of_lists_masks(current_loc[ii], origin_len[ii]).unsqueeze(1)
            # mask_batch_ix_non_local[ii].unsqueeze(1)
            sequence_length = origin_len[ii]
            # do average pooling for current_session
            # 1 * hidden_size
            current_session_represent = torch.sum(current_session_embed * current_session_mask,
                                                  dim=0).unsqueeze(0)/sum(current_session_mask)
            list_for_sessions = []  # his_cnt * hidden_size
            h2 = torch.zeros(1, 1, self.hidden_size).to(self.device)
            c2 = torch.zeros(1, 1, self.hidden_size).to(self.device)
            # 处理历史轨迹
            for jj in range(len(batch['history_loc'][ii])):
                sequence = batch['history_loc'][ii][jj]
                sequence_emb = self.loc_emb(sequence).unsqueeze(1)  # his_seq_len * 1 * embedding_size
                sequence_emb, (h2, c2) = self.lstmcell_history(sequence_emb, (h2, c2))
                sequence_tim_id = batch['history_tim'][ii][jj].tolist()
                # 根据 time slot 相似度修正历史轨迹表征
                # tim_size
                jaccard_sim_row = torch.FloatTensor(self.tim_sim_matrix[current_session_timid]).to(self.device)
                jaccard_sim_expicit = jaccard_sim_row[sequence_tim_id]  # his_seq_len
                jaccard_sim_expicit_last = F.softmax(jaccard_sim_expicit, dim=0).unsqueeze(0)  # 1 * his_seq_len
                # 1 * hidden_size
                hidden_sequence_for_current = torch.mm(jaccard_sim_expicit_last, sequence_emb.squeeze(1))
                list_for_sessions.append(hidden_sequence_for_current)
            # 1 * his_cnt
            avg_distance = batch['history_avg_distance'][ii].unsqueeze(0)
            # 1 * his_cnt * hidden_size
            sessions_represent = torch.cat(list_for_sessions, dim=0).unsqueeze(0)
            # 1 * hidden_size * 1
            current_session_represent = current_session_represent.unsqueeze(2)
            # 1 * 1 * his_cnt
            sim_between_cur_his = F.softmax(sessions_represent.bmm(current_session_represent).squeeze(2),
                                            dim=1).unsqueeze(1)
            # TODO: why do linear1 and selu?
            # 1 * hidden_size
            out_y_current = torch.selu(self.linear1(sim_between_cur_his.bmm(sessions_represent).squeeze(1)))
            # 1 * hidden_size * 1
            layer_2_current = (0.5 * out_y_current + 0.5 * current_session_embed[sequence_length - 1]).unsqueeze(2)
            # 1 * 1 * his_cnt
            layer_2_sims = F.softmax(sessions_represent.bmm(layer_2_current).squeeze(2) *
                                     1.0 / avg_distance, dim=1).unsqueeze(1)
            # 1 * hidden_size
            out_layer_2 = layer_2_sims.bmm(sessions_represent).squeeze(1)
            y_list.append(out_layer_2)
        # batch_size * hidden_size
        y = torch.selu(torch.cat(y_list, dim=0))
        # 得到 shor-term 的输出
        # batch_size * hidden_size
        out_hie = F.selu(torch.cat(out_hie, dim=0))
        # get the final lstm out
        final_out_index = torch.tensor(origin_len) - 1
        final_out_index = final_out_index.reshape(final_out_index.shape[0], 1, -1)
        final_out_index = final_out_index.repeat(1, 1, self.hidden_size).to(self.device)
        final_out = torch.gather(out, 1, final_out_index).squeeze(1)
        final_out = F.selu(final_out)
        final_out = (final_out + out_hie) * 0.5
        out_put_emb_v1 = torch.cat([y, final_out], dim=1)
        output_ln = self.linear(out_put_emb_v1)
        # batch_size * loc_size
        output = F.log_softmax(output_ln, dim=1)
        return output

    def calculate_loss(self, batch):
        criterion = nn.NLLLoss(reduction='sum').to(self.device)
        scores = self.forward(batch)
        return criterion(scores, batch['target'])

    def predict(self, batch):
        return self.forward(batch)
