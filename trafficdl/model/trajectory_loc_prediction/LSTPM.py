import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from geopy import distance

from trafficdl.model.abstract_model import AbstractModel
from trafficdl.utils.dataset import parse_coordinate


class LSTPM(AbstractModel):
    """
    reference: https://github.com/NLPWM-WHU/LSTPM
    """

    def __init__(self, config, data_feature):
        super(LSTPM, self).__init__(config, data_feature)
        self.loc_size = data_feature['loc_size']
        self.tim_size = data_feature['tim_size']
        self.poi_profile = data_feature['poi_profile']
        self.hidden_size = config['hidden_size']
        self.emb_size = config['emb_size']
        self.device = config['device']
        # todo why embeding?
        self.item_emb = nn.Embedding(
            self.loc_size, self.emb_size, padding_idx=data_feature['loc_pad'])
        # self.emb_tim = nn.Embedding(self.tim_size, 10) 根本就没有用时间???这都能 soda ??
        self.lstmcell = nn.LSTM(input_size=self.emb_size,
                                hidden_size=self.hidden_size)
        self.lstmcell_history = nn.LSTM(
            input_size=self.emb_size, hidden_size=self.hidden_size)
        self.linear = nn.Linear(self.hidden_size * 2, self.loc_size)
        self.dropout = nn.Dropout(config['dropout'])
        # self.user_dropout = nn.Dropout(user_dropout) 不是这模型对吗？？
        self.tim_sim_matrix = data_feature['tim_sim_matrix']
        # could be the same as self.lstmcell
        self.dilated_rnn = nn.LSTMCell(
            input_size=self.emb_size, hidden_size=self.hidden_size)
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.init_weights()

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters()
              if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters()
              if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters()
             if 'bias' in name)
        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def _create_dilated_rnn_input(self, session_sequence_current_padded,
                                  sequence_length):
        session_sequence_current = \
            session_sequence_current_padded[:sequence_length]
        session_sequence_current.reverse()
        session_dilated_rnn_input_index = [0] * sequence_length
        for i in range(sequence_length - 1):
            current_poi = session_sequence_current[i]
            poi_before = session_sequence_current[i + 1:]
            current_poi_profile = self.poi_profile.iloc[current_poi]
            lon_cur, lat_cur = parse_coordinate(
                current_poi_profile['coordinates'])
            distance_row_explicit = []
            for target in poi_before:
                lon, lat = parse_coordinate(
                    self.poi_profile.iloc[target]['coordinates'])
                distance_row_explicit.append(distance.distance(
                    (lat_cur, lon_cur), (lat, lon)).kilometers)
            # distance_row = self.poi_distance_matrix[current_poi]
            # distance_row_explicit = distance_row[:, poi_before][0]
            index_closet = np.argmin(distance_row_explicit)
            session_dilated_rnn_input_index[sequence_length -
                                            i - 1] = sequence_length - 2 - \
                index_closet - i
        session_sequence_current.reverse()
        return session_dilated_rnn_input_index

    def _pad_batch_of_lists_masks(self, current_session, origin_len):
        # 因为没有 pad 所以 max_len == len(l)
        padde_mask_non_local = [
            1.0] * (origin_len) + [0.0] * (len(current_session) - origin_len)
        padde_mask_non_local = torch.FloatTensor(
            padde_mask_non_local).to(self.device)
        return padde_mask_non_local

    def forward(self, batch):
        # 因为源码是假设的 target 就是 current_loc 的最后一个点
        # 所以需要做一下修改，把 target append 到 current_loc 里面
        batch_size = batch['current_loc'].shape[0]
        pad_loc = torch.LongTensor(
            [batch.pad_item['current_loc']] * batch_size).unsqueeze(1).to(
                self.device)
        pad_tim = torch.LongTensor(
            [batch.pad_item['current_tim']] * batch_size).unsqueeze(1).to(
                self.device)
        expand_current_loc = torch.cat([batch['current_loc'], pad_loc], dim=1)
        expand_current_tim = torch.cat([batch['current_tim'], pad_tim], dim=1)
        origin_len = batch.get_origin_len('current_loc').copy()
        for i in range(batch_size):
            origin_len[i] += 1
            expand_current_loc[i][origin_len[i] - 1] = batch['target'][i]
        # batch['current_loc'] = expand_current_loc
        # batch['current_tim'] = expand_current_tim
        item_vectors = expand_current_loc
        sequence_size = item_vectors.size()[1]
        items = self.item_emb(item_vectors)
        item_vectors = item_vectors.tolist()
        x = items  # batch_size * sequence * embedding
        x = x.transpose(0, 1)
        h1 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        c1 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        out, (h1, c1) = self.lstmcell(x, (h1, c1))
        # batch_size * sequence_length * embedding_dim
        out = out.transpose(0, 1)
        x1 = items
        # ###########################################################
        y_list = []
        out_hie = []
        for ii in range(batch_size):
            ##########################################
            current_session_input_dilated_rnn_index = \
                self._create_dilated_rnn_input(
                    expand_current_loc[ii].tolist(), origin_len[ii])
            hiddens_current = x1[ii]
            dilated_lstm_outs_h = []
            dilated_lstm_outs_c = []
            for index_dilated in range(
                    len(current_session_input_dilated_rnn_index)):
                index_dilated_explicit = \
                    current_session_input_dilated_rnn_index[index_dilated]
                hidden_current = hiddens_current[index_dilated].unsqueeze(0)
                if index_dilated == 0:
                    h = torch.zeros(1, self.hidden_size).to(self.device)
                    c = torch.zeros(1, self.hidden_size).to(self.device)
                    (h, c) = self.dilated_rnn(hidden_current, (h, c))
                    dilated_lstm_outs_h.append(h)
                    dilated_lstm_outs_c.append(c)
                else:
                    (h, c) = self.dilated_rnn(
                        hidden_current, (
                            dilated_lstm_outs_h[index_dilated_explicit],
                            dilated_lstm_outs_c[index_dilated_explicit]))
                    dilated_lstm_outs_h.append(h)
                    dilated_lstm_outs_c.append(c)
            dilated_lstm_outs_h.append(
                hiddens_current[len(current_session_input_dilated_rnn_index):])
            dilated_out = torch.cat(dilated_lstm_outs_h, dim=0).unsqueeze(0)
            out_hie.append(dilated_out)
            current_session_timid = expand_current_tim[ii].tolist(
            )[:origin_len[ii] - 1]  # 不包含我 pad 的那个点
            current_session_poiid = item_vectors[ii][:len(
                current_session_timid)]
            current_session_embed = out[ii]  # sequence_len * embedding_size
            current_session_mask = self._pad_batch_of_lists_masks(
                expand_current_loc[ii].cpu(),
                origin_len[ii]).unsqueeze(1)  # FloatTensor sequence_len * 1
            # mask_batch_ix_non_local[ii].unsqueeze(1)
            sequence_length = origin_len[ii]
            current_session_represent_list = []
            for iii in range(sequence_length - 1):
                current_session_represent = torch.sum(
                    current_session_embed *
                    current_session_mask, dim=0).unsqueeze(0) \
                    / sum(current_session_mask)
                current_session_represent_list.append(
                    current_session_represent)

            current_session_represent = torch.cat(
                current_session_represent_list, dim=0)
            list_for_sessions = []
            list_for_avg_distance = []
            h2 = torch.zeros(1, 1, self.hidden_size).to(
                self.device)  # whole sequence
            c2 = torch.zeros(1, 1, self.hidden_size).to(self.device)
            # 处理历史轨迹
            for jj in range(len(batch['history_loc'][ii])):
                sequence = batch['history_loc'][ii][jj]
                sequence_emb = self.item_emb(sequence).unsqueeze(
                    1)  # sequence_len * 1 * embedding_size
                sequence = sequence.tolist()
                sequence_emb, (h2, c2) = self.lstmcell_history(
                    sequence_emb, (h2, c2))
                sequence_tim_id = batch['history_tim'][ii][jj].tolist()
                # 相当于自己做了一个 tim 的表征
                jaccard_sim_row = torch.FloatTensor(
                    self.tim_sim_matrix[current_session_timid]).to(self.device)
                jaccard_sim_expicit = jaccard_sim_row[:, sequence_tim_id]
                # 使用 profile 计算局部距离矩阵
                distance_matrix = []
                for origin in current_session_poiid:
                    lon_cur, lat_cur = parse_coordinate(
                        self.poi_profile.iloc[origin]['coordinates'])
                    distance_row = []
                    for target in sequence:
                        lon, lat = parse_coordinate(
                            self.poi_profile.iloc[target]['coordinates'])
                        distance_row.append(distance.distance(
                            (lat_cur, lon_cur), (lat, lon)).kilometers)
                    distance_matrix.append(distance_row)
                distance_row_expicit = torch.FloatTensor(
                    distance_matrix).to(self.device)
                distance_row_expicit_avg = torch.mean(
                    distance_row_expicit, dim=1)
                jaccard_sim_expicit_last = F.softmax(
                    jaccard_sim_expicit, dim=1)
                hidden_sequence_for_current1 = torch.mm(
                    jaccard_sim_expicit_last, sequence_emb.squeeze(1))
                hidden_sequence_for_current = hidden_sequence_for_current1
                list_for_sessions.append(
                    hidden_sequence_for_current.unsqueeze(0))
                list_for_avg_distance.append(
                    distance_row_expicit_avg.unsqueeze(0))
            avg_distance = torch.cat(
                list_for_avg_distance, dim=0).transpose(0, 1)
            # current_items * history_session_length * embedding_size
            sessions_represent = torch.cat(list_for_sessions, dim=0).transpose(
                0, 1)
            # current_items * embedding_size * 1
            current_session_represent = current_session_represent.unsqueeze(
                2)
            # ==> current_items * 1 * history_session_length
            sims = F.softmax(sessions_represent.bmm(
                current_session_represent).squeeze(2),
                dim=1).unsqueeze(1)
            out_y_current = torch.selu(self.linear1(
                sims.bmm(sessions_represent).squeeze(1)))
            layer_2_current = (0.5 * out_y_current + 0.5 *
                               current_session_embed[:sequence_length - 1]
                               ).unsqueeze(2)
            # ==>>current_items * 1 * history_session_length
            layer_2_sims = F.softmax(
                sessions_represent.bmm(layer_2_current).squeeze(2) *
                1.0 / avg_distance, dim=1).unsqueeze(1)
            out_layer_2 = layer_2_sims.bmm(sessions_represent).squeeze(1)
            # TODO: 感觉他这个是把 target 当成 current 的最后一个点了，有点问题
            out_y_current_padd = torch.FloatTensor(
                sequence_size - sequence_length + 1, self.emb_size).zero_().to(
                    self.device)
            out_layer_2_list = []
            out_layer_2_list.append(out_layer_2)
            out_layer_2_list.append(out_y_current_padd)
            out_layer_2 = torch.cat(out_layer_2_list, dim=0).unsqueeze(0)
            y_list.append(out_layer_2)
        y = torch.selu(torch.cat(y_list, dim=0))
        out_hie = F.selu(torch.cat(out_hie, dim=0))
        out = F.selu(out)
        out = (out + out_hie) * 0.5
        out_put_emb_v1 = torch.cat([y, out], dim=2)
        output_ln = self.linear(out_put_emb_v1)
        output = F.log_softmax(output_ln, dim=-1)
        return output

    def calculate_loss(self, batch):
        # 仍然不考虑做 batch 的情况
        origin_len = batch.get_origin_len('current_loc')
        sequence_len = batch['current_loc'].shape[1]
        mask_batch_ix = torch.FloatTensor(
            [[1.0] * (l) + [0.0] * (sequence_len - l + 1) for l in origin_len]
            ).to(self.device)
        logp_seq = self.forward(batch)
        predictions_logp = logp_seq[:, :-2] * mask_batch_ix[:, :-2, None]
        actual_next_tokens = batch['current_loc'][:, 1:]
        logp_next = torch.gather(
            predictions_logp, dim=2, index=actual_next_tokens[:, :, None])
        loss = -logp_next.sum() / mask_batch_ix[:, :-2].sum()
        # 目测 NaN 是由于学习率过高导致梯度消失等现象，解决办法是加 batch
        if torch.isnan(loss):
            # 将当前 batch 保存到本地
            torch.save(batch['current_loc'], 'current_loc.pt')
            torch.save(batch['current_tim'], 'current_tim.pt')
            torch.save(batch['uid'], 'uid.pt')
            torch.save(batch['history_loc'], "history_loc.pt")
            torch.save(batch['history_tim'], 'history_tim.pt')
            torch.save(batch['target'], 'target.pt')
            torch.save(self.state_dict(), 'model_state.m')
            print('loss is nan')
            exit()
        return loss

    def predict(self, batch):
        # 得拿到 true score
        output = self.forward(batch)
        origin_len = batch.get_origin_len('current_loc')
        for i in range(output.shape[0]):
            if i == 0:
                # 这里的 origin_len 并没有算新加的 target，所以 origin_len - 1 就是倒数第二个
                true_scores = output[i][origin_len[i] - 1].reshape(1, -1)
            else:
                true_scores = torch.cat(
                    (true_scores, output[i][origin_len[i] - 1].reshape(1, -1)),
                    0)
        return true_scores
