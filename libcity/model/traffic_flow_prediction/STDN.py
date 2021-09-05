from logging import getLogger
import torch
from torch import nn
from torch.nn import init
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def modify_input(input):
    output = input.reshape(-1, *input.shape[4:]).transpose(0, 1)
    return [output[i] for i in range(output.shape[0])]


class CBAAttention(nn.Module):
    def __init__(self, device, att_size, query_dim):
        super(CBAAttention, self).__init__()
        self.att_size = att_size
        self.query_dim = query_dim
        self.Wq = nn.Linear(query_dim, att_size).to(device)
        self.Wh = nn.Linear(att_size, att_size).to(device)
        self.v = nn.Linear(att_size, 1).to(device)
        self.tanh = nn.Tanh().to(device)
        self.softmax = nn.Softmax(dim=-1).to(device)
        self._glorot_init()

    def _glorot_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, init.calculate_gain('tanh'))

    def forward(self, inputs):
        assert isinstance(inputs, list) and len(inputs) == 2 and \
               inputs[0].shape[-1] == self.att_size and inputs[1].shape[-1] == self.query_dim
        memory, query = inputs
        n_memory = self.Wh(memory)
        n_query = self.Wq(query).unsqueeze(1)
        hidden = n_memory + n_query
        hidden = self.tanh(hidden)
        s = self.v(hidden).squeeze(-1)
        s = self.softmax(s)
        return torch.sum(memory * s.unsqueeze(-1), dim=1)


class STDN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super(STDN, self).__init__(config, data_feature)
        self.lstm_seq_len = config.get('input_window', 7)
        self.feature_vec_len = self.data_feature.get('feature_vec_len', 160)
        self.len_row = self.data_feature.get('len_row', 10)
        self.len_column = self.data_feature.get('len_column', 20)
        self.output_dim = self.data_feature.get('output_dim', 2)
        self.nbhd_size = 2 * config.get('cnn_nbhd_size', 3) + 1
        self.nbhd_type = self.data_feature.get('nbhd_type', 2)
        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))
        self.output_window = config.get('output_window', 1)
        self.att_lstm_num = config['att_lstm_num']
        self.att_lstm_seq_len = config['att_lstm_seq_len']
        self.cnn_flat_size = config.get('cnn_flat_size', 128)
        self.lstm_out_size = config.get('lstm_out_size', 128)
        self.max_x_num = config.get('max_x_num', 10)
        self.max_y_num = config.get('max_y_num', 20)
        self.flow_type = config.get('flow_type', 4)
        self._scaler = self.data_feature.get('scaler')
        self._init_model()

    def _init_model(self):
        '''
        input:
            flatten_att_nbhd_inputs: shape=[att_lstm_num * att_lstm_seq_len, (batch_size, nbhd_size, nbhd_size, nbhd_type)]
                -> att_nbhd_inputs: shape=[att_lstm_num, att_lstm_seq_len, (batch_size, nbhd_size, nbhd_size, nbhd_type)]
            flatten_att_flow_inputs: shape=[att_lstm_num * att_lstm_seq_len, (batch_size, nbhd_size, nbhd_size, flow_type)]
                -> att_flow_inputs: shape=[att_lstm_num, att_lstm_seq_len, (batch_size, nbhd_size, nbhd_size, flow_type)]
            att_lstm_inputs: shape=[att_lstm_num, (batch_size, att_lstm_seq_len, feature_vec_len)]
            nbhd_inputs: shaoe=[lstm_seq_len, (batch_size, nbhd_size, nbhd_size, nbhd_type)]
            flow_inputs: shape=[lstm_seq_len, (batch_size, nbhd_size, nbhd_size, flow_type)]
            lstm_inputs: shape=(batch_size, lstm_seq_len, feature_vec_len)

        remark:
            tensor part should have shape of (batch_size, input_channel, H, W), use permute
        '''
        # 1st level gate
        self.nbhd_cnns_1st = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=self.nbhd_type, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        ).to(self.device) for i in range(self.lstm_seq_len)])
        self.flow_cnns_1st = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=self.flow_type, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Sigmoid()
        ).to(self.device) for i in range(self.lstm_seq_len)])
        # [nbhd * flow] shape=[lstm_seq_len, (batch_size, 64, nbhd_size, nbhd_size)]

        # 2nd level gate
        self.nbhd_cnns_2nd = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        ).to(self.device) for i in range(self.lstm_seq_len)])
        self.flow_cnns_2nd = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=self.flow_type, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Sigmoid()
        ).to(self.device) for i in range(self.lstm_seq_len)])
        # [nbhd * flow] shape=[lstm_seq_len, (batch_size, 64, nbhd_size, nbhd_size)]

        # 3rd level gate
        self.nbhd_cnns_3rd = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        ).to(self.device) for i in range(self.lstm_seq_len)])
        self.flow_cnns_3rd = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=self.flow_type, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Sigmoid()
        ).to(self.device) for i in range(self.lstm_seq_len)])
        # [nbhd * flow] shape=[lstm_seq_len, (batch_size, 64, nbhd_size, nbhd_size)]

        # dense part
        self.nbhd_vecs = nn.ModuleList([nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * self.nbhd_size * self.nbhd_size, self.cnn_flat_size),
            nn.ReLU()
        ).to(self.device) for i in range(self.lstm_seq_len)])  # shape=[lstm_seq_len, (batch_size, cnn_flat_size)]

        # feature concatenate
        # torch.cat(list, dim=-1), shape=(batch_size, cnn_flat_size * lstm_seq_len)
        # torch.reshape(tensor, (tensor.shape[0], lstm_seq_len, cnn_flat_size))
        # torch.cat(list, dim=-1), shape=(batch_size, lstm_seq_len, feature_vec_len + cnn_flat_size)

        # lstm
        self.lstm = nn.LSTM(input_size=self.feature_vec_len + self.cnn_flat_size, hidden_size=self.lstm_out_size,
                            batch_first=True, dropout=0.1).to(self.device)
        # result shape=(batch_size, lstm_seq_len, lstm_out_size)
        # result, (hn, cn) = lstm -> hn[-1] shape=(batch, lstm_out_size)

        # attention part
        self.att_nbhd_cnns_1st = nn.ModuleList([nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=self.nbhd_type, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        ).to(self.device) for j in range(self.att_lstm_seq_len)]) for i in range(self.att_lstm_num)])
        self.att_flow_cnns_1st = nn.ModuleList([nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=self.flow_type, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        ).to(self.device) for j in range(self.att_lstm_seq_len)]) for i in range(self.att_lstm_num)])
        self.att_flow_gate_1st = nn.ModuleList(
            [nn.ModuleList([nn.Sigmoid().to(self.device) for j in range(self.att_lstm_seq_len)])
             for i in range(self.att_lstm_num)])
        # [[nbhd * flow]] shape=[att_lstm_num, att_lstm_seq_len, (batch_size, 64, nbhd_size, nbhd_size)]

        self.att_nbhd_cnns_2nd = nn.ModuleList([nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        ).to(self.device) for j in range(self.att_lstm_seq_len)]) for i in range(self.att_lstm_num)])
        self.att_flow_cnns_2nd = nn.ModuleList([nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        ).to(self.device) for j in range(self.att_lstm_seq_len)]) for i in range(self.att_lstm_num)])
        self.att_flow_gate_2nd = nn.ModuleList(
            [nn.ModuleList([nn.Sigmoid().to(self.device) for j in range(self.att_lstm_seq_len)])
             for i in range(self.att_lstm_num)])
        # [[nbhd * flow]] shape=[att_lstm_num, att_lstm_seq_len, (batch_size, 64, nbhd_size, nbhd_size)]

        self.att_nbhd_cnns_3rd = nn.ModuleList([nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        ).to(self.device) for j in range(self.att_lstm_seq_len)]) for i in range(self.att_lstm_num)])
        self.att_flow_cnns_3rd = nn.ModuleList([nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        ).to(self.device) for j in range(self.att_lstm_seq_len)]) for i in range(self.att_lstm_num)])
        self.att_flow_gate_3rd = nn.ModuleList(
            [nn.ModuleList([nn.Sigmoid().to(self.device) for j in range(self.att_lstm_seq_len)])
             for i in range(self.att_lstm_num)])
        # [[nbhd * flow]] shape=[att_lstm_num, att_lstm_seq_len, (batch_size, 64, nbhd_size, nbhd_size)]

        self.att_nbhd_vecs = nn.ModuleList([nn.ModuleList([nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * self.nbhd_size * self.nbhd_size, self.cnn_flat_size),
            nn.ReLU()
        ).to(self.device) for j in range(self.att_lstm_seq_len)]) for i in range(self.att_lstm_num)])
        # shape=[att_lstm_num, att_lstm_seq_len, (batch_size, cnn_flat_size)]

        # [torch.cat(list, dim=-1)], shape=[att_lstm_num, (batch_size, cnn_flat_size * att_lstm_seq_len)]
        # [torch.reshape(tensor, (tensor.shape[0], att_lstm_seq_len, cnn_flat_size))]
        # [torch.cat(list, dim=-1)], shape=[att_lstm_num, (batch_size, att_lstm_seq_len, feature_vec_len + cnn_flat_size)]

        self.att_lstms = nn.ModuleList(
            [nn.LSTM(input_size=self.feature_vec_len + self.cnn_flat_size, hidden_size=self.lstm_out_size,
                     batch_first=True, dropout=0.1).to(self.device) for i in range(self.att_lstm_num)])
        # [result] shape=[att_lstm_num, (batch_size, lstm_seq_len, lstm_out_size)]
        # [result, (hn, cn) = lstm -> result] [att_lstm_num, (batch_size, lstm_seq_len, lstm_out_size)]

        # compare
        self.att_low_level = nn.ModuleList([CBAAttention(self.device, self.lstm_out_size, self.lstm_out_size)
                                            for i in range(self.att_lstm_num)])
        # shape=[att_lstm_num, (batch_size, lstm_out_size)]
        # torch.cat(list, dim=-1), shape=(batch_size, lstm_out_size * att_lstm_num)
        # torch.reshape(tensor, (tensor.shape[0], att_lstm_num, lstm_out_size))
        # shape=(batch_size, att_lstm_num, lstm_out_size)

        self.att_high_level = nn.LSTM(input_size=self.lstm_out_size, hidden_size=self.lstm_out_size,
                                      batch_first=True, dropout=0.1).to(self.device)
        # result shape=(batch_size, att_lstm_num, lstm_out_size)
        # result, (hn, cn) = lstm -> hn[-1] shape=(batch_size, lstm_out_size)

        self.lstm_all = nn.Linear(self.lstm_out_size + self.lstm_out_size, self.output_dim).to(self.device)
        self.pred_volume = nn.Tanh().to(self.device)

    def forward(self, batch):
        flatten_att_nbhd_inputs = modify_input(batch['flatten_att_nbhd_inputs'])
        flatten_att_flow_inputs = modify_input(batch['flatten_att_flow_inputs'])
        att_lstm_inputs = modify_input(batch['att_lstm_inputs'])
        nbhd_inputs = modify_input(batch['nbhd_inputs'])
        flow_inputs = modify_input(batch['flow_inputs'])
        lstm_inputs = batch['lstm_inputs']
        lstm_inputs = lstm_inputs.reshape(-1, *lstm_inputs.shape[4:])

        att_nbhd_inputs = []
        att_flow_inputs = []
        for att in range(self.att_lstm_num):
            att_nbhd_inputs.append(flatten_att_nbhd_inputs[att * self.att_lstm_seq_len:
                                                           (att + 1) * self.att_lstm_seq_len])
            att_flow_inputs.append(flatten_att_flow_inputs[att * self.att_lstm_seq_len:
                                                           (att + 1) * self.att_lstm_seq_len])

        nbhd_inputs = [nbhd_inputs[ts].permute(0, 3, 1, 2) for ts in range(self.lstm_seq_len)]
        flow_inputs = [flow_inputs[ts].permute(0, 3, 1, 2) for ts in range(self.lstm_seq_len)]

        nbhd_convs = [self.nbhd_cnns_1st[ts](nbhd_inputs[ts]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.flow_cnns_1st[ts](flow_inputs[ts]) for ts in range(self.lstm_seq_len)]

        nbhd_convs = [nbhd_convs[ts] * flow_convs[ts] for ts in range(self.lstm_seq_len)]

        nbhd_convs = [self.nbhd_cnns_2nd[ts](nbhd_convs[ts]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.flow_cnns_2nd[ts](flow_inputs[ts]) for ts in range(self.lstm_seq_len)]

        nbhd_convs = [nbhd_convs[ts] * flow_convs[ts] for ts in range(self.lstm_seq_len)]

        nbhd_convs = [self.nbhd_cnns_3rd[ts](nbhd_convs[ts]) for ts in range(self.lstm_seq_len)]
        flow_convs = [self.flow_cnns_3rd[ts](flow_inputs[ts]) for ts in range(self.lstm_seq_len)]

        nbhd_convs = [nbhd_convs[ts] * flow_convs[ts] for ts in range(self.lstm_seq_len)]

        nbhd_vecs = [self.nbhd_vecs[ts](nbhd_convs[ts]) for ts in range(self.lstm_seq_len)]

        nbhd_vec = torch.stack(nbhd_vecs, dim=1)
        lstm_input = torch.cat([lstm_inputs, nbhd_vec], dim=-1)
        result, (hn, cn) = self.lstm(lstm_input)

        lstm = hn[-1]

        att_nbhd_inputs = [[att_nbhd_inputs[att][ts].permute(0, 3, 1, 2) for ts in range(self.att_lstm_seq_len)]
                           for att in range(self.att_lstm_num)]
        att_flow_inputs = [[att_flow_inputs[att][ts].permute(0, 3, 1, 2) for ts in range(self.att_lstm_seq_len)]
                           for att in range(self.att_lstm_num)]

        att_nbhd_convs = [[self.att_nbhd_cnns_1st[att][ts](att_nbhd_inputs[att][ts])
                           for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_cnns_1st[att][ts](att_flow_inputs[att][ts])
                           for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_gates = [[self.att_flow_gate_1st[att][ts](att_flow_convs[att][ts])
                           for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[att_nbhd_convs[att][ts] * att_flow_gates[att][ts]
                           for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]

        att_nbhd_convs = [[self.att_nbhd_cnns_2nd[att][ts](att_nbhd_convs[att][ts])
                           for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_cnns_2nd[att][ts](att_flow_convs[att][ts])
                           for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_gates = [[self.att_flow_gate_2nd[att][ts](att_flow_convs[att][ts])
                           for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[att_nbhd_convs[att][ts] * att_flow_gates[att][ts]
                           for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]

        att_nbhd_convs = [[self.att_nbhd_cnns_3rd[att][ts](att_nbhd_convs[att][ts])
                           for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_convs = [[self.att_flow_cnns_3rd[att][ts](att_flow_convs[att][ts])
                           for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_flow_gates = [[self.att_flow_gate_3rd[att][ts](att_flow_convs[att][ts])
                           for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        att_nbhd_convs = [[att_nbhd_convs[att][ts] * att_flow_gates[att][ts]
                           for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]

        att_nbhd_vecs = [[self.att_nbhd_vecs[att][ts](att_nbhd_convs[att][ts])
                          for ts in range(self.att_lstm_seq_len)] for att in range(self.att_lstm_num)]
        # shape=[att_lstm_num, att_lstm_seq_len, (batch_size, cnn_flat_size)]

        att_nbhd_vec = [torch.stack(att_nbhd_vecs[att], dim=1) for att in range(self.att_lstm_num)]
        att_lstm_input = [torch.cat([att_lstm_inputs[att], att_nbhd_vec[att]], dim=-1)
                          for att in range(self.att_lstm_num)]
        att_lstms = [self.att_lstms[att](att_lstm_input[att])[0] for att in range(self.att_lstm_num)]
        # shape=[att_lstm_num, (batch_size, lstm_out_size)]

        att_low_level = [self.att_low_level[att]([att_lstms[att], lstm]) for att in range(self.att_lstm_num)]
        att_low_level = torch.stack(att_low_level, dim=1)
        # shape=(batch_size, att_lstm_num, lstm_out_size)

        result, (hn, cn) = self.att_high_level(att_low_level)
        att_high_level = hn[-1]
        # shape = (batch_size, lstm_out_size)

        lstm_all = torch.cat([att_high_level, lstm], dim=-1)
        lstm_all = self.lstm_all(lstm_all)
        return self.pred_volume(lstm_all)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        res = loss.masked_mse_torch(y_predicted, y_true, 0)
        return res

    def predict(self, batch):
        return self.forward(batch).reshape(-1, self.output_window, self.len_row, self.len_column, self.output_dim)
