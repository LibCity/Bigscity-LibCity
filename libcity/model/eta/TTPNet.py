import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def normalize(data, mean, std):
    return (data - mean) / std


def unnormalize(data, mean, std):
    return data * std + mean


class Attr(nn.Module):
    def __init__(self, embed_dims, data_feature):
        super(Attr, self).__init__()

        self.embed_dims = embed_dims
        self.data_feature = data_feature

        for name, dim_in, dim_out in self.embed_dims:
            self.add_module(name + '_em', nn.Embedding(dim_in, dim_out))

    def out_size(self):
        sz = 0
        for _, _, dim_out in self.embed_dims:
            sz += dim_out
        # append total distance and holiday
        return sz + 2

    def forward(self, batch):
        em_list = []
        for name, _, _ in self.embed_dims:
            embed = getattr(self, name + '_em')
            attr_t = batch[name]

            attr_t = torch.squeeze(embed(attr_t), dim=1)

            em_list.append(attr_t)

        dist_mean, dist_std = self.data_feature["dist_mean"], self.data_feature["dist_std"]
        dist = normalize(batch["dist"], dist_mean, dist_std)
        em_list.append(dist)
        em_list.append(batch['holiday'].float())

        return torch.cat(em_list, dim=1)


class ShortSpeed(nn.Module):
    def __init__(self, data_feature):
        super(ShortSpeed, self).__init__()
        self.short_kernel_size = 2
        self.short_cnn = nn.Conv1d(3, 4, kernel_size=self.short_kernel_size, stride=1)
        self.short_rnn = nn.RNN(input_size=4, hidden_size=16, num_layers=1, batch_first=True)
        self.data_feature = data_feature

    def forward(self, batch):
        # short-term travel speed features
        n_batchs = batch['speeds'].size()[0]
        speeds_forward = batch['speeds'].reshape(-1, 4)  # (B * L, 4)
        speeds_adjacent1 = batch['speeds_relevant1'].reshape(-1, 4)
        speeds_adjacent2 = batch['speeds_relevant2'].reshape(-1, 4)
        grid_len = batch['grid_len'].reshape(-1, 1)  # (B * L, 1)

        speeds_forward = torch.unsqueeze(speeds_forward, dim=2)  # (B * L, 4, 1)
        speeds_adjacent1 = torch.unsqueeze(speeds_adjacent1, dim=2)
        speeds_adjacent2 = torch.unsqueeze(speeds_adjacent2, dim=2)

        grid_len = torch.unsqueeze(grid_len, dim=2)  # (B * L, 1, 1)
        grid_len_short = grid_len.expand(speeds_forward.size()[:2] + (grid_len.size()[-1], ))  # (B * L, 4, 1)

        times_forward = speeds_forward.clone()  # (B * L, 4, 1)
        times_forward[times_forward==0] = 0.2
        times_forward = grid_len_short / times_forward * 3600
        times_adjacent1 = speeds_adjacent1.clone()
        times_adjacent1[times_adjacent1==0] = 0.2
        times_adjacent1 = grid_len_short / times_adjacent1 * 3600
        times_adjacent2 = speeds_adjacent2.clone()
        times_adjacent2[times_adjacent2==0] = 0.2
        times_adjacent2 = grid_len_short / times_adjacent2 * 3600

        speeds_forward = normalize(speeds_forward, self.data_feature["speeds_mean"], self.data_feature["speeds_std"])
        speeds_adjacent1 = normalize(speeds_adjacent1, self.data_feature["speeds_relevant1_mean"], self.data_feature["speeds_relevant1_std"])
        speeds_adjacent2 = normalize(speeds_adjacent2, self.data_feature["speeds_relevant2_mean"], self.data_feature["speeds_relevant2_std"])
        grid_len_short = normalize(grid_len_short, self.data_feature["grid_len_mean"], self.data_feature["grid_len_std"])
        times_forward = normalize(times_forward, self.data_feature["time_gap_mean"], self.data_feature["time_gap_std"])
        times_adjacent1 = normalize(times_adjacent1, self.data_feature["time_gap_mean"], self.data_feature["time_gap_std"])
        times_adjacent2 = normalize(times_adjacent2, self.data_feature["time_gap_mean"], self.data_feature["time_gap_std"])

        inputs_0 = torch.cat([speeds_forward, grid_len_short, times_forward], dim=2)  # (B * L, 4, 3)
        inputs_1 = torch.cat([speeds_adjacent1, grid_len_short, times_adjacent1], dim=2)  # (B * L, 4, 3)
        inputs_2 = torch.cat([speeds_adjacent2, grid_len_short, times_adjacent2], dim=2)  # (B * L, 4, 3)

        outputs_0 = torch.tanh(self.short_cnn(inputs_0.permute(0, 2, 1)))
        outputs_0 = outputs_0.permute(0, 2, 1)
        outputs_1 = torch.tanh(self.short_cnn(inputs_1.permute(0, 2, 1)))
        outputs_1 = outputs_1.permute(0, 2, 1)
        outputs_2 = torch.tanh(self.short_cnn(inputs_2.permute(0, 2, 1)))
        outputs_2 = outputs_2.permute(0, 2, 1)

        outputs_0, _ = self.short_rnn(outputs_0)
        outputs_1, _ = self.short_rnn(outputs_1)
        outputs_2, _ = self.short_rnn(outputs_2)

        outputs_0 = outputs_0.reshape(n_batchs, -1, 4-self.short_kernel_size+1, 16)
        outputs_1 = outputs_1.reshape(n_batchs, -1, 4-self.short_kernel_size+1, 16)
        outputs_2 = outputs_2.reshape(n_batchs, -1, 4-self.short_kernel_size+1, 16)

        V_short = torch.cat([outputs_0[:, :, -1], outputs_1[:, :, -1], outputs_2[:, :, -1]], dim=2)

        return V_short


class LongSpeed(nn.Module):
    def __init__(self, data_feature):
        super(LongSpeed, self).__init__()
        self.long_kernel_size = 3
        self.long_cnn = nn.Conv1d(3, 4, kernel_size=self.long_kernel_size, stride=1)
        self.long_rnn = nn.RNN(input_size=4, hidden_size=16, num_layers=1, batch_first=True)
        self.data_feature = data_feature

    def forward(self, batch):
        n_batchs = batch['speeds_long'].size()[0]
        speeds_history = batch['speeds_long'].reshape(-1, 7)
        grid_len = batch['grid_len'].reshape(-1, 1)

        speeds_history = torch.unsqueeze(speeds_history, dim=2)

        grid_len = torch.unsqueeze(grid_len, dim=2)
        grid_len_long = grid_len.expand(speeds_history.size()[:2] + (grid_len.size()[-1], ))
        
        times_history = speeds_history.clone()
        times_history[times_history==0] = 0.2
        times_history = grid_len_long / times_history * 3600
        
        speeds_history = normalize(speeds_history, self.data_feature["speeds_long_mean"], self.data_feature["speeds_long_std"])
        grid_len_long = normalize(grid_len_long, self.data_feature["grid_len_mean"], self.data_feature["grid_len_std"])
        times_history = normalize(times_history, self.data_feature["time_gap_mean"], self.data_feature["time_gap_std"])
        
        inputs_3 = torch.cat([speeds_history, grid_len_long, times_history], dim=2)
        outputs_3 = self.long_cnn(inputs_3.permute(0, 2, 1))
        outputs_3 = outputs_3.permute(0, 2, 1)
        outputs_3, _ = self.long_rnn(outputs_3)
        outputs_3 = outputs_3.reshape(n_batchs, -1, 7 - self.long_kernel_size + 1, 16)
        
        V_long = outputs_3[:, :, -1]
        
        return V_long


class SpeedLSTM(nn.Module):
    def __init__(self, data_feature):
        super(SpeedLSTM, self).__init__()
        self.shortspeed_net = ShortSpeed(data_feature)
        self.longspeed_net = LongSpeed(data_feature)
        self.process_speeds = nn.Linear(64, 32)
        self.speed_lstm = nn.LSTM(
            input_size = 32,
            hidden_size = 32,
            num_layers = 1,
            batch_first = True,
            bidirectional = False,
            dropout = 0,
        )

    def forward(self, batch):
        shortspeeds_t = self.shortspeed_net(batch)
        longspeeds_t = self.longspeed_net(batch)
        whole_t = torch.cat([shortspeeds_t, longspeeds_t], dim=2)
        whole_t = self.process_speeds(whole_t)
        whole_t = torch.tanh(whole_t)

        lens = [batch["current_dis"].shape[1]] * batch["current_dis"].shape[0]
        lens = list(map(lambda x: x, lens))

        packed_inputs = nn.utils.rnn.pack_padded_sequence(whole_t, lens, batch_first=True)
        packed_hiddens, (_, _) = self.speed_lstm(packed_inputs)
        speeds_hiddens, _ = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first=True)

        return speeds_hiddens


class Road(nn.Module):
    def __init__(self, data_feature):
        super(Road, self).__init__()
        self.data_feature = data_feature
        self.embedding = nn.Embedding(128 * 128, 32)
        emb_vectors = data_feature["geo_embedding"]
        self.embedding.weight.data.copy_(torch.tensor(emb_vectors))
        self.process_coords = nn.Linear(2 + 32, 32)

    def forward(self, batch):
        longi_mean, longi_std = self.data_feature["longi_mean"], self.data_feature["longi_std"]
        current_longi = normalize(batch["current_longi"], longi_mean, longi_std)
        lngs = torch.unsqueeze(current_longi, dim=2)
        lati_mean, lati_std = self.data_feature["lati_mean"], self.data_feature["lati_std"]
        current_lati = normalize(batch["current_lati"], lati_mean, lati_std)
        lats = torch.unsqueeze(current_lati, dim=2)

        grid_ids = torch.unsqueeze(batch['current_loc'].long(), dim=2)
        grids = torch.squeeze(self.embedding(grid_ids), dim=2)
        locs = torch.cat([lngs, lats, grids], dim=2)
        locs = self.process_coords(locs)
        locs = torch.tanh(locs)
        
        return locs


class RoadLSTM(nn.Module):
    def __init__(self, data_feature):
        super(RoadLSTM, self).__init__()
        self.Road_net = Road(data_feature)
        self.Road_lstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0,
        )

    def forward(self, batch):
        Roads_t = self.Road_net(batch)
        whole_t = Roads_t

        lens = [batch["current_dis"].shape[1]] * batch["current_dis"].shape[0]
        lens = list(map(lambda x: x, lens))

        packed_inputs = nn.utils.rnn.pack_padded_sequence(whole_t, lens, batch_first=True)
        packed_hiddens, (_, _) = self.Road_lstm(packed_inputs)
        Roads_hiddens, _ = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first=True)

        return Roads_hiddens


class PredictionBiLSTM(nn.Module):
    def __init__(self, embed_dims, data_feature):
        super(PredictionBiLSTM, self).__init__()
        self.attr_net = Attr(embed_dims, data_feature)
        self.speed_lstm = SpeedLSTM(data_feature)
        self.road_lstm = RoadLSTM(data_feature)
        self.bi_lstm = nn.LSTM(
            input_size=self.attr_net.out_size() + 64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25,
        )
        self.lnhiddens = nn.LayerNorm(self.attr_net.out_size() + 64, elementwise_affine=True)

    def forward(self, batch):
        speeds_t = self.speed_lstm(batch)
        roads_t = self.road_lstm(batch)

        attr_t = self.attr_net(batch)
        attr_t = torch.unsqueeze(attr_t, dim=1)
        expand_attr_t = attr_t.expand(roads_t.size()[:2] + (attr_t.size()[-1], ))

        hiddens = torch.cat([expand_attr_t, speeds_t, roads_t], dim=2)
        hiddens = self.lnhiddens(hiddens)
        lens = [batch["current_longi"].shape[1]] * batch["current_longi"].shape[0]
        lens = list(map(lambda x: x, lens))

        packed_inputs = nn.utils.rnn.pack_padded_sequence(hiddens, lens, batch_first=True)
        packed_hiddens, (_, _) = self.bi_lstm(packed_inputs)
        hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first=True)

        return hiddens


class TTPNet(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super(TTPNet, self).__init__(config, data_feature)
        self.config = config
        self.data_feature = data_feature
        self.device = config.get('device', torch.device('cpu'))

        uid_emb_size = config.get("uid_emb_size", 16)
        weekid_emb_size = config.get("weekid_emb_size", 3)
        timdid_emb_size = config.get("timdid_emb_size", 8)
        uid_size = data_feature.get("uid_size", 13000)
        embed_dims = [
            ('uid', uid_size, uid_emb_size),
            ('weekid', 7, weekid_emb_size),
            ('timeid', 96, timdid_emb_size),
        ]

        self.bi_lstm = PredictionBiLSTM(embed_dims, data_feature)
        self.input2hid = nn.Linear(128, 128)
        self.hid2hid = nn.Linear(128, 64)
        self.hid2out = nn.Linear(64, 1)

    def forward(self, batch):
        hiddens = self.bi_lstm(batch)
        n = hiddens.size()[1]
        h_f = []

        for i in range(2, n):
            h_f_temp = torch.sum(hiddens[:, :i], dim=1)
            h_f.append(h_f_temp)
  
        h_f.append(torch.sum(hiddens, dim=1))
        h_f = torch.stack(h_f).permute(1, 0, 2)

        T_f_hat = self.input2hid(h_f)
        T_f_hat = F.relu(T_f_hat)
        T_f_hat = self.hid2hid(T_f_hat)
        T_f_hat = F.relu(T_f_hat)
        T_f_hat = self.hid2out(T_f_hat)

        return T_f_hat.squeeze(dim=2)

    def calculate_loss(self, batch):
        if self.training:
            T_f_hat = self.predict(batch).unsqueeze(dim=2)

            T_f = torch.unsqueeze(batch["current_tim"][:, 1:], dim=2)
            M_f = torch.unsqueeze(batch["masked_current_tim"][:, 1:], dim=1)
            loss_f = torch.bmm(M_f, torch.pow((T_f_hat - T_f) / T_f, 2)) / torch.bmm(M_f, M_f.permute(0, 2, 1))
            loss_f = torch.pow(loss_f, 1/2)
            return loss_f.mean()
        else:
            pred = self.predict(batch)
            label = batch["time"]
            return loss.masked_mape_torch(pred, label)

    def predict(self, batch):
        T_f_hat = self.forward(batch)
        time_gap_mean, time_gap_std = self.data_feature["time_gap_mean"], self.data_feature["time_gap_std"]
        T_f_hat = unnormalize(T_f_hat, time_gap_mean, time_gap_std)
        if self.training:
            return T_f_hat
        else:
            return T_f_hat[:, -1:]
