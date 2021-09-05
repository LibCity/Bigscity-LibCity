from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class FC(nn.Module):  # is_training: self.training
    def __init__(self, input_dims, units, activations, bn, bn_decay, device, use_bias=True):
        super(FC, self).__init__()
        self.input_dims = input_dims
        self.units = units
        self.activations = activations
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.use_bias = use_bias
        self.layers = self._init_layers()

    def _init_layers(self):
        ret = nn.Sequential()
        units, activations = self.units, self.activations
        if isinstance(units, int):
            units, activations = [units], [activations]
        elif isinstance(self.units, tuple):
            units, activations = list(units), list(activations)
        assert type(units) == list
        index = 1
        input_dims = self.input_dims
        for num_unit, activation in zip(units, activations):
            if self.use_bias:
                basic_conv2d = nn.Conv2d(input_dims, num_unit, (1, 1), stride=1, padding=0, bias=True)
                nn.init.constant_(basic_conv2d.bias, 0)
            else:
                basic_conv2d = nn.Conv2d(input_dims, num_unit, (1, 1), stride=1, padding=0, bias=False)
            nn.init.xavier_normal_(basic_conv2d.weight)
            ret.add_module('conv2d' + str(index), basic_conv2d)
            if activation is not None:
                if self.bn:
                    decay = self.bn_decay if self.bn_decay is not None else 0.1
                    basic_batch_norm = nn.BatchNorm2d(num_unit, eps=1e-3, momentum=decay)
                    ret.add_module('batch_norm' + str(index), basic_batch_norm)
                ret.add_module('activation' + str(index), activation())
            input_dims = num_unit
            index += 1
        return ret

    def forward(self, x):
        # x: (N, H, W, C)
        x = x.transpose(1, 3).transpose(2, 3)  # x: (N, C, H, W)
        x = self.layers(x)
        x = x.transpose(2, 3).transpose(1, 3)  # x: (N, H, W, C)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, K, D, bn, bn_decay, device):
        super(SpatialAttention, self).__init__()
        self.K = K
        self.D = D
        self.d = self.D / self.K
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.input_query_fc = FC(input_dims=2 * self.D, units=self.D, activations=nn.ReLU,
                                 bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.input_key_fc = FC(input_dims=2 * self.D, units=self.D, activations=nn.ReLU,
                               bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.input_value_fc = FC(input_dims=2 * self.D, units=self.D, activations=nn.ReLU,
                                 bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.output_fc = FC(input_dims=self.D, units=[self.D, self.D], activations=[nn.ReLU, None],
                            bn=self.bn, bn_decay=self.bn_decay, device=self.device)

    def forward(self, x, ste):
        '''
        spatial attention mechanism
        x:      (batch_size, num_step, num_nodes, D)
        ste:    (batch_size, num_step, num_nodes, D)
        return: (batch_size, num_step, num_nodes, D)
        '''
        x = torch.cat((x, ste), dim=-1)
        # (batch_size, num_step, num_nodes, D)
        query = self.input_query_fc(x)
        key = self.input_key_fc(x)
        value = self.input_value_fc(x)
        # (K*batch_size, num_step, num_nodes, d)
        query = torch.cat(torch.split(query, query.size(-1) // self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, key.size(-1) // self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, value.size(-1) // self.K, dim=-1), dim=0)

        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= self.d ** 0.5
        attention = torch.softmax(attention, dim=-1)  # (K*batch_size, num_step, num_nodes, num_nodes)

        x = torch.matmul(attention, value)
        x = torch.cat(torch.split(x, x.size(0) // self.K, dim=0), dim=-1)
        x = self.output_fc(x)  # (batch_size, num_steps, num_nodes, D)
        return x


class TemporalAttention(nn.Module):
    def __init__(self, K, D, bn, bn_decay, device, mask=True):
        super(TemporalAttention, self).__init__()
        self.K = K
        self.D = D
        self.d = self.D / self.K
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.mask = mask
        self.input_query_fc = FC(input_dims=2 * self.D, units=self.D, activations=nn.ReLU,
                                 bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.input_key_fc = FC(input_dims=2 * self.D, units=self.D, activations=nn.ReLU,
                               bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.input_value_fc = FC(input_dims=2 * self.D, units=self.D, activations=nn.ReLU,
                                 bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.output_fc = FC(input_dims=self.D, units=[self.D, self.D], activations=[nn.ReLU, None],
                            bn=self.bn, bn_decay=self.bn_decay, device=self.device)

    def forward(self, x, ste):
        '''
        temporal attention mechanism
        x:      (batch_size, num_step, num_nodes, D)
        ste:    (batch_size, num_step, num_nodes, D)
        return: (batch_size, num_step, num_nodes, D)
        '''
        x = torch.cat((x, ste), dim=-1)
        # (batch_size, num_step, num_nodes, D)
        query = self.input_query_fc(x)
        key = self.input_key_fc(x)
        value = self.input_value_fc(x)
        # (K*batch_size, num_step, num_nodes, d)
        query = torch.cat(torch.split(query, query.size(-1) // self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, key.size(-1) // self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, value.size(-1) // self.K, dim=-1), dim=0)
        # query: (K*batch_size, num_nodes, num_step, d)
        # key:   (K*batch_size, num_nodes, d, num_step)
        # value: (K*batch_size, num_nodes, num_step, d)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2).transpose(2, 3)
        value = value.transpose(1, 2)

        attention = torch.matmul(query, key)
        attention /= self.d ** 0.5  # (K*batch_size, num_nodes, num_step, num_step)
        if self.mask:
            batch_size = x.size(0)
            num_step = x.size(1)
            num_nodes = x.size(2)
            mask = torch.ones((num_step, num_step), device=self.device)
            mask = torch.tril(mask)
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.repeat(self.K * batch_size, num_nodes, 1, 1)
            mask = mask.bool().int()
            mask_rev = -(mask - 1)
            attention = mask * attention + mask_rev * torch.full(attention.shape, -2 ** 15 + 1, device=self.device)
        attention = torch.softmax(attention, dim=-1)

        x = torch.matmul(attention, value)
        x = x.transpose(1, 2)
        x = torch.cat(torch.split(x, x.size(0) // self.K, dim=0), dim=-1)
        x = self.output_fc(x)  # (batch_size, output_length, num_nodes, D)
        return x


class GatedFusion(nn.Module):
    def __init__(self, D, bn, bn_decay, device):
        super(GatedFusion, self).__init__()
        self.D = D
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.HS_fc = FC(input_dims=self.D, units=self.D, activations=None,
                        bn=self.bn, bn_decay=self.bn_decay, device=self.device, use_bias=False)
        self.HT_fc = FC(input_dims=self.D, units=self.D, activations=None,
                        bn=self.bn, bn_decay=self.bn_decay, device=self.device, use_bias=True)
        self.output_fc = FC(input_dims=self.D, units=[self.D, self.D], activations=[nn.ReLU, None],
                            bn=self.bn, bn_decay=self.bn_decay, device=self.device)

    def forward(self, HS, HT):
        '''
        gated fusion
        HS:     (batch_size, num_step, num_nodes, D)
        HT:     (batch_size, num_step, num_nodes, D)
        return: (batch_size, num_step, num_nodes, D)
        '''
        XS = self.HS_fc(HS)
        XT = self.HT_fc(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.multiply(z, HS), torch.multiply(1 - z, HT))
        H = self.output_fc(H)
        return H


class STAttBlock(nn.Module):
    def __init__(self, K, D, bn, bn_decay, device, mask=True):
        super(STAttBlock, self).__init__()
        self.K = K
        self.D = D
        self.d = self.D / self.K
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.mask = mask
        self.sp_att = SpatialAttention(K=self.K, D=self.D, bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.temp_att = TemporalAttention(K=self.K, D=self.D, bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.gated_fusion = GatedFusion(D=self.D, bn=self.bn, bn_decay=self.bn_decay, device=self.device)

    def forward(self, x, ste):
        HS = self.sp_att(x, ste)
        HT = self.temp_att(x, ste)
        H = self.gated_fusion(HS, HT)
        return torch.add(x, H)


class TransformAttention(nn.Module):
    def __init__(self, K, D, bn, bn_decay, device):
        super(TransformAttention, self).__init__()
        self.K = K
        self.D = D
        self.d = self.D / self.K
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.input_query_fc = FC(input_dims=self.D, units=self.D, activations=nn.ReLU,
                                 bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.input_key_fc = FC(input_dims=self.D, units=self.D, activations=nn.ReLU,
                               bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.input_value_fc = FC(input_dims=self.D, units=self.D, activations=nn.ReLU,
                                 bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.output_fc = FC(input_dims=self.D, units=[self.D, self.D], activations=[nn.ReLU, None],
                            bn=self.bn, bn_decay=self.bn_decay, device=self.device)

    def forward(self, x, ste1, ste2):
        '''
        transform attention mechanism
        x:      (batch_size, input_length, num_nodes, D)
        ste_1:  (batch_size, input_length, num_nodes, D)
        ste_2:  (batch_size, output_length, num_nodes, D)
        return: (batch_size, output_length, num_nodes, D)
        '''
        # query: (batch_size, output_length, num_nodes, D)
        # key:   (batch_size, input_length, num_nodes, D)
        # value: (batch_size, input_length, num_nodes, D)
        query = self.input_query_fc(ste2)
        key = self.input_key_fc(ste1)
        value = self.input_value_fc(x)
        # query: (K*batch_size, output_length, num_nodes, d)
        # key:   (K*batch_size, input_length, num_nodes, d)
        # value: (K*batch_size, input_length, num_nodes, d)
        query = torch.cat(torch.split(query, query.size(-1) // self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, key.size(-1) // self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, value.size(-1) // self.K, dim=-1), dim=0)
        # query: (K*batch_size, num_nodes, output_length, d)
        # key:   (K*batch_size, num_nodes, d, input_length)
        # value: (K*batch_size, num_nodes, input_length, d)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2).transpose(2, 3)
        value = value.transpose(1, 2)

        attention = torch.matmul(query, key)
        attention /= self.d ** 0.5
        attention = torch.softmax(attention, dim=-1)  # (K*batch_size, num_nodes, output_length, input_length)

        x = torch.matmul(attention, value)
        x = x.transpose(1, 2)
        x = torch.cat(torch.split(x, x.size(0) // self.K, dim=0), dim=-1)
        x = self.output_fc(x)  # (batch_size, output_length, num_nodes, D)
        return x


class STEmbedding(nn.Module):
    def __init__(self, T, D, bn, bn_decay, add_day_in_week, device):
        super(STEmbedding, self).__init__()
        self.T = T
        self.D = D
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.SE_fc = FC(input_dims=self.D, units=[self.D, self.D], activations=[nn.ReLU, None],
                        bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.TE_fc = FC(input_dims=7 + self.T if add_day_in_week else self.T, units=[self.D, self.D],
                        activations=[nn.ReLU, None], bn=self.bn, bn_decay=self.bn_decay, device=self.device)

    def forward(self, SE, TE):
        '''
        spatio-temporal embedding
        SE:     (num_nodes, D)
        TE:     (batch_size, input_length+output_length, 7+T or T)
        retrun: (batch_size, input_length+output_length, num_nodes, D)
        '''
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.SE_fc(SE)
        TE = self.TE_fc(TE)
        return torch.add(SE, TE)


class GMAN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # get data feature
        self.adj_mx = self.data_feature.get('adj_mx')
        self.SE = self.data_feature.get('SE')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._scaler = self.data_feature.get('scaler')
        self.D = self.data_feature.get('D', 64)  # num_nodes
        self.T = self.data_feature.get('points_per_hour', 12) * 24  # points_per_data
        self.add_day_in_week = self.data_feature.get('add_day_in_week', False)
        # init logger
        self._logger = getLogger()
        # get model config
        self.input_window = config.get('input_window', 12)  # input_length
        self.output_window = config.get('output_window', 12)  # output_length
        self.L = config.get('L', 5)
        self.K = config.get('K', 8)
        self.device = config.get('device', torch.device('cpu'))
        self.bn = True
        self.bn_decay = 0.1
        # define the model structure
        self.input_fc = FC(input_dims=self.output_dim, units=[self.D, self.D], activations=[nn.ReLU, None],
                           bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.st_embedding = STEmbedding(T=self.T, D=self.D, bn=self.bn, bn_decay=self.bn_decay,
                                        add_day_in_week=self.add_day_in_week, device=self.device)
        self.encoder = nn.ModuleList()
        for _ in range(self.L):
            self.encoder.append(STAttBlock(K=self.K, D=self.D, bn=self.bn, bn_decay=self.bn_decay, device=self.device))
        # self.encoder = STAttBlock(K=self.K, D=self.D, bn=self.bn, bn_decay=self.bn_decay,
        #                           device=self.device)
        self.trans_att = TransformAttention(K=self.K, D=self.D, bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.decoder = nn.ModuleList()
        for _ in range(self.L):
            self.decoder.append(STAttBlock(K=self.K, D=self.D, bn=self.bn, bn_decay=self.bn_decay, device=self.device))
        # self.decoder = STAttBlock(K=self.K, D=self.D, bn=self.bn, bn_decay=self.bn_decay,
        #                           device=self.device)
        self.output_fc_1 = FC(input_dims=self.D, units=[self.D], activations=[nn.ReLU],
                              bn=self.bn, bn_decay=self.bn_decay, device=self.device, use_bias=True)
        self.output_fc_2 = FC(input_dims=self.D, units=[self.output_dim], activations=[None],
                              bn=self.bn, bn_decay=self.bn_decay, device=self.device, use_bias=True)

    def forward(self, batch):
        # ret: (batch_size, output_length, num_nodes, output_dim)
        # handle data
        x_all = batch['X']  # (batch_size, input_length, num_nodes, feature_dim)
        y_all = batch['y']  # (batch_size, out_length, num_nodes, feature_dim)
        index = -8 if self.add_day_in_week else -1
        x = x_all[:, :, :, 0:index]  # (batch_size, input_length, num_nodes, output_dim)
        SE = torch.from_numpy(self.SE).to(device=self.device)
        TE = torch.cat((x_all[:, :, :, index:], y_all[:, :, :, index:]), dim=1)
        _timeofday = TE[:, :, :, 0:1]
        _timeofday = torch.round(_timeofday * self.T)
        _timeofday = _timeofday.to(int)  # (batch_size, input_length+output_length, num_nodes, 1)
        _timeofday = _timeofday[:, :, 0, :]  # (batch_size, input_length+output_length, 1)
        timeofday = torch.zeros((_timeofday.size(0), _timeofday.size(1), self.T), device=self.device).long()
        timeofday.scatter_(dim=2, index=_timeofday.long(), src=torch.ones(timeofday.shape, device=self.device).long())
        if self.add_day_in_week:
            _dayofweek = TE[:, :, :, 1:]
            _dayofweek = _dayofweek.to(int)  # (batch_size, input_length+output_length, num_nodes, 7)
            dayofweek = _dayofweek[:, :, 0, :]  # (batch_size, input_length+output_length, 7)
            TE = torch.cat((dayofweek, timeofday), dim=2).type(torch.FloatTensor)
        else:
            TE = timeofday.type(torch.FloatTensor)
        TE = TE.unsqueeze(2).to(device=self.device)  # (batch_size, input_length+output_length, 1, 7+T or T)

        # create network
        # input
        x = self.input_fc(x)  # (batch_size, input_length, num_nodes, D)
        # STE
        ste = self.st_embedding(SE, TE)
        ste_p = ste[:, :self.input_window]  # (batch_size, input_length, num_nodes, D)
        ste_q = ste[:, self.input_window:]  # (batch_size, output_length, num_nodes, D)
        # encoder
        for encoder_layer in self.encoder:
            x = encoder_layer(x, ste_p)
        # transAtt
        x = self.trans_att(x, ste_p, ste_q)  # (batch_size, output_length, num_nodes, D)
        # decoder
        for decoder_layer in self.decoder:
            x = decoder_layer(x, ste_q)
        # output
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.output_fc_1(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.output_fc_2(x)  # (batch_size, output_length, num_nodes, output_dim)
        return x

    def calculate_loss(self, batch):
        y_true = batch['y']  # (batch_size, output_length, num_nodes, feature_dim)
        y_predicted = self.predict(batch)  # (batch_size, output_length, num_nodes, output_dim)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true, 0.0)

    def predict(self, batch):
        return self.forward(batch)  # (batch_size, output_length, num_nodes, output_dim)
