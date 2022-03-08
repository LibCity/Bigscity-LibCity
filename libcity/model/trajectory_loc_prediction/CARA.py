# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model.abstract_model import AbstractModel
from math import sin, cos, sqrt, atan2, radians
import numpy as np


def identity_loss(y_true, y_pred):
    return torch.mean(y_pred - 0 * y_true)


class CARA1(nn.Module):

    def hard_sigmoid(self, x):
        x = x / 6 + 0.5
        x = F.threshold(-x, -1, -1)
        x = F.threshold(-x, 0, 0)
        return x

    def __init__(self, output_dim, input_dim,
                 init='glorot_uniform', inner_init='orthogonal', device=None,
                 **kwargs):
        super(CARA1, self).__init__()
        self.output_dim = output_dim
        self.init = init
        self.inner_init = inner_init
        self.activation = self.hard_sigmoid
        self.inner_activation = nn.Tanh()
        self.device = device
        self.build(input_dim)

    def add_weight(self, shape, initializer):
        ts = torch.zeros(shape)
        if initializer == 'glorot_uniform':
            ts = nn.init.xavier_normal_(ts)
        elif initializer == 'orthogonal':
            ts = nn.init.orthogonal_(ts)

        return nn.Parameter(ts)

    def build(self, input_shape):
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape

        self.W_z = self.add_weight((self.input_dim, self.output_dim),
                                   initializer=self.init)
        self.U_z = self.add_weight((self.output_dim, self.output_dim),
                                   initializer=self.init)
        self.b_z = self.add_weight((self.output_dim,),
                                   initializer='zero')
        self.W_r = self.add_weight((self.input_dim, self.output_dim),
                                   initializer=self.init)
        self.U_r = self.add_weight((self.output_dim, self.output_dim),
                                   initializer=self.init)
        self.b_r = self.add_weight((self.output_dim,),
                                   initializer='zero')
        self.W_h = self.add_weight((self.input_dim, self.output_dim),
                                   initializer=self.init)
        self.U_h = self.add_weight((self.output_dim, self.output_dim),
                                   initializer=self.init)
        self.b_h = self.add_weight((self.output_dim,),
                                   initializer='zero')

        self.A_h = self.add_weight((self.output_dim, self.output_dim),
                                   initializer=self.init)
        self.A_u = self.add_weight((self.output_dim, self.output_dim),
                                   initializer=self.init)

        self.b_a_h = self.add_weight((self.output_dim,),
                                     initializer='zero')
        self.b_a_u = self.add_weight((self.output_dim,),
                                     initializer='zero')

        self.W_t = self.add_weight((self.input_dim, self.output_dim),
                                   initializer=self.init)
        self.U_t = self.add_weight((1, self.output_dim),
                                   initializer=self.init)
        self.b_t = self.add_weight((self.output_dim,),
                                   initializer='zero')

        self.W_g = self.add_weight((self.input_dim, self.output_dim),
                                   initializer=self.init)
        self.U_g = self.add_weight((1, self.output_dim),
                                   initializer=self.init)
        self.b_g = self.add_weight((self.output_dim,),
                                   initializer='zero')

    def preprocess_input(self, x):
        return x

    def forward(self, x):
        """
        X : batch * timeLen * dims(有拓展)
        """
        tlen = x.shape[1]
        output = torch.zeros((x.shape[0], self.output_dim)).to(self.device)
        for i in range(tlen):
            output = self.step(x[:, i, :], output)
        return output

    def step(self, x, states):
        """
        用于多批次同一时间
        states为上一次多批次统一时间数据
        """
        h_tm1 = states

        # phi_t
        u = x[:, self.output_dim: 2 * self.output_dim]
        # delta_t
        t = x[:, 2 * self.output_dim: (2 * self.output_dim) + 1]
        # delta_g
        g = x[:, (2 * self.output_dim) + 1:]
        # phi_v
        x = x[:, :self.output_dim]

        t = self.inner_activation(torch.matmul(t, self.U_t))
        g = self.inner_activation(torch.matmul(g, self.U_g))
        #       Time-based gate
        t1 = self.inner_activation(torch.matmul(x, self.W_t) + t + self.b_t)
        #       Geo-based gate
        g1 = self.inner_activation(torch.matmul(x, self.W_g) + g + self.b_g)

        #       Contextual Attention Gate
        a = self.inner_activation(
            torch.matmul(h_tm1, self.A_h) + torch.matmul(u, self.A_u) + self.b_a_h + self.b_a_u)

        x_z = torch.matmul(x, self.W_z) + self.b_z
        x_r = torch.matmul(x, self.W_r) + self.b_r
        x_h = torch.matmul(x, self.W_h) + self.b_h

        u_z_ = torch.matmul((1 - a) * u, self.W_z) + self.b_z
        u_r_ = torch.matmul((1 - a) * u, self.W_r) + self.b_r
        u_h_ = torch.matmul((1 - a) * u, self.W_h) + self.b_h

        u_z = torch.matmul(a * u, self.W_z) + self.b_z
        u_r = torch.matmul(a * u, self.W_r) + self.b_r
        u_h = torch.matmul(a * u, self.W_h) + self.b_h

        #       update gate
        z = self.inner_activation(x_z + torch.matmul(h_tm1, self.U_z) + u_z)
        #       reset gate
        r = self.inner_activation(x_r + torch.matmul(h_tm1, self.U_r) + u_r)
        #       hidden state
        hh = self.activation(x_h + torch.matmul(r * t1 * g1 * h_tm1, self.U_h) + u_h)

        h = z * h_tm1 + (1 - z) * hh
        h = (1 + u_z_ + u_r_ + u_h_) * h
        return h
        # return h


def bpr_triplet_loss(x):
    positive_item_latent, negative_item_latent = x

    reg = 0

    loss = 1 - torch.log(torch.sigmoid(
        torch.sum(positive_item_latent, dim=-1, keepdim=True) -
        torch.sum(negative_item_latent, dim=-1, keepdim=True))) - reg

    return loss


class Recommender(nn.Module):
    def __init__(self, num_users, num_items, num_times, latent_dim, maxvenue=5, device=None):
        super(Recommender, self).__init__()
        self.maxVenue = maxvenue
        self.latent_dim = latent_dim
        self.device = device
        # num * maxVenue * dim
        self.U_Embedding = nn.Embedding(num_users, latent_dim)
        self.V_Embedding = nn.Embedding(num_items, latent_dim)
        self.T_Embedding = nn.Embedding(num_times, latent_dim)
        torch.nn.init.uniform_(self.U_Embedding.weight)
        torch.nn.init.uniform_(self.V_Embedding.weight)
        torch.nn.init.uniform_(self.T_Embedding.weight)
        self.rnn = nn.Sequential(
            CARA1(latent_dim, latent_dim, device=self.device, input_shape=(self.maxVenue, (self.latent_dim * 2) + 2,),
                  unroll=True,
                  ))
        #       latent_dim * 2 + 2 = v_embedding + t_embedding + time_gap + distance

    def forward(self, x):
        # INPUT = [self.user_input, self.time_input, self.gap_time_input, self.pos_distance_input,
        #          self.neg_distance_input, self.checkins_input,
        #          self.neg_checkins_input]
        # pass
        #       User latent factor
        user_input = x[0]
        time_input = x[1]
        gap_time_input = x[2]
        pos_distance_input = x[3]
        neg_distance_input = x[4]
        checkins_input = x[5]
        neg_checkins_input = x[6]

        self.u_latent = self.U_Embedding(user_input)
        self.t_latent = self.T_Embedding(time_input)

        h, w = gap_time_input.shape
        gap_time_input = gap_time_input.view(h, w, 1)
        rnn_input = torch.cat([self.V_Embedding(checkins_input), self.T_Embedding(time_input), gap_time_input], -1)
        neg_rnn_input = torch.cat([self.V_Embedding(neg_checkins_input), self.T_Embedding(time_input), gap_time_input],
                                  -1)
        h, w = pos_distance_input.shape
        pos_distance_input = pos_distance_input.view(h, w, 1)
        h, w = neg_distance_input.shape
        neg_distance_input = neg_distance_input.view(h, w, 1)
        rnn_input = torch.cat([rnn_input, pos_distance_input], -1)
        neg_rnn_input = torch.cat([neg_rnn_input, neg_distance_input], -1)
        self.checkins_emb = self.rnn(rnn_input)
        self.neg_checkins_emb = self.rnn(neg_rnn_input)

        pred = (self.checkins_emb * self.u_latent).sum(dim=1)
        neg_pred = (self.neg_checkins_emb * self.u_latent).sum(dim=1)
        return bpr_triplet_loss([pred, neg_pred])

    def rank(self, uid, hist_venues, hist_times, hist_time_gap, hist_distances):
        #         hist_venues = hist_venues + [candidate_venue]
        #         hist_times = hist_times + [time]
        #         hist_time_gap = hist_time_gap + [time_gap]
        #         hist_distances = hist_distances + [distance]

        # u_latent = self.U_Embedding(torch.tensor(uid))
        # v_latent = self.V_Embedding(torch.tensor(hist_venues))
        # t_latent = self.T_Embedding(torch.tensor(hist_times))
        u_latent = self.U_Embedding.weight[uid]
        v_latent = self.V_Embedding.weight[hist_venues.reshape(-1)].view(hist_venues.shape[0], hist_venues.shape[1], -1)
        t_latent = self.T_Embedding.weight[hist_times.reshape(-1)].view(hist_times.shape[0], hist_times.shape[1], -1)

        h, w = hist_time_gap.shape
        hist_time_gap = hist_time_gap.reshape(h, w, 1)
        h, w = hist_distances.shape
        hist_distances = hist_distances.reshape(h, w, 1)
        rnn_input = torch.cat([t_latent, torch.FloatTensor(hist_time_gap).to(self.device)], dim=-1)
        rnn_input = torch.cat([rnn_input, torch.FloatTensor(hist_distances).to(self.device)], dim=-1)

        rnn_input = torch.cat([v_latent, rnn_input], dim=-1)

        dynamic_latent = self.rnn(rnn_input)
        scores = torch.mul(dynamic_latent, u_latent).sum(1)
        # scores = np.dot(dynamic_latent, u_latent)
        return scores


class CARA(AbstractModel):
    """rnn model with long-term history attention"""

    def __init__(self, config, data_feature):
        super(CARA, self).__init__(config, data_feature)
        self.loc_size = data_feature['loc_size']
        self.tim_size = data_feature['tim_size']
        self.uid_size = data_feature['uid_size']
        self.id2locid = data_feature['id2locid']
        self.id2loc = []
        self.device = config['device']
        for i in range(self.loc_size - 1):
            self.id2loc.append(self.id2locid[str(i)])
        self.id2loc.append(self.loc_size)
        self.id2loc = np.array(self.id2loc)
        self.coor = data_feature['poi_coor']
        self.rec = Recommender(self.uid_size, self.loc_size, self.tim_size, 10, device=self.device)

    def get_time_interval(self, x):
        y = x[:, :-1]
        y = np.concatenate([x[:, 0, None], y], axis=1)
        return x - y

    def get_time_interval2(self, x):
        y = x[:-1]
        y = np.concatenate([x[0, None], y], axis=0)
        return x - y

    def get_pos_distance(self, x):
        y = np.concatenate([x[:, 0, None, :], x[:, :-1, :]], axis=1)
        r = 6373.0
        rx = np.radians(x)
        ry = np.radians(y)

        d = x - y
        a = np.sin(d[:, :, 0] / 2) ** 2 + np.cos(rx[:, :, 0]) * np.cos(ry[:, :, 0]) * np.sin(d[:, :, 1] / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return r * c

    def get_pos_distance2(self, x):
        x = np.array(x.tolist())
        y = np.concatenate([x[0, None, :], x[:-1, :]], axis=0)
        r = 6373.0
        rx = np.radians(x)
        ry = np.radians(y)

        d = x - y
        a = np.sin(d[:, 0] / 2) ** 2 + np.cos(rx[:, 0]) * np.cos(ry[:, 0]) * np.sin(d[:, 1] / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return r * c

    def get_distance(self, lat1, lng1, lat2, lng2):
        r = 6373.0

        lat1 = radians(lat1)
        lon1 = radians(lng1)
        lat2 = radians(lat2)
        lon2 = radians(lng2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = int(r * c)
        return distance

    def get_neg_checkins(self, vis, x, y):
        len1, len2 = x.shape
        x_res = []
        x_res_distance = y[:].copy()
        for i in range(len1):
            visits = x[i]
            j = np.random.randint(self.loc_size - 1)
            while j in vis[i]:
                j = np.random.randint(self.loc_size - 1)
            tmp = visits[:].copy()
            tmp[-1] = j
            x_res.append(tmp)
            j1 = self.coor[str(self.id2loc[visits[-1]])]
            j = self.coor[str(self.id2loc[j])]
            x_res_distance[i, -1] = self.get_distance(j1[0], j1[1], j[0], j[1])
        return x_res, x_res_distance

    def forward(self, batch):
        hloc = np.array(batch['current_loc'].cpu())[:, :5]
        target = np.array(batch['target'].cpu())
        h = target.shape
        target = target.reshape((*h, 1))
        hloc = np.concatenate([hloc, target], axis=1)
        hloc1 = self.id2loc[hloc]
        tloc = np.array(batch['current_tim'].cpu())[:, :5]
        target_tim = np.array(batch['target_tim'].cpu())
        h = target_tim.shape
        target_tim = target_tim.reshape((*h, 1))
        tloc = np.concatenate([tloc, target_tim], axis=1)
        x_users = batch['uid']
        t_interval = self.get_time_interval(tloc)

        titude = []
        for i in hloc1.reshape(-1):
            titude.append(self.coor[str(i)])
        titude = np.array(titude).reshape((hloc.shape[0], hloc.shape[1], 2))
        pos_distance = self.get_pos_distance(titude)
        x_neg_checkins, x_neg_distance = self.get_neg_checkins(np.array(batch['current_loc'].cpu()), hloc, pos_distance)
        x_neg_checkins = np.array(x_neg_checkins)
        x = [x_users, torch.tensor(tloc).to(self.device),
             torch.FloatTensor(t_interval).to(self.device), torch.FloatTensor(pos_distance).to(self.device),
             torch.FloatTensor(x_neg_distance).to(self.device), torch.tensor(hloc).to(self.device),
             torch.tensor(x_neg_checkins).to(self.device)]
        return self.rec(x)

    def predict(self, batch):
        hloc = np.array(batch['current_loc'].cpu())[:, :5]
        tloc = np.array(batch['current_tim'].cpu())[:, :5]
        x_users = batch['uid']
        my_true = batch['target']
        my_true_tim = batch['target_tim']
        output = []
        for id, mloc in enumerate(hloc):
            hlocs = []
            tlocs = []
            users = []
            t_intervals = []
            distances = []
            target = my_true[id].item()
            target_tim = my_true_tim[id].item()
            mu = x_users[id]
            mh, mt = hloc[id], tloc[id].copy()
            mt = np.append(mt, target_tim)
            mi = self.get_time_interval2(mt)
            for i in range(101):
                mh = hloc[id].copy()
                if i == 0:
                    mh = np.append(mh, target)
                    tmh = self.id2loc[mh]
                    mtt = []
                    for i in tmh.reshape(-1):
                        mtt.append(self.coor[str(i)])
                    mtt = np.array(mtt).reshape((tmh.shape[0], 2))
                    md = self.get_pos_distance2(mtt)
                else:
                    j = target
                    while j == target:
                        j = np.random.randint(0, self.loc_size - 1)
                    mh = np.append(mh, j)
                    tmh = self.id2loc[mh]
                    mtt = []
                    for i in tmh.reshape(-1):
                        mtt.append(self.coor[str(i)])
                    mtt = np.array(mtt).reshape((tmh.shape[0], 2))
                    md = self.get_pos_distance2(mtt)

                hlocs.append(mh)
                tlocs.append(mt)
                users.append(mu.item())
                t_intervals.append(mi)
                distances.append(md)
            output.append(self.rec.rank(np.array(users), np.array(hlocs), np.array(tlocs), np.array(t_intervals),
                                        np.array(distances)).cpu().detach().numpy())
        output = np.array(output)
        return torch.tensor(output).to(self.device)

    def calculate_loss(self, batch):
        return torch.mean(self.forward(batch))
