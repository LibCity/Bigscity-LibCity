import math
import torch
import torch.nn as nn

from libcity.model.abstract_model import AbstractModel


class STRNN(AbstractModel):
    def __init__(self, config, data_feature):
        super(STRNN, self).__init__(config, data_feature)
        self.hidden_size = config['hidden_size']
        self.device = config['device']
        self.loc_size = data_feature['loc_size']
        self.uid_size = data_feature['uid_size']
        self.lw_time = 0.0
        self.up_time = data_feature['tim_size'] - 1
        self.lw_loc = 0.0
        self.up_loc = data_feature['distance_upper']
        self.h0 = nn.Parameter(torch.randn(size=[self.hidden_size, 1]))  # h0
        self.weight_ih = nn.Parameter(torch.randn(
            size=[self.hidden_size, self.hidden_size]))  # C
        self.weight_th_upper = nn.Parameter(torch.randn(
            size=[self.hidden_size, self.hidden_size]))  # T Tu
        self.weight_th_lower = nn.Parameter(torch.randn(
            size=[self.hidden_size, self.hidden_size]))  # T Tl
        self.weight_sh_upper = nn.Parameter(torch.randn(
            size=[self.hidden_size, self.hidden_size]))  # S
        self.weight_sh_lower = nn.Parameter(torch.randn(
            size=[self.hidden_size, self.hidden_size]))  # S

        self.location_weight = nn.Embedding(
            self.loc_size, self.hidden_size)  # 还是按编号来的，但是需要经纬度额外信息
        self.permanet_weight = nn.Embedding(self.uid_size, self.hidden_size)

        self.sigmoid = nn.Sigmoid()

        self.reset_parameters()  # 这个应该是初始化参数的

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, td_upper, td_lower, ld_upper, ld_lower, current_loc,
                loc_len):
        # 所以他是直接把 target 放到 loc 里面去了嘛，所以只需要计算对 target 的 loss ??
        # td_upper 是 U(td) - td 的结果
        batch_size = current_loc.shape[0]
        output = []
        for i in range(batch_size):
            ttd = [((self.weight_th_upper * td_upper[i][j] +
                     self.weight_th_lower * td_lower[i][j])
                    / (td_upper[i][j] + td_lower[i][j]))
                   for j in range(loc_len[i])]
            sld = [((self.weight_sh_upper * ld_upper[i][j] +
                     self.weight_sh_lower * ld_lower[i][j])
                    / (ld_upper[i][j] + ld_lower[i][j]))
                   for j in range(loc_len[i])]
            loc = current_loc[i][:loc_len[i]]  # sequence_len
            loc = self.location_weight(loc).unsqueeze(2)
            loc_vec = torch.sum(torch.cat(
                [torch.mm(sld[j], torch.mm(ttd[j], loc[j])).unsqueeze(0)
                 for j in range(loc_len[i])], dim=0), dim=0)
            usr_vec = torch.mm(self.weight_ih, self.h0)
            # hidden_size x 1
            hx = (loc_vec + usr_vec).reshape(1,
                                             self.hidden_size)
            output.append(hx)
        output = torch.cat(output, dim=0)
        return self.sigmoid(output)

    def calculate_loss(self, batch):
        user = batch['uid']
        dst = batch['target'].tolist()
        dst_time = batch['target_tim']
        current_tim = batch['current_tim']
        # 计算 td ld
        batch_size = len(dst)
        td = dst_time.unsqueeze(1) - current_tim
        ld = batch['current_dis']
        loc_len = batch.get_origin_len('current_loc')

        td_upper = torch.LongTensor(
            [self.up_time] * batch_size).to(self.device).unsqueeze(1)
        td_upper = td_upper - td
        td_lower = td  # 因为 lower 是 0
        ld_upper = torch.LongTensor(
            [self.up_loc] * batch_size).to(self.device).unsqueeze(1)
        ld_upper = ld_upper - ld
        ld_lower = ld  # 因为下界是 0
        # batch_size * hidden_size
        h_tq = self.forward(td_upper, td_lower, ld_upper, ld_lower,
                            batch['current_loc'], loc_len)
        dst = batch['target']
        p_u = self.permanet_weight(user)  # batch_size * hidden_size
        q_v = self.location_weight(dst)  # batch_size * hidden_size
        user_vector = h_tq + p_u
        output = torch.zeros([batch_size, 1])
        for i in range(batch_size):
            output[i] = torch.dot(user_vector[i], q_v[i])
        output = torch.sum(output, dim=0)
        return torch.log(1 + torch.exp(torch.neg(output)))

    def predict(self, batch):
        user = batch['uid']
        dst = batch['target'].tolist()
        dst_time = batch['target_tim']
        current_tim = batch['current_tim']
        # 计算 td ld
        batch_size = len(dst)
        td = dst_time.unsqueeze(1) - current_tim
        ld = batch['current_dis']
        loc_len = batch.get_origin_len('current_loc')

        td_upper = torch.LongTensor(
            [self.up_time] * batch_size).to(self.device).unsqueeze(1)
        td_upper = td_upper - td
        td_lower = td  # 因为 lower 是 0
        ld_upper = torch.LongTensor(
            [self.up_loc] * batch_size).to(self.device).unsqueeze(1)
        ld_upper = ld_upper - ld
        ld_lower = ld  # 因为下界是 0
        # batch_size * hidden_size
        h_tq = self.forward(td_upper, td_lower, ld_upper, ld_lower,
                            batch['current_loc'], loc_len)
        p_u = self.permanet_weight(user)  # batch_size * hidden_size
        user_vector = h_tq + p_u  # batch_size * hidden_size
        # 这里有问题，因为 user_vector 是依据 target 来算的，实际上应该是每个 loc 一个对应的 user_vector
        # batch_size * loc_size
        ret = torch.mm(user_vector, self.location_weight.weight.T)
        return ret
