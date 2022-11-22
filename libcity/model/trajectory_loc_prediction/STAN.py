from libcity.model.abstract_model import AbstractModel
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import random


class Attn(nn.Module):
    def __init__(self, max_len, emb_loc, loc_max, device, dropout=0.1):
        super(Attn, self).__init__()
        self.max_len = max_len
        self.value = nn.Linear(self.max_len, 1, bias=False)
        self.emb_loc = emb_loc
        self.loc_max = loc_max
        self.device = device

    def forward(self, self_attn, self_delta, traj_len):
        # self_attn (N, M, emb), candidate (N, L, emb), \
        # self_delta (N, M, L, emb), len [N]
        self_delta = torch.sum(self_delta, -1).transpose(-1, -2)
        # squeeze the embed dimension
        [N, L, M] = self_delta.shape
        candidates = torch.linspace(
            1, int(self.loc_max), int(self.loc_max)
        ).long()  # (L)
        candidates = candidates.unsqueeze(0).expand(N, -1).to(self.device)  # (N, L)
        emb_candidates = self.emb_loc(candidates)  # (N, L, emb)
        attn = torch.mul(
            torch.bmm(emb_candidates, self_attn.transpose(-1, -2)),
            self_delta)  # (N, L, M)
        # pdb.set_trace()
        attn_out = self.value(attn).view(N, L)  # (N, L)
        # attn_out = F.log_softmax(attn_out, dim=-1)
        # ignore if cross_entropy_loss

        return attn_out  # (N, L)


class SelfAttn(nn.Module):
    def __init__(self, emb_size, output_size, dropout=0.1):
        super(SelfAttn, self).__init__()
        self.query = nn.Linear(emb_size, output_size, bias=False)
        self.key = nn.Linear(emb_size, output_size, bias=False)
        self.value = nn.Linear(emb_size, output_size, bias=False)

    def forward(self, joint, delta, traj_len):
        delta = torch.sum(delta, -1)  # squeeze the embed dimension
        # joint (N, M, emb), delta (N, M, M, emb), len [N]
        # construct attention mask
        mask = torch.zeros_like(delta, dtype=torch.float32)
        for i in range(mask.shape[0]):
            mask[i, 0:traj_len[i], 0:traj_len[i]] = 1

        attn = torch.add(
            torch.bmm(self.query(joint), self.key(joint).transpose(-1, -2)),
            delta)  # (N, M, M)
        attn = F.softmax(attn, dim=-1) * mask  # (N, M, M)

        attn_out = torch.bmm(attn, self.value(joint))  # (N, M, emb)

        return attn_out  # (N, M, emb)


class Embed(nn.Module):
    def __init__(self, ex, emb_size, loc_max, embed_layers):
        super(Embed, self).__init__()
        _, _, _, self.emb_su, self.emb_sl, self.emb_tu, self.emb_tl = \
            embed_layers
        self.su, self.sl, self.tu, self.tl = ex
        self.emb_size = emb_size
        self.loc_max = loc_max

    def forward(self, traj_loc, mat2, vec, traj_len):
        # traj_loc (N, M), mat2 (L, L), vec (N, M), delta_t (N, M, L)
        delta_t = vec.unsqueeze(-1).expand(-1, -1, self.loc_max)
        delta_s = torch.zeros_like(delta_t, dtype=torch.float32)
        mask = torch.zeros_like(delta_t, dtype=torch.long)
        for i in range(mask.shape[0]):  # N
            mask[i, 0:traj_len[i]] = 1
            delta_s[i, :traj_len[i]] = \
                torch.index_select(mat2, 0, (traj_loc[i]-1)[:traj_len[i]])

        # pdb.set_trace()

        esl, esu, etl, etu = \
            self.emb_sl(mask), self.emb_su(mask), \
            self.emb_tl(mask), self.emb_tu(mask)
        vsl, vsu, vtl, vtu = \
            (delta_s - self.sl).unsqueeze(-1).expand(
                -1, -1, -1, self.emb_size), \
            (self.su - delta_s).unsqueeze(-1).expand(
                -1, -1, -1, self.emb_size), \
            (delta_t - self.tl).unsqueeze(-1).expand(
                -1, -1, -1, self.emb_size), \
            (self.tu - delta_t).unsqueeze(-1).expand(
                -1, -1, -1, self.emb_size)

        space_interval = (esl * vsu + esu * vsl) / (self.su - self.sl)
        time_interval = (etl * vtu + etu * vtl) / (self.tu - self.tl)
        delta = space_interval + time_interval  # (N, M, L, emb)

        return delta


class MultiEmbed(nn.Module):
    def __init__(self, ex, emb_size, embed_layers):
        super(MultiEmbed, self).__init__()
        self.emb_t, self.emb_l, self.emb_u, self.emb_su, \
            self.emb_sl, self.emb_tu, self.emb_tl = embed_layers
        self.su, self.sl, self.tu, self.tl = ex
        self.emb_size = emb_size

    def forward(self, traj, mat, traj_len):
        # traj (N, M, 3), mat (N, M, M, 2), len [N]
        time = self.emb_t(traj[:, :, 2])  # (N, M) --> (N, M, embed)
        loc = self.emb_l(traj[:, :, 1])  # (N, M) --> (N, M, embed)
        user = self.emb_u(traj[:, :, 0])  # (N, M) --> (N, M, embed)
        joint = time + loc + user  # (N, M, embed)

        delta_s, delta_t = mat[:, :, :, 0], mat[:, :, :, 1]  # (N, M, M)
        mask = torch.zeros_like(delta_s, dtype=torch.long)
        for i in range(mask.shape[0]):
            mask[i, 0:traj_len[i], 0:traj_len[i]] = 1

        esl, esu, etl, etu = \
            self.emb_sl(mask), \
            self.emb_su(mask), \
            self.emb_tl(mask), \
            self.emb_tu(mask)
        vsl, vsu, vtl, vtu = \
            (delta_s - self.sl).unsqueeze(-1).expand(
                -1, -1, -1, self.emb_size), \
            (self.su - delta_s).unsqueeze(-1).expand(
                -1, -1, -1, self.emb_size), \
            (delta_t - self.tl).unsqueeze(-1).expand(
                -1, -1, -1, self.emb_size), \
            (self.tu - delta_t).unsqueeze(-1).expand(
                -1, -1, -1, self.emb_size)

        space_interval = (esl*vsu+esu*vsl) / (self.su-self.sl)
        time_interval = (etl*vtu+etu*vtl) / (self.tu-self.tl)
        delta = space_interval + time_interval  # (N, M, M, emb)

        return joint, delta


class STAN(AbstractModel):
    def __init__(self, config, data_feature):
        super(STAN, self).__init__(config, data_feature)

        t_dim = data_feature['tim_size']
        l_dim = data_feature['loc_size']
        u_dim = data_feature['uid_size']
        self.max_len = config['max_session_len']
        ex = data_feature['ex']
        embed_dim = config['embed_dim']
        self.batch_size = config['batch_size']
        self.device = config['device']
        self.num_neg = config['num_neg']
        self.seed = 0
        self.mat2s = torch.tensor(data_feature['spatial_matrix']).to(self.device)

        emb_t = nn.Embedding(t_dim, embed_dim, padding_idx=0)
        emb_l = nn.Embedding(l_dim, embed_dim, padding_idx=0)
        emb_u = nn.Embedding(u_dim, embed_dim, padding_idx=0)
        emb_su = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_sl = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tu = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tl = nn.Embedding(2, embed_dim, padding_idx=0)
        embed_layers = emb_t, emb_l, emb_u, emb_su, emb_sl, emb_tu, emb_tl

        self.MultiEmbed = MultiEmbed(ex, embed_dim, embed_layers)
        self.SelfAttn = SelfAttn(embed_dim, embed_dim)
        self.Embed = Embed(ex, embed_dim, l_dim-1, embed_layers)
        self.Attn = Attn(self.max_len, emb_l, l_dim-1, self.device)

    def forward(self, traj, mat1, vec, traj_len):
        # long(N, M, [u, l, t]), float(N, M, M, 2), \
        # float(L, L), float(N, M), long(N)

        # (N, M, emb), (N, M, M, emb)
        joint, delta = \
            self.MultiEmbed(traj, mat1, traj_len)
        self_attn = self.SelfAttn(joint, delta, traj_len)  # (N, M, emb)
        self_delta = self.Embed(
            traj[:, :, 1], self.mat2s, vec, traj_len
        )  # (N, M, L, emb)
        output = self.Attn(self_attn, self_delta, traj_len)  # (N, L)
        return output

    def predict(self, batch):
        """
        参数说明:
            batch (libcity.data.batch): 类 dict 文件，其中包含的键值参见任务说明文件。
        返回值:
            score (pytorch.tensor): 对应张量 shape 应为 batch_size *
                loc_size。这里返回的是模型对于输入当前轨迹的下一跳位置的预测值。
        """
        # get spatial_relation_mat from mat2s
        traj_temporal_mat = batch['traj_temporal_mat']
        traj_loc = batch['traj'][:, :, 1]
        traj_len = batch['traj_len']
        traj_spatial_mat = torch.zeros_like(traj_temporal_mat, dtype=torch.float32)
        for i in range(traj_spatial_mat.shape[0]):
            traj_loc_dis = torch.index_select(self.mat2s, 0, (traj_loc[i]-1)[:traj_len[i]])
            traj_spatial_mat[i, :traj_len[i], :traj_len[i]] = \
                torch.index_select(traj_loc_dis, 1, (traj_loc[i]-1)[:traj_len[i]])
        mat1 = torch.cat((traj_spatial_mat.unsqueeze(3), traj_temporal_mat.unsqueeze(3)), 3)
        return self.forward(batch['traj'], mat1, batch['candiate_temporal_vec'], batch['traj_len'])

    def calculate_loss(self, batch):
        """
        参数说明:
            batch (libcity.data.batch): 类 dict 文件，其中包含的键值参见任务说明文件。
        返回值:
            loss (pytorch.tensor): 可以调用 pytorch 实现的 loss 函数与 batch['target']
                目标值进行 loss 计算，并将计算结果返回。如模型有自己独特的 loss 计算方式则自行参考实现。
        """
        traj_temporal_mat = batch['traj_temporal_mat']
        traj_loc = batch['traj'][:, :, 1]
        traj_len = batch['traj_len']
        traj_spatial_mat = torch.zeros_like(traj_temporal_mat, dtype=torch.float32)
        for i in range(traj_spatial_mat.shape[0]):
            traj_loc_dis = torch.index_select(self.mat2s, 0, (traj_loc[i]-1)[:traj_len[i]])
            traj_spatial_mat[i, :traj_len[i], :traj_len[i]] = \
                torch.index_select(traj_loc_dis, 1, (traj_loc[i]-1)[:traj_len[i]])
        mat1 = torch.cat((traj_spatial_mat.unsqueeze(3), traj_temporal_mat.unsqueeze(3)), 3)
        score = self.forward(batch['traj'], mat1, batch['candiate_temporal_vec'], batch['traj_len'])
        score_sample, label_sample = self._sampling_prob(score, batch['target'])
        loss = F.cross_entropy(score_sample, label_sample)
        return loss

    def _sampling_prob(self, score, label):
        num_label, l_m = score.shape[0], score.shape[1]-1  # prob (N, L)
        label = label.view(-1)  # label (N)
        init_label = np.linspace(0, num_label-1, num_label)  # (N), [0 -- num_label-1]
        init_prob = torch.zeros(size=(num_label, self.num_neg+len(label)))  # (N, num_neg+num_label)
        random_ig = random.sample(range(1, l_m+1), self.num_neg)  # (num_neg) from (1 -- l_max)
        while len([lab for lab in label if lab in random_ig]) != 0:  # no intersection
            random_ig = random.sample(range(1, l_m+1), self.num_neg)
        random.seed(self.seed)
        self.seed += 1
        # place the pos labels ahead and neg samples in the end
        for k in range(num_label):
            for i in range(self.num_neg + len(label)):
                if i < len(label):
                    init_prob[k, i] = score[k, label[i]]
                else:
                    init_prob[k, i] = score[k, random_ig[i-len(label)]]
        return torch.FloatTensor(init_prob), torch.LongTensor(init_label)  # (N, num_neg+num_label), (N)
