#todo list 1. multi-model embeding 2. location encoder layer 3 grid mapping
import torch
import torch.nn as nn
from Output.mlp import FC



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
    

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=120):
        super(PositionalEmbedding, self).__init__()
        self.pos_emb_table = Embedding(max_len, d_model, zeros_pad=False, scale=False)
        pos_vector = torch.arange(max_len)
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pos_vector', pos_vector)

    def forward(self, x):
        pos_emb = self.pos_emb_table(self.pos_vector[:x.size(0)].unsqueeze(1).repeat(1, x.size(1)))
        x += pos_emb
        return self.dropout(x)
        
        
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