# coding: utf-8
from __future__ import print_function
from __future__ import division

from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from libcity.model.abstract_model import AbstractModel
import math


class GeoSAN(AbstractModel):

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.device = config['device']
        # depend on dataset
        self.num_neg = config['executor_config']['train']['num_negative_samples']
        self.temperature = config['executor_config']['train']['temperature']

        # from dataset
        # from train_dataset!!
        nuser = data_feature['nuser']
        nloc = data_feature['nloc']
        ntime = data_feature['ntime']
        nquadkey = data_feature['nquadkey']

        # from config
        user_dim = int(config['model_config']['user_embedding_dim'])
        loc_dim = int(config['model_config']['location_embedding_dim'])
        time_dim = int(config['model_config']['time_embedding_dim'])
        reg_dim = int(config['model_config']['region_embedding_dim'])
        # nhid = int(config['model_config']['hidden_dim_encoder'])
        nhead_enc = int(config['model_config']['num_heads_encoder'])
        # nhead_dec = int(config['model_config']['num_heads_decoder'])
        nlayers = int(config['model_config']['num_layers_encoder'])
        dropout = float(config['model_config']['dropout'])
        extra_config = config['model_config']['extra_config']
        # print(f"nloc: {nloc} \t loc_dim: {loc_dim}")
        # essential
        self.emb_loc = Embedding(nloc, loc_dim, zeros_pad=True, scale=True)
        self.emb_reg = Embedding(nquadkey, reg_dim, zeros_pad=True, scale=True)
        # optional
        self.emb_user = Embedding(nuser, user_dim, zeros_pad=True, scale=True)
        self.emb_time = Embedding(ntime, time_dim, zeros_pad=True, scale=True)
        ninp = user_dim

        pos_encoding = extra_config.get("position_encoding", "transformer")
        if pos_encoding == "embedding":
            self.pos_encoder = PositionalEmbedding(loc_dim + reg_dim, dropout)
        elif pos_encoding == "transformer":
            self.pos_encoder = PositionalEncoding(loc_dim + reg_dim, dropout)
        self.enc_layer = TransformerEncoderLayer(loc_dim + reg_dim, nhead_enc, loc_dim + reg_dim, dropout)
        self.encoder = TransformerEncoder(self.enc_layer, nlayers)

        self.region_pos_encoder = PositionalEmbedding(reg_dim, dropout, max_len=20)
        self.region_enc_layer = TransformerEncoderLayer(reg_dim, 1, reg_dim, dropout=dropout)
        self.region_encoder = TransformerEncoder(self.region_enc_layer, 2)

        if not extra_config.get("use_location_only", False):
            if extra_config.get("embedding_fusion", "multiply") == "concat":
                if extra_config.get("user_embedding", False):
                    self.lin = nn.Linear(user_dim + loc_dim + reg_dim + time_dim, ninp)
                else:
                    self.lin = nn.Linear(loc_dim + reg_dim, ninp)

        ident_mat = torch.eye(ninp)
        self.register_buffer('ident_mat', ident_mat)
        self.layer_norm = nn.LayerNorm(ninp)

        self.extra_config = extra_config
        self.dropout = dropout

    def predict(self, batch):
        """
        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: predict result of this batch
        """
        user, loc, time, region, trg, trg_reg, trg_nov, sample_probs, ds = batch

        user = user.to(self.device)
        loc = loc.to(self.device)
        time = time.to(self.device)
        region = region.to(self.device)
        trg = trg.to(self.device)
        trg_reg = trg_reg.to(self.device)
        sample_probs = sample_probs.to(self.device)
        src_mask = pad_sequence([torch.zeros(e, dtype=torch.bool).to(self.device) for e in ds],
                                batch_first=True, padding_value=True)
        att_mask = GeoSAN._generate_square_mask_(max(ds), self.device)

        if self.training:
            output = self.forward(user, loc, region, time, att_mask, src_mask,
                                  trg, trg_reg, att_mask.repeat(self.num_neg + 1, 1))
        else:
            output = self.forward(user, loc, region, time, att_mask, src_mask,
                                  trg, trg_reg, None, ds=ds)
        return output

    @staticmethod
    def _generate_square_mask_(sz, device):
        mask = (torch.triu(torch.ones(sz, sz).to(device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def calculate_loss(self, batch):
        """
        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: return training loss
        """
        # only support "WeightedProbBinaryCELoss"
        user, loc, time, region, trg, trg_reg, trg_nov, sample_probs, ds = batch

        user = user.to(self.device)
        loc = loc.to(self.device)
        time = time.to(self.device)
        region = region.to(self.device)
        trg = trg.to(self.device)
        trg_reg = trg_reg.to(self.device)
        sample_probs = sample_probs.to(self.device)
        src_mask = pad_sequence([torch.zeros(e, dtype=torch.bool).to(self.device) for e in ds],
                                batch_first=True, padding_value=True)
        att_mask = self._generate_square_mask_(max(ds), self.device)

        if self.training:
            output = self.forward(user, loc, region, time, att_mask, src_mask,
                                  trg, trg_reg, att_mask.repeat(self.num_neg + 1, 1))
        else:
            output = self.forward(user, loc, region, time, att_mask, src_mask,
                                  trg, trg_reg, None, ds=ds)

        # shape: [(1+K)*L, N]
        output = output.view(-1, loc.size(0), loc.size(1)).permute(2, 1, 0)
        # shape: [N, L, 1+K]
        pos_score, neg_score = output.split([1, self.num_neg], -1)
        weight = F.softmax(neg_score / self.temperature - torch.log(sample_probs), -1)
        loss = -F.logsigmoid(pos_score.squeeze()) + torch.sum(F.softplus(neg_score) * weight, dim=-1)
        keep = pad_sequence([torch.ones(e, dtype=torch.float32).to(self.device) for e in ds], batch_first=True)
        loss = torch.sum(loss * keep) / torch.sum(torch.tensor(ds).to(self.device))

        return loss

    def forward(self, src_user, src_loc, src_reg, src_time,
                src_square_mask, src_binary_mask, trg_loc, trg_reg, mem_mask, ds=None):
        loc_emb_src = self.emb_loc(src_loc)
        if self.extra_config.get("user_location_only", False):
            src = loc_emb_src
        else:
            user_emb_src = self.emb_user(src_user)
            # (L, N, LEN_QUADKEY, REG_DIM)
            reg_emb = self.emb_reg(src_reg)
            reg_emb = reg_emb.view(reg_emb.size(0) * reg_emb.size(1),
                                   reg_emb.size(2), reg_emb.size(3)).permute(1, 0, 2)
            # (LEN_QUADKEY, L * N, REG_DIM)

            reg_emb = self.region_pos_encoder(reg_emb)
            reg_emb = self.region_encoder(reg_emb)
            # avg pooling
            reg_emb = torch.mean(reg_emb, dim=0)

            # reg_emb, _ = self.region_gru_encoder(reg_emb, self.h_0.expand(4, reg_emb.size(1), -1).contiguous())
            # reg_emb = reg_emb[-1, :, :]

            # (L, N, REG_DIM)
            reg_emb = reg_emb.view(loc_emb_src.size(0), loc_emb_src.size(1), reg_emb.size(1))

            time_emb = self.emb_time(src_time)
            if self.extra_config.get("embedding_fusion", "multiply") == "multiply":
                if self.extra_config.get("user_embedding", False):
                    src = loc_emb_src * reg_emb * time_emb * user_emb_src
                else:
                    src = loc_emb_src * reg_emb * time_emb
            else:
                if self.extra_config.get("user_embedding", False):
                    src = torch.cat([user_emb_src, loc_emb_src, reg_emb, time_emb], dim=-1)
                else:
                    src = torch.cat([loc_emb_src, reg_emb], dim=-1)

        if self.extra_config.get("size_sqrt_regularize", True):
            src = src * math.sqrt(src.size(-1))

        src = self.pos_encoder(src)
        # shape: [L, N, ninp]
        src = self.encoder(src, mask=src_square_mask)

        # shape: [(1+K)*L, N, loc_dim]
        loc_emb_trg = self.emb_loc(trg_loc)

        reg_emb_trg = self.emb_reg(trg_reg)  # [(1+K)*L, N, LEN_QUADKEY, REG_DIM]
        # (LEN_QUADKEY, (1+K)*L * N, REG_DIM)
        reg_emb_trg = reg_emb_trg.view(reg_emb_trg.size(0) * reg_emb_trg.size(1),
                                       reg_emb_trg.size(2), reg_emb_trg.size(3)).permute(1, 0, 2)
        reg_emb_trg = self.region_pos_encoder(reg_emb_trg)
        reg_emb_trg = self.region_encoder(reg_emb_trg)
        reg_emb_trg = torch.mean(reg_emb_trg, dim=0)
        # [(1+K)*L, N, REG_DIM]
        reg_emb_trg = reg_emb_trg.view(loc_emb_trg.size(0),
                                       loc_emb_trg.size(1), reg_emb_trg.size(1))

        loc_emb_trg = torch.cat([loc_emb_trg, reg_emb_trg], dim=-1)
        if self.extra_config.get("use_attention_as_decoder", False):
            # multi-head attention
            output, _ = F.multi_head_attention_forward(
                query=loc_emb_trg,
                key=src,
                value=src,
                embed_dim_to_check=src.size(2),
                num_heads=1,
                in_proj_weight=None,
                in_proj_bias=None,
                bias_k=None,
                bias_v=None,
                add_zero_attn=None,
                dropout_p=0.0,
                out_proj_weight=self.ident_mat,
                out_proj_bias=None,
                training=self.training,
                key_padding_mask=src_binary_mask,
                need_weights=False,
                attn_mask=mem_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.ident_mat,
                k_proj_weight=self.ident_mat,
                v_proj_weight=self.ident_mat
            )

            if self.training:
                src = src.repeat(loc_emb_trg.size(0) // src.size(0), 1, 1)
            else:
                src = src[torch.tensor(ds) - 1, torch.arange(len(ds)), :]
                src = src.unsqueeze(0).repeat(loc_emb_trg.size(0), 1, 1)

            output += src
            output = self.layer_norm(output)
        else:
            # No attention
            if self.training:
                output = src.repeat(loc_emb_trg.size(0) // src.size(0), 1, 1)
            else:
                output = src[torch.tensor(ds) - 1, torch.arange(len(ds)), :]
                output = output.unsqueeze(0).repeat(loc_emb_trg.size(0), 1, 1)

        # shape: [(1+K)*L, N]
        output = torch.sum(output * loc_emb_trg, dim=-1)
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class Embedding(nn.Module):
    def __init__(self, vocab_size, num_units, zeros_pad=True, scale=True):
        '''Embeds a given Variable.
        Args:
          vocab_size: An int. Vocabulary size.
          num_units: An int. Number of embedding hidden units.
          zero_pad: A boolean. If True, all the values of the fist row (id 0)
            should be constant zeros.
          scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
        '''
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale
        self.lookup_table = nn.Parameter(torch.Tensor(vocab_size, num_units))
        nn.init.xavier_normal_(self.lookup_table.data)
        if self.zeros_pad:
            self.lookup_table.data[0, :].fill_(0)

    def forward(self, inputs):
        if self.zeros_pad:
            self.padding_idx = 0
        else:
            self.padding_idx = -1
        outputs = F.embedding(
            inputs, self.lookup_table,
            self.padding_idx, None, 2, False, False)  # copied from torch.nn.modules.sparse.py

        if self.scale:
            outputs = outputs * (self.num_units ** 0.5)

        return outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


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
