from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np
import math
from math import pi
from inspect import isfunction
from einops import rearrange, repeat

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class SpatialEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(SpatialEmbedding, self).__init__()
        self.spa_emb = nn.Embedding(c_in, d_model)

    def forward(self, x):
        x = x.long()
        spa_emb = self.spa_emb(x)

        return spa_emb


class PositionalEmbedding(nn.Module):
    '''local position'''

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]  # [1, 144, 512]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='replicate')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x  # [134, 144, 512]


class FixedEmbedding(nn.Module):
    '''
    use embedding as weight to transform timestamp features
    '''

    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        '''
        month, monthday, weekday, hour, minute
        '''
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        self.freq = freq

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if self.freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3, "10min": 4}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)


class Wr_node_embedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(Wr_node_embedding, self).__init__()
        self.w = nn.Parameter(torch.randn((c_in, d_model), dtype=torch.float32))

    def forward(self, x):
        x = torch.einsum("btn,nd->btd", (x, self.w))
        return x


class DataEmbedding(nn.Module):  # embed_type=TimF                          args.num_nodes
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, args=None):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # self.wr_node_embedding = Wr_node_embedding(args.num_nodes, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # 只有下面这项用了embed_type
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        # x.shape = [134, 144, 12], x_mark.shape = [134, 144, 3]
        x = self.value_embedding(x) + self.position_embedding(
            x)  # + self.wr_node_embedding(x) # [134, 144, 512], [1, 144, 512]
        if x_mark is not None:
            x = x + self.temporal_embedding(x_mark)  # [134, 144, 512]     x_mark input 2277 12 1
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        x = self.value_embedding(x)
        if x_mark is not None:
            x = x + self.temporal_embedding(x_mark)
        return self.dropout(x)


# helper functions
def exists(val):
    return val is not None


# rotary embedding helper functions

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


def apply_rotary_emb(freqs, t, start_index=0):
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[
        -1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t_left, t, t_right), dim=-1)


# learned rotation helpers

def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if exists(freq_ranges):
        rotations = torch.einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r=2)
    return apply_rotary_emb(rotations, t, start_index=start_index)


class RotaryEmbedding(nn.Module):
    def __init__(
            self,
            dim,
            skip=False,
            custom_freqs=None,
            freqs_for='lang',
            theta=10000,
            max_freq=10,
            num_freqs=1,
            learned_freq=False
    ):
        super().__init__()
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.cache = dict()
        self.skip = skip
        if learned_freq:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer('freqs', freqs)

    def rotate_queries_or_keys(self, t, seq_dim=1):
        if self.skip:
            return t
        device = t.device
        seq_len = t.shape[seq_dim]
        # SinusoidalPositionEmbedding
        freqs = self.forward(lambda: torch.arange(seq_len, device=device), cache_key=seq_len)
        return apply_rotary_emb(freqs, t)

    def forward(self, t, cache_key=None):
        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]

        if isfunction(t):
            t = t()

        freqs = self.freqs

        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)

        if exists(cache_key):
            self.cache[cache_key] = freqs

        return freqs


class TriangularCausalMask():
    def __init__(self, B, L, gau=False, device="cpu"):
        if gau:
            mask_shape = [B, L, L]
        else:
            mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  # L=324
        _, S, _, D = values.shape  # S=72
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # sample_k = 25 从144中取25个
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        # [134, 8, 144, 25]

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        # topk函数返回两个，第一个是值，第二个是索引，这里只要索引

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        # scores_top.shape = [134, 8, 25, 144], 前log量个q
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))  # [134, 512, 144]
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)  # [134, 512, 72]
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    # [134, 144, 512]
    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)  # [134, 144, 512]
                x = conv_layer(x)  # [134, 72, 512]
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns  # x.shape = [134, 72, 512]


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s)
            attns.append(attn)
        x_stack = torch.cat(x_stack, -2)

        return x_stack, attns


class Informer_t_and_s(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """

    def __init__(self, pred_len, output_attention, enc_in, d_model, embed, freq, dropout, dec_in, enc_attn, factor,
                 n_heads, d_ff, activation, e_layers, distil, mix, d_layers, out_size, num_nodes):
        super(Informer_t_and_s, self).__init__()
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.d_model = d_model

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        Attn = ProbAttention if enc_attn == 'prob' else FullAttention

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, factor, attention_dropout=dropout,
                             output_attention=output_attention),
                        d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads, mix=mix),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, out_size, bias=True)
        )

        self.w = nn.Parameter(
            torch.randn((num_nodes, d_model), dtype=torch.float32).unsqueeze(dim=0).unsqueeze(dim=2))

    # x_mark_enc 和 x_mark_dec 是3维的时间emb
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        b, n, t, d = x_enc.shape
        # x_enc=[134, 144, 12], x_mark_enc=[134, 144, 3], x_dec=[134, 324, 12], x_mark_dec=[134, 324, 3]
        enc_out = self.enc_embedding(x_enc.reshape((b * n), t, d), x_mark_enc)  # enc_out.shape = [134, 144, 512]
        # 加node_emb
        enc_out = enc_out.reshape(b, n, t, -1) + self.w  # bntd
        enc_out = enc_out.reshape(b, n * t, -1)  # b(nt)d

        enc_out, attns = self.encoder(enc_out,
                                      attn_mask=enc_self_mask)

        # enc_out = enc_out.reshape(b, n, -1, self.args.d_model)  # b,n,t/2,d

        dec_out = self.dec_embedding(x_dec.reshape((b * n), 2 * t, d), x_mark_dec)  # [134, 324, 512]
        # 加node_emb d_ff=512, d_model=64
        dec_out = dec_out.reshape(b, n, 2 * t, self.d_model) + self.w
        dec_out = dec_out.reshape(b, n * 2 * t, self.d_model)

        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)  # [134, 324, 1]
        dec_out = dec_out.reshape(b, n, 2 * t, -1)
        dec_out = dec_out[:, :, -t:, :]  # bntd
        dec_out = dec_out.permute(0, 2, 1, 3)  # btnd

        return dec_out


class Trafformer(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))

        # data
        self.num_nodes = self.data_feature.get('num_nodes')
        self._scaler = self.data_feature.get('scaler')
        self.output_dim = self.data_feature.get('output_dim')
        self.input_dim = self.output_dim

        # model
        self.output_window = config.get("output_window", 12)
        self.output_attention = config.get("output_attention", False)
        self.enc_in = self.input_dim
        self.d_model = config.get("d_model", 32)
        self.embed = config.get("embed", "timeF")
        self.freq = config.get("freq", "m")
        self.dropout = config.get("dropout", 0.0)
        self.dec_in = self.input_dim
        self.enc_attn = config.get("enc_attn", 'full')
        self.factor = config.get("factor", 50)
        self.n_heads = config.get("n_heads", 4)
        self.d_ff = config.get("d_ff", 256)
        self.activation = config.get("activation", 'gelu')
        self.e_layers = config.get("e_layers", 2)
        self.distil = config.get('distil', False)
        self.mix = config.get('mix', True)
        self.d_layers = config.get("d_layers", 1)
        self.out_size = self.output_dim
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 16)

        self.xformer = Informer_t_and_s(pred_len=self.output_window, output_attention=self.output_attention,
                                        enc_in=self.enc_in, d_model=self.d_model, embed=self.embed, freq=self.freq,
                                        dropout=self.dropout, dec_in=self.dec_in, enc_attn=self.enc_attn,
                                        factor=self.factor, n_heads=self.n_heads, d_ff=self.d_ff,
                                        activation=self.activation, e_layers=self.e_layers, distil=self.distil,
                                        mix=self.mix, d_layers=self.d_layers, out_size=self.out_size,
                                        num_nodes=self.num_nodes)

    def forward(self, x, y):
        b, t, n, d = x.shape
        # xy BTNF -> BNTF
        x = x.transpose(1, 2)
        y = y.transpose(1, 2)
        x_enc = x[:, :, :, :self.input_dim]
        x_mark_enc = x.reshape((b * n), t, d)[:, :, self.input_dim:]
        x_dec = torch.cat([x_enc, torch.zeros_like(x_enc)], dim=2)
        x_mark_dec = torch.cat([x_mark_enc, y.reshape((b * n), t, d)[:, :, self.input_dim:]], dim=1)

        dec_out = self.xformer(x_enc, x_mark_enc, x_dec, x_mark_dec)

        return dec_out

    def predict(self, batch):
        x = batch['X']
        y = batch['y']
        return self.forward(x, y)

    def calculate_loss(self, batch):
        y = batch['y']
        output = self.predict(batch)
        y_true = self._scaler.inverse_transform(y[..., :self.output_dim])
        y_pred = self._scaler.inverse_transform(output[..., :self.output_dim])
        c_loss = loss.masked_mae_torch(y_pred, y_true, null_val=0.0)
        return c_loss / self.gradient_accumulation_steps
