from logging import getLogger
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class SSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dim needs to be divisible by num_heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_dim)

    def forward(self, values, keys, query):
        batch_size, num_nodes, input_window, embed_dim = query.shape

        values = values.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)
        keys = keys.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)
        query = query.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        energy = torch.einsum("bqthd,bkthd->bqkth", [queries, keys])

        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=2)

        out = torch.einsum("bqkth,bkthd->bqthd", [attention, values]).reshape(
            batch_size, num_nodes, input_window, self.num_heads * self.head_dim
        )

        out = self.fc_out(out)

        return out


class TSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dim needs to be divisible by num_heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_dim)

    def forward(self, values, keys, query):
        batch_size, num_nodes, input_window, embed_dim = query.shape

        values = values.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)
        keys = keys.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)
        query = query.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        energy = torch.einsum("bnqhd,bnkhd->bnqkh", [queries, keys])

        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)

        out = torch.einsum("bnqkh,bnkhd->bnqhd", [attention, values]).reshape(
            batch_size, num_nodes, input_window, self.num_heads * self.head_dim
        )

        out = self.fc_out(out)

        return out


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=torch.device('cpu')):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features).to(device))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features).to(device))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj_mx):
        support = torch.einsum("bnd, dh->bnh", [x, self.weight])
        output = torch.einsum("mn,bnh->bmh", [adj_mx, support])
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, device=device)
        self.gc2 = GraphConvolution(nhid, nclass, device=device)
        self.dropout_rate = dropout_rate

    def forward(self, x, adj_mx):
        x = F.relu(self.gc1(x, adj_mx))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.gc2(x, adj_mx)
        return F.log_softmax(x, dim=2)


class STransformer(nn.Module):
    def __init__(self, adj_mx, embed_dim=64, num_heads=2,
                 forward_expansion=4, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.adj_mx = torch.FloatTensor(adj_mx).to(device)
        self.D_S = nn.Parameter(torch.FloatTensor(adj_mx).to(device))
        self.embed_linear = nn.Linear(adj_mx.shape[0], embed_dim)

        self.attention = SSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )

        self.gcn = GCN(embed_dim, embed_dim * 2, embed_dim, dropout_rate, device=device)
        self.norm_adj = nn.InstanceNorm2d(1)

        self.dropout_layer = nn.Dropout(dropout_rate)
        self.fs = nn.Linear(embed_dim, embed_dim)
        self.fg = nn.Linear(embed_dim, embed_dim)

    def forward(self, value, key, query):
        batch_size, num_nodes, input_windows, embed_dim = query.shape
        D_S = self.embed_linear(self.D_S)
        D_S = D_S.expand(batch_size, input_windows, num_nodes, embed_dim)
        D_S = D_S.permute(0, 2, 1, 3)

        X_G = torch.Tensor(query.shape[0], query.shape[1], 0, query.shape[3]).to(self.device)
        self.adj_mx = self.adj_mx.unsqueeze(0).unsqueeze(0)
        self.adj_mx = self.norm_adj(self.adj_mx)
        self.adj_mx = self.adj_mx.squeeze(0).squeeze(0)

        for t in range(query.shape[2]):
            o = self.gcn(query[:, :, t, :], self.adj_mx)
            o = o.unsqueeze(2)
            X_G = torch.cat((X_G, o), dim=2)

        query = query + D_S
        attention = self.attention(value, key, query)

        x = self.dropout_layer(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_S = self.dropout_layer(self.norm2(forward + x))

        g = torch.sigmoid(self.fs(U_S) + self.fg(X_G))
        out = g * U_S + (1 - g) * X_G

        return out


class TTransformer(nn.Module):
    def __init__(self, TG_per_day=228, embed_dim=64, num_heads=2,
                 forward_expansion=4, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.temporal_embedding = nn.Embedding(TG_per_day, embed_dim)

        self.attention = TSelfAttention(embed_dim, num_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, value, key, query):
        batch_size, num_nodes, input_windows, embed_dim = query.shape

        D_T = self.temporal_embedding(torch.arange(0, input_windows).to(self.device))
        D_T = D_T.expand(batch_size, num_nodes, input_windows, embed_dim)

        query = query + D_T

        attention = self.attention(value, key, query)

        x = self.dropout_layer(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout_layer(self.norm2(forward + x))
        return out


class STTransformerBlock(nn.Module):
    def __init__(self, adj_mx, embed_dim=64, num_heads=2, TG_per_day=288,
                 forward_expansion=4, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.STransformer = STransformer(
            adj_mx, embed_dim=embed_dim, num_heads=num_heads,
            forward_expansion=forward_expansion, dropout_rate=dropout_rate, device=device)
        self.TTransformer = TTransformer(
            TG_per_day=TG_per_day, embed_dim=embed_dim, num_heads=num_heads,
            forward_expansion=forward_expansion, dropout_rate=dropout_rate, device=device)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, value, key, query):
        x1 = self.norm1(self.STransformer(value, key, query) + query)
        x2 = self.dropout_layer(self.norm2(self.TTransformer(x1, x1, x1) + x1))
        return x2


class Encoder(nn.Module):
    def __init__(self, adj_mx, embed_dim=64, num_layers=3, num_heads=2, TG_per_day=288,
                 forward_expansion=4, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.layers = nn.ModuleList([
            STTransformerBlock(
                adj_mx, embed_dim=embed_dim, num_heads=num_heads, TG_per_day=TG_per_day,
                forward_expansion=forward_expansion, dropout_rate=dropout_rate, device=device
            )
            for _ in range(num_layers)
        ])
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.dropout_layer(x)
        for layer in self.layers:
            out = layer(out, out, out)
        return out


class Transformer(nn.Module):
    def __init__(self, adj_mx, embed_dim=64, num_layers=3, num_heads=2, TG_per_day=288,
                 forward_expansion=4, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.encoder = Encoder(
            adj_mx, embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads, TG_per_day=TG_per_day,
            forward_expansion=forward_expansion, dropout_rate=dropout_rate, device=device,
        )

    def forward(self, src):
        enc_src = self.encoder(src)
        return enc_src


class STTN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._scaler = self.data_feature.get('scaler')
        self.adj_mx = self.data_feature.get('adj_mx', 1)
        # self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        # self.len_row = self.data_feature.get('len_row', 1)
        # self.len_column = self.data_feature.get('len_column', 1)

        self._logger = getLogger()

        self.device = config.get('device', torch.device('cpu'))

        self.embed_dim = config.get('embed_dim', 64)
        self.num_layers = config.get('num_layers', 3)
        self.num_heads = config.get('num_heads', 2)
        self.TG_per_day = config.get('TG_in_one_day', 288)  # number of time intevals per day
        self.forward_expansion = config.get('forward_expansion', 4)
        self.dropout_rate = config.get('dropout_rate', 0)

        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)

        self.conv1 = nn.Conv2d(self.feature_dim, self.embed_dim, 1)
        self.transformer = Transformer(
            self.adj_mx, embed_dim=self.embed_dim, num_layers=self.num_layers, num_heads=self.num_heads,
            TG_per_day=self.TG_per_day, forward_expansion=self.forward_expansion, dropout_rate=self.dropout_rate,
            device=self.device,
        )
        self.conv2 = nn.Conv2d(self.input_window, self.output_window, 1)
        self.conv3 = nn.Conv2d(self.embed_dim, self.output_dim, 1)
        self.act_layer = nn.ReLU()

    def forward(self, batch):
        inputs = batch['X']
        inputs = inputs.permute(0, 3, 2, 1)
        input_transformer = self.conv1(inputs)
        input_transformer = input_transformer.permute(0, 2, 3, 1)

        output_transformer = self.transformer(input_transformer)
        output_transformer = output_transformer.permute(0, 2, 1, 3)

        out = self.act_layer(self.conv2(output_transformer))
        out = out.permute(0, 3, 2, 1)
        out = self.conv3(out)
        out = out.permute(0, 3, 2, 1)
        return out

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true)

    def predict(self, batch):
        return self.forward(batch)
