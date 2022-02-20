import torch
import torch.nn as nn

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def graph_conv_op(X, num_filters, graph_conv_filters, kernel):
    """
    X: (B, N, F)
    graph_conv_filters: (N * K, N)
    """
    conv_op = torch.matmul(graph_conv_filters, X)
    # (B, N * K, F)
    conv_op = torch.split(conv_op, num_filters, dim=1)
    # K , (B, N, F)
    conv_op = torch.cat(conv_op, dim=2)
    # (B, N, K * F)
    conv_out = torch.mm(conv_op, kernel)
    # (B, N, O)
    return conv_out


class MultiGraphCNN(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_filters,
                 activation=True,
                 use_bn=False,
                 use_bias=True):
        super(MultiGraphCNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_filters = num_filters
        if activation:
            self.activation = nn.ReLU()
        else:
            self.register_parameter('activation', None)

        self.kernel = nn.Parameter(data=torch.zeros((self.num_filters * self.input_dim, self.output_dim)))

        if use_bias:
            self.bias = nn.Parameter(data=torch.zeros((self.output_dim,)))
        else:
            self.register_parameter('bias', None)

        if use_bn:
            self.bn = nn.BatchNorm1d(num_features=input_dim)
        else:
            self.register_parameter('bn', None)

    def forward(self, X, adj_mx):
        """
        X: (B, N, F)
        adj_mx: (N * K, N)

        output: (B, N, O)
        """
        output = graph_conv_op(X, self.num_filters, adj_mx, self.kernel)

        if self.bias is not None:
            output += self.bias

        if self.activation is not None:
            output = self.activation(output)

        if self.bn is not None:
            output = output.swapaxes(1, 2)
            output = self.bn(output)
            output = output.swapaxes(1, 2)

        return output


class ConvolutionBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_filters):
        super(ConvolutionBlock, self).__init__()
        self.shortcut_path = MultiGraphCNN(input_dim=input_dim, output_dim=hidden_dims[-1], num_filters=num_filters)
        self.main_path = nn.ModuleList()

        input_dim_ = input_dim
        for dim in hidden_dims:
            self.main_path.append(MultiGraphCNN(input_dim=input_dim_, output_dim=dim, num_filters=num_filters))
            input_dim_ = dim

        self.activation = nn.ReLU()

    def forward(self, X, adj_mx):
        """
          X: (B, N, F)
          adj_mx: (N * K, N)
        """
        X_shortcut = X
        for model in self.main_path:
            X = model(X, adj_mx)

        X_shortcut = self.shortcut_path(X_shortcut, adj_mx)

        X += X_shortcut
        X = self.activation(X)
        return X


class IdentityBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_filters):
        super(IdentityBlock, self).__init__()
        self.main_path = nn.ModuleList()

        input_dim_ = input_dim
        for dim in hidden_dims:
            self.main_path.append(MultiGraphCNN(input_dim=input_dim_, output_dim=dim, num_filters=num_filters))
            input_dim_ = dim

        self.activation = nn.ReLU()

    def forward(self, X, adj_mx):
        """
          X: (B, N, F)
          adj_mx: (N * K, N)
        """
        X_shortcut = X
        for model in self.main_path:
            X = model(X, adj_mx)

        X += X_shortcut
        X = self.activation(X)
        return X


class Encoder(nn.Module):
    def __init__(self, config, data_feature):
        super(Encoder, self).__init__()
        self.input_dim = config.get('input_dim')
        self.num_filters = config.get('num_filters')
        self.encode_conv_dims = config.get('encode_conv_dims', [128 // 4, 128 // 4, 128])
        self.lstm_dims = config.get('lstm_dims', [128, 64])
        self.spatial_emb_dim = config.get('spatial_emb_dim', 90)
        self.temporal_emb_dim = config.get('temporal_emb_dim', 30)
        self.emb_dim = config.get('emb_dim', 90)

        self.num_nodes = data_feature.get('num_nodes')
        self.adj_mx = data_feature['adj_mx']

        self.convolution_block = ConvolutionBlock(input_dim=self.input_dim, hidden_dims=self.encode_conv_dims,
                                                  num_filters=self.num_filters)
        self.identity_block = IdentityBlock(input_dim=self.encode_conv_dims[-1], hidden_dims=self.encode_conv_dims,
                                            num_filters=self.num_filters)

        self.fc_spatial = nn.Linear(in_features=self.encode_conv_dims[-1] * self.num_nodes,
                                    out_features=self.spatial_emb_dim, bias=True)

        self.muti_lstm = nn.ModuleList()

        input_size_ = self.num_nodes
        for dim in self.lstm_dims:
            self.muti_lstm.append(nn.LSTM(input_size=input_size_, hidden_size=dim, batch_first=True))
            input_size_ = dim

        self.muti_lstm.append(nn.LSTM(input_size=input_size_, hidden_size=self.spatial_emb_dim, batch_first=True))

        self.fc_temporal = nn.Linear(in_features=self.spatial_emb_dim,
                                     out_features=self.temporal_emb_dim, bias=True)

        self.fc_decode = nn.Linear(in_features=self.spatial_emb_dim + self.temporal_emb_dim,
                                   out_features=self.num_nodes * self.emb_dim, bias=True)

    def forward(self, X):
        graph_encoded = self.convolution_block(X, self.adj_mx)
        graph_encoded = self.identity_block(graph_encoded, self.adj_mx)
        graph_encoded = torch.flatten(graph_encoded, start_dim=1)
        graph_encoded = self.fc_spatial(graph_encoded)
        # (B, E1)

        lstm_encoded = X.swapaxes(1, 2)
        # (B, F, N), F treated as input length, and N treated as feature
        for model in self.muti_lstm:
            lstm_encoded, _ = model(lstm_encoded)

        lstm_encoded = lstm_encoded[:, -1, :]
        # (B, E2)
        lstm_encoded = self.fc_temporal(lstm_encoded)

        merge_encoded = torch.cat([graph_encoded, lstm_encoded], dim=1)

        merge_encoded = self.fc_decode(merge_encoded)

        merge_encoded = merge_encoded.reshape((-1, self.num_nodes, self.emb_dim))
        # (B, E1)
        return merge_encoded


class Decoder(nn.Module):
    def __init__(self, config, data_feature):
        super(Decoder, self).__init__()
        self.emb_dim = config.get('emb_dim', 90)
        self.num_filters = config.get('num_filters')
        self.output_dim = config.get('output_dim', 1)

        self.decode_conv_dims = config.get('decode_conv_dims', [128 // 4, 128 // 4, 128])
        self.convolution_block = ConvolutionBlock(input_dim=self.emb_dim, hidden_dims=self.decode_conv_dims,
                                                  num_filters=self.num_filters)
        self.identity_block = IdentityBlock(input_dim=self.decode_conv_dims[-1], hidden_dims=self.decode_conv_dims,
                                            num_filters=self.num_filters)

        self.output = MultiGraphCNN(input_dim=self.decode_conv_dims[-1], output_dim=self.output_dim,
                                    num_filters=self.num_filters)

        self.adj_mx = data_feature['adj_mx']

    def forward(self, emb):
        emb = self.convolution_block(emb)
        emb = self.identity_block(emb)
        output = self.output(emb)
        return output


class STEDGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.encoder = Encoder(config, data_feature)
        self.decoder = Decoder(config, data_feature)

    def calculate_loss(self, batch):
        pass

    def forward(self, X):
        emb = self.encoder(X)
        return self.decoder(emb)


if __name__ == '__main__':
    config = {
        'input_dim': 4,
        'num_filters': 8,
        'encode_conv_dims': [32, 32, 128],
        'lstm_dims': [128, 64],
        'spatial_emb_dim': 90,
        'temporal_emb_dim': 30,
        'emb_dim': 90,
        'output_dim': 1,
        'decode_conv_dims': [32, 32, 128]
    }
