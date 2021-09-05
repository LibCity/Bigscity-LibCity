import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class CRANN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._scaler = self.data_feature.get('scaler')
        self.adj_mx = self.data_feature.get('adj_mx')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.ext_dim = self.data_feature.get('ext_dim', 0)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._logger = getLogger()

        # ---- spatial module
        self.dim_x = config.get('dim_x', 5)
        self.dim_y = config.get('dim_y', 6)
        # ---- temporal module
        self.n_hidden_tem = config.get('n_hidden_tem', 100)
        self.n_layers_tem = config.get('n_layers_tem', 1)
        # ---- dense module
        self.n_hidden_dns = config.get('n_hidden_dns', 0)
        self.n_layers_dns = config.get('n_layers_dns', 1)
        self.n_ar = config.get('n_ar', 4)

        self.device = config.get('device', torch.device('cpu'))
        self.input_window = config.get('input_window', 24)
        self.output_window = config.get('output_window', 24)

        self.len_inputs = self.output_window * (self.num_nodes + self.ext_dim + 1)
        self.len_outputs = self.output_window * self.num_nodes

        self.spatial_model = AttentionCNN(in_channels=self.input_window, out_channels=self.output_window,
                                          dim_x=self.dim_x, dim_y=self.dim_y)
        self.temporal_encoder = EncoderLSTM(self.feature_dim, self.n_hidden_tem, device=self.device)
        self.temporal_decoder = BahdanauDecoder(self.n_hidden_tem, self.output_dim)
        self.mlp = MLP(n_inputs=self.len_inputs + self.n_ar * self.num_nodes,
                       n_outputs=self.len_outputs,
                       n_layers=self.n_layers_dns, n_hidden=self.n_hidden_dns)

    def evaluate_temp_att(self, encoder, decoder, batch, n_pred, device):
        output = torch.Tensor().to(device)
        h = encoder.init_hidden(batch.size(0))
        encoder_output, h = encoder(batch, h)
        decoder_hidden = h
        decoder_input = torch.zeros(batch.size(0), 1, device=device)
        for k in range(n_pred):
            decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output)
            decoder_input = decoder_output
            output = torch.cat((output, decoder_output), 1)
        return output

    def forward(self, batch):
        x_time = batch['x_time']
        x_space = batch['x_space']
        x_ext = batch['x_ext']

        y_time = self.evaluate_temp_att(self.temporal_encoder, self.temporal_decoder,
                                        x_time, self.output_window, self.device)
        y_space = self.spatial_model(x_space)[0]
        x = torch.cat((y_time.unsqueeze(2), y_space.squeeze().view(-1, self.output_window, self.num_nodes),
                       x_ext), dim=2).view(-1, self.len_inputs)
        x = torch.cat((x, x_space[:, -self.n_ar:].view(-1, self.n_ar * self.num_nodes)), dim=1)
        y_pred = self.mlp(x).view(-1, self.output_window, self.dim_x, self.dim_y)
        return y_pred

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        # print('y_true', y_true.shape)
        # print('y_predicted', y_predicted.shape)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true)

    def predict(self, batch):
        return self.forward(batch)


class AttentionCNN(nn.Module):
    """
    ---------------
    | Description |
    ---------------
    Spatial module with spatio-temporal attention

    --------------
    | Attributes |
    --------------
    in_channels : int
        Number of input timesteps
    out_channels : int
        Number of output timesteps
    dim_x : int
        Dimension of x-axis for input images
    dim_y : int
        Dimension of y-axis for input images

    -----------
    | Methods |
    -----------
    forward(x)
        Forward pass of the network
    """

    def __init__(self, in_channels, out_channels, dim_x, dim_y):
        super(AttentionCNN, self).__init__()
        # Variables
        self.out_channels = out_channels
        self.dim_x = dim_x
        self.dim_y = dim_y

        # Conv blocks
        self.conv_block1 = ConvBlock(in_channels, 64, 5)

        # Attention
        self.att1 = AttentionBlock(dim_x, dim_y, 24, method='hadamard')

        # Output
        self.regressor = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.regressor(out)
        out, att = self.att1(out)
        return out, att


class ConvBlock(nn.Module):
    """
    ---------------
    | Description |
    ---------------
    Convolutional blocks of num_conv convolutions with out_features channels

    --------------
    | Attributes |
    --------------
    in_features : int
        Number of input channels
    out_features : int
        Number of middle and output channels
    num_conv : int
        Number of convolutions

    -----------
    | Methods |
    -----------
    forward(x)
        Forward pass of the network
    """

    def __init__(self, in_features, out_features, num_conv):
        super(ConvBlock, self).__init__()
        features = [in_features] + [out_features for i in range(num_conv)]
        layers = []
        for i in range(len(features) - 1):
            layers.append(
                nn.Conv2d(in_channels=features[i], out_channels=features[i + 1], kernel_size=3, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(num_features=features[i + 1], affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
        self.op = nn.Sequential(*layers)

    def forward(self, x):
        return self.op(x)


class AttentionBlock(nn.Module):
    """
    ---------------
    | Description |
    ---------------
    Attentional block for spatio-temporal attention mechanism

    --------------
    | Attributes |
    --------------
    dim_x : int
        Dimension of x-axis for input images
    dim_y : int
        Dimension of y-axis for input images
    timesteps : int
        Number of input timesteps
    method : str
        Attentional function to calculate attention weights

    -----------
    | Methods |
    -----------
    forward(x)
        Forward pass of the network
    """

    def __init__(self, dim_x, dim_y, timesteps, method='hadamard'):
        super(AttentionBlock, self).__init__()
        # Variables
        self.method = method
        self.weight = nn.Parameter(torch.FloatTensor(timesteps, dim_x * dim_y, dim_x * dim_y))
        torch.nn.init.xavier_uniform_(self.weight)
        if method == 'general':
            self.fc = nn.Linear(timesteps * (dim_x * dim_y) ** 2, timesteps * (dim_x * dim_y) ** 2, bias=False)
        elif method == 'concat':
            self.fc = nn.Linear(timesteps * (dim_x * dim_y) ** 2, timesteps * (dim_x * dim_y) ** 2, bias=False)

    def forward(self, x, y=0):
        N, T, W, H = x.size()
        if self.method == 'hadamard':
            xp = x.view(N, T, -1).repeat(1, 1, W * H).view(N, T, W * H, W * H)
            wp = self.weight.expand_as(xp)
            alig_scores = wp.mul(xp)
        elif self.method == 'general':
            xp = x.view(N, T, -1).repeat(1, 1, W * H).view(N, T, W * H, W * H)
            wp = self.weight.expand_as(xp)
            alig_scores = self.fc((wp.mul(xp)).view(N, -1))
        elif self.method == 'concat':
            xp = x.view(N, T, -1).repeat(1, 1, W * H).view(N, T, W * H, W * H)
            wp = self.weight.expand_as(xp)
            alig_scores = torch.tanh(self.fc((wp + xp).view(N, -1)))
        elif self.method == 'dot':
            xp = x.view(N, T, -1).repeat(1, 1, W * H).view(N, T, W * H, W * H)
            alig_scores = self.weight.matmul(xp)

        att_weights = F.softmax(alig_scores.view(N, T, W * H, W * H), dim=3)
        out = att_weights.matmul(x.view(N, T, -1).unsqueeze(3))
        return out.view(N, T, W, H), att_weights


class EncoderLSTM(nn.Module):
    """
    ---------------
    | Description |
    ---------------
    Encoder for temporal module

    --------------
    | Attributes |
    --------------
    input_size : int
        Number of input features
    hidden_size : int
        Dimension of hidden space
    n_layers : int
        Number of layers for the encoder
    drop_prob : float
        Dropout for the encoder
    device : int/str
        Device in which hiddens are stored

    -----------
    | Methods |
    -----------
    forward(x)
        Forward pass of the network
    """

    def __init__(self, input_size, hidden_size, n_layers=1, drop_prob=0, device='cuda'):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=drop_prob, batch_first=True)

    def forward(self, inputs, hidden):
        output, hidden = self.lstm(inputs, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device))


class BahdanauDecoder(nn.Module):
    """
    ---------------
    | Description |
    ---------------
    Decoder an attention mechanism for temporal module

    --------------
    | Attributes |
    --------------
    hidden_size : int
        Dimension of hidden space
    output_size : int
        Number of output features
    n_layers : int
        Number of layers for the encoder
    drop_prob : float
        Dropout for the encoder

    -----------
    | Methods |
    -----------
    forward(x)
        Forward pass of the network
    """

    def __init__(self, hidden_size, output_size, n_layers=1, drop_prob=0.1):
        super(BahdanauDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prob = drop_prob

        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))
        torch.nn.init.xavier_uniform_(self.weight)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size + self.output_size, self.hidden_size, batch_first=True)
        self.fc_prediction = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs):
        encoder_outputs = encoder_outputs.squeeze()

        # Calculating Alignment Scores
        x = torch.tanh(self.fc_hidden(hidden[0].view(-1, 1, self.hidden_size)) +
                       self.fc_encoder(encoder_outputs))

        alignment_scores = x.matmul(self.weight.unsqueeze(2))

        # Softmaxing alignment scores to get Attention weights
        attn_weights = F.softmax(alignment_scores.view(inputs.size(0), -1), dim=1)

        # Multiplying the Attention weights with encoder outputs to get the context vector
        self.context_vector = torch.matmul(attn_weights.unsqueeze(1), encoder_outputs)

        # Concatenating context vector with embedded input word
        output = torch.cat((inputs, self.context_vector.squeeze(1)), 1).unsqueeze(1)
        # Passing the concatenated vector as input to the LSTM cell
        output, hidden = self.lstm(output, hidden)

        output = self.fc_prediction(output).squeeze(2)

        return output, hidden, attn_weights


class MLP(nn.Module):
    """
    ---------------
    | Description |
    ---------------
    Dense module

    --------------
    | Attributes |
    --------------
    n_inputs : int
        Number of input features
    n_outputs : int
        Number of output features
    n_layers : int
        Number of layers
    n_hidden : int
        Dimension of hidden layers

    -----------
    | Methods |
    -----------
    forward(x)
        Forward pass of the network
    """

    def __init__(self, n_inputs, n_outputs, n_layers=1, n_hidden=0, dropout=0):
        super(MLP, self).__init__()
        if n_layers < 1:
            raise ValueError('Number of layers needs to be at least 1.')
        elif n_layers == 1:
            self.module = nn.Linear(n_inputs, n_outputs)
        else:
            modules = [nn.Linear(n_inputs, n_hidden), nn.ReLU(), nn.Dropout(dropout)]
            n_layers -= 1
            while n_layers > 1:
                modules += [nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Dropout(dropout)]
                n_layers -= 1
            modules.append(nn.Linear(n_hidden, n_outputs))
            self.module = nn.Sequential(*modules)

    def forward(self, x):
        return self.module(x)
