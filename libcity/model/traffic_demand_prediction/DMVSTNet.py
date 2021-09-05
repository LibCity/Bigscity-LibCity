import random
from decimal import Decimal
from logging import getLogger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from tqdm import tqdm

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class SpatialViewConv(nn.Module):
    def __init__(self, inp_channel, oup_channel, kernel_size, stride=1, padding=0):
        super(SpatialViewConv, self).__init__()
        self.inp_channel = inp_channel
        self.oup_channel = oup_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels=inp_channel, out_channels=oup_channel,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch = nn.BatchNorm2d(oup_channel)
        self.relu = nn.ReLU()

    def forward(self, inp):
        return self.relu(self.batch(self.conv(inp)))


class TemporalView(nn.Module):
    def __init__(self, fc_oup_dim, lstm_oup_dim):
        super(TemporalView, self).__init__()
        self.lstm = nn.LSTM(fc_oup_dim, lstm_oup_dim)

    def forward(self, inp):
        # inp = [T, B, fc_oup_dim]
        lstm_res, (h, c) = self.lstm(inp)
        # lstm_res = [T, B, lstm_oup_dim]
        # h/c = [1, B, lstm_oup_dim]
        return h[0]  # [B, lstm_oup_dim]


class VoseAlias(object):
    """
    Adding a few modifs to https://github.com/asmith26/Vose-Alias-Method
    """

    def __init__(self, dist):
        """
        (VoseAlias, dict) -> NoneType
        """
        self.dist = dist
        self.alias_initialisation()

    def alias_initialisation(self):
        """
        Construct probability and alias tables for the distribution.
        """
        # Initialise variables
        n = len(self.dist)
        self.table_prob = {}  # probability table
        self.table_alias = {}  # alias table
        scaled_prob = {}  # scaled probabilities
        small = []  # stack for probabilities smaller that 1
        large = []  # stack for probabilities greater than or equal to 1

        # Construct and sort the scaled probabilities into their appropriate stacks
        print("1/2. Building and sorting scaled probabilities for alias table...")
        for o, p in tqdm(self.dist.items()):
            scaled_prob[o] = Decimal(p) * n

            if scaled_prob[o] < 1:
                small.append(o)
            else:
                large.append(o)

        print("2/2. Building alias table...")
        # Construct the probability and alias tables
        while small and large:
            s = small.pop()
            l = large.pop()

            self.table_prob[s] = scaled_prob[s]
            self.table_alias[s] = l

            scaled_prob[l] = (scaled_prob[l] + scaled_prob[s]) - Decimal(1)

            if scaled_prob[l] < 1:
                small.append(l)
            else:
                large.append(l)

        # The remaining outcomes (of one stack) must have probability 1
        while large:
            self.table_prob[large.pop()] = Decimal(1)

        while small:
            self.table_prob[small.pop()] = Decimal(1)
        self.listprobs = list(self.table_prob)

    def alias_generation(self):
        """
        Yields a random outcome from the distribution.
        """
        # Determine which column of table_prob to inspect
        col = random.choice(self.listprobs)
        # Determine which outcome to pick in that column
        if self.table_prob[col] >= random.uniform(0, 1):
            return col
        else:
            return self.table_alias[col]

    def sample_n(self, size):
        """
        Yields a sample of size n from the distribution, and print the results to stdout.
        """
        for i in range(size):
            yield self.alias_generation()


class Line(nn.Module):
    def __init__(self, size, embed_dim=128, order=1):
        super(Line, self).__init__()

        assert order in [1, 2], print("Order should either be int(1) or int(2)")

        self.embed_dim = embed_dim
        self.order = order
        self.nodes_embeddings = nn.Embedding(size, embed_dim)

        if order == 2:
            self.contextnodes_embeddings = nn.Embedding(size, embed_dim)
            # Initialization
            self.contextnodes_embeddings.weight.data = self.contextnodes_embeddings.weight.data.uniform_(
                -.5, .5) / embed_dim

        # Initialization
        self.nodes_embeddings.weight.data = self.nodes_embeddings.weight.data.uniform_(
            -.5, .5) / embed_dim

    def forward(self, v_i, v_j, negsamples, device):

        v_i = self.nodes_embeddings(v_i).to(device)

        if self.order == 2:
            v_j = self.contextnodes_embeddings(v_j).to(device)
            negativenodes = -self.contextnodes_embeddings(negsamples).to(device)

        else:
            v_j = self.nodes_embeddings(v_j).to(device)
            negativenodes = -self.nodes_embeddings(negsamples).to(device)

        mulpositivebatch = torch.mul(v_i, v_j)
        positivebatch = F.logsigmoid(torch.sum(mulpositivebatch, dim=1))

        mulnegativebatch = torch.mul(v_i.view(len(v_i), 1, self.embed_dim), negativenodes)
        negativebatch = torch.sum(
            F.logsigmoid(
                torch.sum(mulnegativebatch, dim=2)
            ),
            dim=1)
        loss = positivebatch + negativebatch
        return -torch.mean(loss)

    def get_embeddings(self):
        if self.order == 1:
            return self.nodes_embeddings.weight.data
        else:
            return self.contextnodes_embeddings.weight.data


def negSampleBatch(sourcenode, targetnode, negsamplesize, weights, nodedegrees, nodesaliassampler, t=10e-3):
    """
    For generating negative samples.
    """
    negsamples = 0
    while negsamples < negsamplesize:
        samplednode = nodesaliassampler.sample_n(1)
        if (samplednode == sourcenode) or (samplednode == targetnode):
            continue
        else:
            negsamples += 1
            yield samplednode


def makeData(samplededges, negsamplesize, weights, nodedegrees, nodesaliassampler):
    for e in samplededges:
        sourcenode, targetnode = e[0], e[1]
        negnodes = []
        for negsample in negSampleBatch(sourcenode, targetnode, negsamplesize,
                                        weights, nodedegrees, nodesaliassampler):
            for node in negsample:
                negnodes.append(node)
        yield [e[0], e[1]] + negnodes


class SemanticView(nn.Module):
    def __init__(self, config, data_feature):
        super(SemanticView, self).__init__()

        self.num_nodes = data_feature.get('num_nodes', 1)
        self.len_row = data_feature.get('len_row', 1)  # 网格行数
        self.len_column = data_feature.get('len_column', 1)  # 网格列数

        self.embedding_dim = config.get('line_dimension')
        self.semantic_dim = config.get('semantic_dim')
        self.order = config.get('line_order')
        self.negsamplesize = config.get('line_negsamplesize')
        self.embedding_dim = config.get("line_dimension")
        self.batchsize = config.get('line_batchsize')
        self.epochs = config.get('line_epochs')
        self.lr = config.get('line_learning_rate')
        self.negativepower = config.get('line_negativepower')
        self.fc = nn.Linear(self.embedding_dim, self.semantic_dim)
        self.device = config.get('device', torch.device('cpu'))

        print("Data Pretreatment: Line embedding...")
        [edgedistdict, nodedistdict, weights, nodedegrees] = data_feature.get('dtw_graph')

        edgesaliassampler = VoseAlias(edgedistdict)
        nodesaliassampler = VoseAlias(nodedistdict)
        batchrange = int(len(edgedistdict) / self.batchsize)

        line = Line(self.num_nodes, self.embedding_dim, self.order)
        opt = optim.SGD(line.parameters(), lr=self.lr, momentum=0.9, nesterov=True)

        for _ in range(self.epochs):
            for _ in trange(batchrange):
                samplededges = edgesaliassampler.sample_n(self.batchsize)
                batch = list(makeData(samplededges, self.negsamplesize, weights, nodedegrees, nodesaliassampler))
                batch = torch.LongTensor(batch)
                v_i = batch[:, 0]
                v_j = batch[:, 1]
                negsamples = batch[:, 2:]
                line.zero_grad()
                loss = line(v_i, v_j, negsamples, self.device)
                loss.backward()
                opt.step()

        self.embedding = line.get_embeddings().reshape((self.len_row, self.len_column, -1)).to(self.device)

    def forward(self, i, j):
        return self.fc(self.embedding[i, j, :])


class DMVSTNet(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._scaler = self.data_feature.get('scaler')  # 用于数据归一化
        self.adj_mx = self.data_feature.get('adj_mx', 1)  # 邻接矩阵
        self.num_nodes = self.data_feature.get('num_nodes', 1)  # 网格个数
        self.feature_dim = self.data_feature.get('feature_dim', 1)  # 输入维度
        self.output_dim = self.data_feature.get('output_dim', 1)  # 输出维度
        self.len_row = self.data_feature.get('len_row', 1)  # 网格行数
        self.len_column = self.data_feature.get('len_column', 1)  # 网格列数
        self._logger = getLogger()

        self.device = config.get('device', torch.device('cpu'))
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.local_image_size = config.get('local_image_size', 5)
        self.padding_size = self.local_image_size // 2
        self.cnn_hidden_dim_first = config.get('cnn_hidden_dim_first', 32)
        self.fc_oup_dim = config.get('fc_oup_dim', 64)
        self.lstm_oup_dim = config.get('lstm_oup_dim', 32)
        self.graph_embedding_dim = config.get('graph_embedding_dim', 32)
        self.semantic_dim = config.get('semantic_dim', 6)

        self.padding = nn.ZeroPad2d((self.padding_size, self.padding_size, self.padding_size, self.padding_size))

        # 三层Local CNN
        self.local_conv1 = SpatialViewConv(inp_channel=self.feature_dim, oup_channel=self.cnn_hidden_dim_first,
                                           kernel_size=3, stride=1, padding=1)
        self.local_conv2 = SpatialViewConv(inp_channel=self.cnn_hidden_dim_first,
                                           oup_channel=self.cnn_hidden_dim_first,
                                           kernel_size=3, stride=1, padding=1)
        self.local_conv3 = SpatialViewConv(inp_channel=self.cnn_hidden_dim_first,
                                           oup_channel=self.cnn_hidden_dim_first,
                                           kernel_size=3, stride=1, padding=1)

        # 全连接降维
        self.fc1 = nn.Linear(in_features=self.cnn_hidden_dim_first * self.local_image_size * self.local_image_size,
                             out_features=self.fc_oup_dim)

        # TemporalView
        self.temporalLayers = TemporalView(self.fc_oup_dim, self.lstm_oup_dim)

        #  SemanticView
        self.semanticLayer = SemanticView(config, self.data_feature)

        # 输出层
        self.fc2 = nn.Linear(in_features=self.lstm_oup_dim + self.semantic_dim, out_features=self.output_dim)

    def spatial_forward(self, grid_batch):
        x1 = self.local_conv1(grid_batch)
        x2 = self.local_conv2(x1)
        x3 = self.local_conv3(x2)
        x4 = self.fc1(torch.flatten(x3, start_dim=1))
        return x4

    def forward(self, batch):
        # input转换为卷积运算的格式 (B, input_window, len_row, len_col, feature_dim)
        x = batch['X'].permute(0, 1, 4, 2, 3)  # (B, input_window, feature_dim, len_row, len_col)
        batch_size = x.shape[0]
        x = x.reshape((batch_size * self.input_window, self.feature_dim, self.len_row, self.len_column))

        # 对输入进行0填充
        x_padding = self.padding(x)
        # 构造输出
        oup = torch.zeros((batch_size, 1, self.len_row, self.len_column, self.output_dim)).to(self.device)
        # 对每个grid进行预测
        for i in range(self.padding_size, self.len_row - self.padding_size):
            for j in range(self.padding_size, self.len_column - self.padding_size):
                spatial_res = self.spatial_forward(
                    x_padding[:, :, i - self.padding_size:i + self.padding_size + 1,
                    j - self.padding_size: j + self.padding_size + 1])
                # print('spatial_res', spatial_res.shape)  # (B*T, fc_oup_dim)
                seq_res = spatial_res.reshape((batch_size, self.input_window, self.fc_oup_dim)).permute(1, 0, 2)
                # print('seq_res', seq_res.shape)  # (T, B, fc_oup_dim)
                temporal_res = self.temporalLayers(seq_res)
                # print('temporal_res', temporal_res.shape)  # (B, lstm_oup_dim)
                emb_res = self.semanticLayer(i, j)
                emb_res = emb_res.repeat(batch_size, 1)
                res = self.fc2(torch.cat([temporal_res, emb_res], dim=1))
                oup[:, :, i, j, :] = res.reshape(batch_size, 1, self.output_dim)
        return oup

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        # print('y_true', y_true.shape, y_true.device, y_true.requires_grad)
        # print('y_predicted', y_predicted.shape, y_predicted.device, y_predicted.requires_grad)
        res = loss.masked_mse_torch(y_predicted, y_true)
        return res

    def predict(self, batch):
        # 多步预测
        x = batch['X']  # (batch_size, input_window, len_row, len_column, feature_dim)
        y = batch['y']  # (batch_size, input_window, len_row, len_column, feature_dim)
        y_preds = []
        x_ = x.clone()
        for i in range(self.output_window):
            batch_tmp = {'X': x_}
            y_ = self.forward(batch_tmp)  # (batch_size, 1, len_row, len_column, output_dim)
            y_preds.append(y_.clone())
            if y_.shape[-1] < x_.shape[-1]:  # output_dim < feature_dim
                y_ = torch.cat([y_, y[:, i:i + 1, :, :, self.output_dim:]], dim=-1)
            x_ = torch.cat([x_[:, 1:, :, :, :], y_], dim=1)
        y_preds = torch.cat(y_preds, dim=1)  # (batch_size, output_length, len_row, len_column, output_dim)
        return y_preds
