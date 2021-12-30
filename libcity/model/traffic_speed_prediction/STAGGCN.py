import math
from logging import getLogger
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
# from torch_scatter import scatter  # NB : Install this package manually

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def remove_self_loops(edge_index: torch.Tensor):
    return edge_index[:, edge_index[0] != edge_index[1]]


def maybe_num_nodes(edge_index: torch.Tensor, num_nodes: Optional[int] = None):
    if num_nodes is not None:
        return num_nodes
    else:
        return int(edge_index.max()) + 1


def add_self_loops(edge_index: torch.Tensor, num_nodes: Optional[int] = None):
    return torch.cat((edge_index,
                      torch.arange(maybe_num_nodes(edge_index, num_nodes))
                      .repeat(2, 1)
                      .to(edge_index.device)), dim=1)


def softmax(x: torch.Tensor, index: torch.Tensor, num_nodes: Optional[int] = None, dim: int = 0):
    N = maybe_num_nodes(index, num_nodes)
    x_max = scatter(x, index, dim, dim_size=N, reduce='max').index_select(dim, index)
    out = (x - x_max).exp()
    out_sum = scatter(out, index, dim, dim_size=N, reduce='sum').index_select(dim, index)
    return out / out_sum


class STAGGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._scaler = self.data_feature.get('scaler')
        self.adj_mx = self.data_feature.get('adj_mx', 1)
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.input_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.ext_dim = self.data_feature.get('ext_dim', 1)

        # 以下两项是STAG-GCN对数据集额外进行预处理得到的边关系数据
        # 对数据集预处理得到的空间邻接边集
        self.edge_index = self.data_feature.get('edge_index', torch.tensor([[], []], dtype=torch.long))  # 空间邻接边
        # 对数据集预处理得到的语义邻接边集
        self.dtw_edge_index = self.data_feature.get('dtw_edge_index', torch.tensor([[], []], dtype=torch.long))  # 语义邻接边

        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.graph_dim = config.get('graph_dim', 32)
        self.tcn_dim = config.get('tcn_dim', [10])
        self.attn_head = config.get('atten_head', 3)
        self.choice = config.get('choice', [1, 1, 1])
        self.batch_size = config.get('batch_size', 64)

        self.edge_index = self.edge_index.to(self.device)
        self.dtw_edge_index = self.dtw_edge_index.to(self.device)

        self.model = STAGGCNModel(input_dim=self.input_dim,
                                  output_dim=self.output_dim,
                                  node_num=self.num_nodes,
                                  seq_len=self.input_window,
                                  pred_len=self.output_window,
                                  graph_dim=self.graph_dim,
                                  tcn_dim=self.tcn_dim,
                                  attn_head=self.attn_head,
                                  choice=self.choice).to(self.device)

    def forward(self, batch):
        x = batch['X']  # shape = (batch_size, input_length, num_nodes, input_dim)

        # [batch_size, pred_len, num_nodes, output_dim]
        return self.model(x, self.edge_index, self.dtw_edge_index)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true)

    def predict(self, batch):
        # one-inference multi-step prediction
        return self.forward(batch)


class STAGGCNModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1,
                 node_num=325, seq_len=12, pred_len=6, graph_dim=32,
                 tcn_dim=[10], attn_head=4, choice=[1, 1, 1]):
        super(STAGGCNModel, self).__init__()
        self.node_num = node_num
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.graph_dim = graph_dim
        # self.output_dim = seq_len + np.sum(choice) * graph_dim
        self.pred_len_raw = np.sum(choice) * graph_dim

        self.STCell = STCell(node_num, seq_len, graph_dim, tcn_dim,
                             choice=choice, attn_head=attn_head,
                             input_dim=input_dim, output_dim=output_dim)
        self.output_linear = nn.Linear(in_features=self.pred_len_raw, out_features=self.pred_len)
        # self.output_linear_0 = nn.Linear(in_features=self.graph_dim, out_features=256)
        # self.output_linear_1 = nn.Linear(in_features=256, out_features=self.pred_len)

    def forward(self, x, edge_index, dtw_edge_index):
        # x: [batch_size, seq_len, num_nodes, input_dim]
        # st_output: [batch_size, num_nodes, output_dim, sum(choice)*graph_dim ==
        # [batch_size, num_nodes, output_dim, pred_len_raw]]
        st_output = self.STCell(x, edge_index, dtw_edge_index)
        output = st_output

        # [batch_size, num_nodes, output_dim, pred_len]
        output = self.output_linear(output)
        # output = F.relu(self.output_linear_0(output))
        # output = self.output_linear_1(output)
        # output = torch.reshape(output, (-1, self.node_num, self.pred_len))

        # [batch_size, pred_len, num_nodes, output_dim]
        return output.permute(0, 3, 1, 2).contiguous()


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x: [batch_size*input_dim*num_nodes, n_inputs, seq_len]
        # self.conv1(x): [batch_size*input_dim*num_nodes, n_outputs, ...]
        # self.chomp1(self.conv2(x)): [batch_size*input_dim*num_nodes, n_outputs, seq_len]
        # return: [batch_size*input_dim*num_nodes, n_outputs, seq_len]
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch_size*num_nodes, input_dim, seq_len]
        # return: [batch_size*num_nodes, output_dim*num_channels[-1], seq_len]
        return self.network(x)


class LearnedGCN(nn.Module):
    def __init__(self, node_num, in_feature, out_feature):
        super(LearnedGCN, self).__init__()
        self.node_num = node_num
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.source_embed = nn.Parameter(torch.Tensor(self.node_num, 10))
        self.target_embed = nn.Parameter(torch.Tensor(10, self.node_num))
        self.linear = nn.Linear(self.in_feature, self.out_feature)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.source_embed.size(0))
        self.source_embed.data.uniform_(-stdv, stdv)
        self.target_embed.data.uniform_(-stdv, stdv)

    def forward(self, input):
        learned_matrix = F.softmax(F.relu(torch.mm(self.source_embed, self.target_embed)), dim=1)
        output = learned_matrix.matmul(input)
        output = self.linear(output)
        return output


class GATConv(nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int,
                 heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True):
        super(GATConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_heads = heads

        self.negative_slope = negative_slope
        self.dropout = dropout

        self.bias = bias
        self.concat = concat
        self.add_self_loops = add_self_loops

        self.linear = nn.Linear(self.in_channels, self.attn_heads * self.out_channels, bias=False)
        self.attn_j = nn.Parameter(torch.Tensor(1, self.attn_heads, self.out_channels))
        self.attn_i = nn.Parameter(torch.Tensor(1, self.attn_heads, self.out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.attn_heads * self.out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.init_weights()

    def init_weights(self):
        self._glorot(self.linear.weight)
        self._glorot(self.attn_j)
        self._glorot(self.attn_i)
        self._zeros(self.bias)

    @staticmethod
    def _glorot(t: torch.Tensor):
        if t is None:
            return
        stdv = math.sqrt(6. / (t.size(-2) * t.size(-1)))
        t.data.uniform_(-stdv, stdv)

    @staticmethod
    def _zeros(t: torch.Tensor):
        if t is None:
            return
        t.data.fill_(0.)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        num_nodes = x.size(0)

        edge_index = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)

        edge_index_j, edge_index_i = edge_index

        # x: [num_nodes, num_features]
        # [num_edges, attn_heads, out_channels]
        x_j = self.linear(x).view(-1, self.attn_heads, self.out_channels)[edge_index_j]
        x_i = self.linear(x).view(-1, self.attn_heads, self.out_channels)[edge_index_i]

        # [num_edges, attn_heads]
        alpha_j = (x_j * self.attn_j).sum(dim=-1)[edge_index_j]
        alpha_i = (x_i * self.attn_i).sum(dim=-1)[edge_index_i]

        # message passing
        # [num_edges, attn_heads]
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, x_i.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # [num_edges, attn_heads, out_channels]
        message = x_j * alpha.unsqueeze(-1)

        out = scatter(message, edge_index_i, dim=0, reduce='add')

        if self.concat:
            out = out.view(-1, self.attn_heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out += self.bias

        return out


class STCell(nn.Module):
    def __init__(self, node_num=524, seq_len=12, graph_dim=16, tcn_dim=[10],
                 choice=[1, 1, 1], attn_head=2, input_dim=1, output_dim=1):
        super(STCell, self).__init__()
        self.node_num = node_num
        self.seq_len = seq_len
        self.graph_dim = graph_dim
        self.tcn_dim = tcn_dim
        self.pred_len_raw = np.sum(choice) * graph_dim
        self.choice = choice
        # self.jklayer = JumpingKnowledge("max")
        # self.jklayer = JumpingKnowledge("lstm", self.graph_dim, 1)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_features = seq_len * input_dim

        self.seq_linear = nn.Linear(in_features=self.input_dim * seq_len, out_features=self.input_dim * seq_len)

        if choice[0] == 1:
            print("[TCN]")
            print("node_num:", node_num, "\tattn_head:", attn_head)
            # one node of one input feature per embedding element
            self.self_attn = nn.MultiheadAttention(embed_dim=node_num * input_dim, num_heads=attn_head)
            # expand convolution output_dimension by output_dim
            self.tcn = TemporalConvNet(num_inputs=self.input_dim,
                                       num_channels=[x * self.output_dim for x in self.tcn_dim])
            self.tlinear = nn.Linear(in_features=self.output_dim * self.tcn_dim[-1] * self.seq_len,
                                     out_features=self.output_dim * self.graph_dim)

        if choice[1] == 1:
            print("[SP]")
            self.sp_origin = nn.Linear(in_features=self.input_dim * seq_len, out_features=self.output_dim * graph_dim)
            self.sp_gconv1 = GATConv(self.input_dim * seq_len, self.output_dim * graph_dim, heads=3, concat=False)
            self.sp_gconv2 = GATConv(self.output_dim * graph_dim, self.output_dim * graph_dim, heads=3, concat=False)
            self.sp_gconv3 = GATConv(self.output_dim * graph_dim, self.output_dim * graph_dim, heads=3, concat=False)
            self.sp_gconv4 = GATConv(self.output_dim * graph_dim, self.output_dim * graph_dim, heads=1, concat=False)
            # self.sp_gconv5 = GATConv(graph_dim, graph_dim, heads = 1, concat = False)
            self.sp_source_embed = nn.Parameter(torch.Tensor(self.node_num, 12))
            self.sp_target_embed = nn.Parameter(torch.Tensor(12, self.node_num))
            self.sp_linear_1 = nn.Linear(self.input_dim * seq_len, self.output_dim * self.graph_dim)
            self.sp_linear_2 = nn.Linear(self.output_dim * self.graph_dim, self.output_dim * self.graph_dim)
            self.sp_linear_3 = nn.Linear(self.output_dim * self.graph_dim, self.output_dim * self.graph_dim)
            self.sp_linear_4 = nn.Linear(self.output_dim * self.graph_dim, self.output_dim * self.graph_dim)
            # self.sp_linear_5 = nn.Linear(self.graph_dim, self.graph_dim)
            # self.sp_jklayer = JumpingKnowledge("max")

            nn.init.xavier_uniform_(self.sp_source_embed)
            nn.init.xavier_uniform_(self.sp_target_embed)

        if choice[2] == 1:
            print("[DTW]")
            self.dtw_origin = nn.Linear(in_features=self.input_dim * seq_len, out_features=self.output_dim * graph_dim)
            self.dtw_gconv1 = GATConv(self.input_dim * seq_len, self.output_dim * graph_dim, heads=3, concat=False)
            self.dtw_gconv2 = GATConv(self.output_dim * graph_dim, self.output_dim * graph_dim, heads=3, concat=False)
            self.dtw_gconv3 = GATConv(self.output_dim * graph_dim, self.output_dim * graph_dim, heads=3, concat=False)
            self.dtw_gconv4 = GATConv(self.output_dim * graph_dim, self.output_dim * graph_dim, heads=3, concat=False)
            # self.dtw_gconv5 = GATConv(graph_dim, graph_dim, heads = 1, concat = False)
            self.dtw_source_embed = nn.Parameter(torch.Tensor(self.node_num, 12))
            self.dtw_target_embed = nn.Parameter(torch.Tensor(12, self.node_num))
            self.dtw_linear_1 = nn.Linear(self.input_dim * self.seq_len, self.output_dim * self.graph_dim)
            self.dtw_linear_2 = nn.Linear(self.output_dim * self.graph_dim, self.output_dim * self.graph_dim)
            self.dtw_linear_3 = nn.Linear(self.output_dim * self.graph_dim, self.output_dim * self.graph_dim)
            self.dtw_linear_4 = nn.Linear(self.output_dim * self.graph_dim, self.output_dim * self.graph_dim)
            # self.dtw_linear_5 = nn.Linear(self.graph_dim, self.graph_dim)
            # self.dtw_jklayer = JumpingKnowledge("max")

            nn.init.xavier_uniform_(self.dtw_source_embed)
            nn.init.xavier_uniform_(self.dtw_target_embed)

    def forward(self, x, edge_index, dtw_edge_index):
        # x: [batch_size, seq_len, num_nodes, input_dim]
        output_list = [0, 0, 0]
        batch_size = x.shape[0]

        if self.choice[0] == 1:
            # [seq_len, batch_size, input_dim*num_nodes]
            attn_input = x.permute(1, 0, 3, 2).reshape(self.seq_len, batch_size, -1).contiguous()
            # [seq_len, batch_size, input_dim*num_nodes]
            # input_dim*num_nodes is the embedding dimension
            attn_output, _ = self.self_attn(attn_input, attn_input, attn_input)
            # [seq_len, batch_size, input_dim*num_nodes]
            attn_output = torch.tanh(attn_output + attn_input)
            # [batch_size*num_nodes, input_dim, seq_len]
            attn_output = attn_output.reshape(self.seq_len, batch_size, self.input_dim, self.node_num) \
                .permute(1, 3, 2, 0) \
                .reshape(-1, self.input_dim, self.seq_len)

            # [batch_size*num_nodes, input_dim, seq_len]
            tcn_input = attn_output
            # [batch_size*num_nodes, output_dim*self.tcn_dim[-1], seq_len]
            tcn_output = self.tcn(tcn_input)
            # [batch_size*num_nodes, output_dim*self.tcn_dim[-1]*seq_len]
            tcn_output = torch.reshape(tcn_output,
                                       (-1, self.output_dim * self.tcn_dim[-1] * self.seq_len))
            # [batch_size*num_nodes, output_dim*self.graph_dim]
            tcn_output = self.tlinear(tcn_output)
            # [batch_size, num_nodes, output_dim, self.graph_dim]
            tcn_output = torch.reshape(tcn_output, (batch_size, self.node_num, self.output_dim, self.graph_dim))

            output_list[0] = tcn_output

        if self.choice[1] == 1 or self.choice[2] == 1:
            # [batch_size, num_nodes, input_dim*seq_len]
            sp_gout_0 = x.permute(0, 2, 3, 1).reshape(-1, self.input_dim * self.seq_len).contiguous()
            dtw_gout_0 = sp_gout_0.detach().clone()

        if self.choice[1] == 1:
            # [batch_size*num_nodes, input_dim*seq_len]
            sp_gout_0 = self.seq_linear(sp_gout_0) + sp_gout_0

            # [num_nodes, num_nodes]
            sp_learned_matrix = F.softmax(F.relu(torch.mm(self.sp_source_embed, self.sp_target_embed)), dim=1)

            # GATConv: [input_dim*seq_len, output_dim*graph_dim]
            # [batch_size*num_nodes, output_dim*graph_dim]
            sp_gout_1 = self.sp_gconv1(sp_gout_0, edge_index)
            # [batch_size, num_nodes, input_dim*seq_len]
            adp_input_1 = torch.reshape(sp_gout_0, (-1, self.node_num, self.input_dim * self.seq_len))
            # [batch_size, num_nodes, output_dim*graph_dim]
            sp_adp_1 = self.sp_linear_1(sp_learned_matrix.matmul(F.dropout(adp_input_1, p=0.1)))
            # [batch_size*num_nodes, output_dim*graph_dim]
            sp_adp_1 = torch.reshape(sp_adp_1, (-1, self.output_dim * self.graph_dim))
            # [batch_size*num_nodes, output_dim*graph_dim]
            sp_origin = self.sp_origin(sp_gout_0)
            # [batch_size*num_nodes, output_dim*graph_dim]
            sp_output_1 = torch.tanh(sp_gout_1) * torch.sigmoid(sp_adp_1) + sp_origin * (1 - torch.sigmoid(sp_adp_1))

            # [batch_size*num_nodes, output_dim*graph_dim]
            sp_gout_2 = self.sp_gconv2(torch.tanh(sp_output_1), edge_index)
            # [batch_size, num_nodes, output_dim*graph_dim]
            adp_input_2 = torch.reshape(torch.tanh(sp_output_1), (-1, self.node_num, self.output_dim * self.graph_dim))
            # [batch_size, num_nodes, output_dim*graph_dim]
            sp_adp_2 = self.sp_linear_2(sp_learned_matrix.matmul(F.dropout(adp_input_2, p=0.1)))
            # [batch_size*num_nodes, output_dim*graph_dim]
            sp_adp_2 = torch.reshape(sp_adp_2, (-1, self.output_dim * self.graph_dim))
            # [batch_size*num_nodes, output_dim*graph_dim]
            sp_output_2 = F.leaky_relu(sp_gout_2) * torch.sigmoid(sp_adp_2) + \
                          sp_output_1 * (1 - torch.sigmoid(sp_adp_2))

            # [batch_size*num_nodes, output_dim*graph_dim]
            sp_gout_3 = self.sp_gconv3(F.relu(sp_output_2), edge_index)
            # [batch_size, num_nodes, output_dim*graph_dim]
            adp_input_3 = torch.reshape(F.relu(sp_output_2), (-1, self.node_num, self.output_dim * self.graph_dim))
            # [batch_size, num_nodes, output_dim*graph_dim]
            sp_adp_3 = self.sp_linear_3(sp_learned_matrix.matmul(F.dropout(adp_input_3, p=0.1)))
            # [batch_size*num_nodes, output_dim*graph_dim]
            sp_adp_3 = torch.reshape(sp_adp_3, (-1, self.output_dim * self.graph_dim))
            # [batch_size*num_nodes, output_dim*graph_dim]
            sp_output_3 = F.relu(sp_gout_3) * torch.sigmoid(sp_adp_3) + sp_output_2 * (1 - torch.sigmoid(sp_adp_3))

            sp_gout_4 = self.sp_gconv4(F.relu(sp_output_3), edge_index)
            adp_input_4 = torch.reshape(F.relu(sp_output_3), (-1, self.node_num, self.output_dim * self.graph_dim))
            sp_adp_4 = self.sp_linear_4(sp_learned_matrix.matmul(F.dropout(adp_input_4, p=0.1)))
            sp_adp_4 = torch.reshape(sp_adp_4, (-1, self.output_dim * self.graph_dim))
            # [batch_size*num_nodes, output_dim*graph_dim]
            sp_output_4 = F.relu(sp_gout_4) * torch.sigmoid(sp_adp_4) + sp_output_3 * (1 - torch.sigmoid(sp_adp_4))

            # sp_gout_5 = self.sp_gconv5(F.relu(sp_output_4), edge_index)
            # adp_input_5 = torch.reshape(F.relu(sp_output_4), (-1, self.node_num, self.graph_dim))
            # sp_adp_5 = self.sp_linear_5(sp_learned_matrix.matmul(F.dropout(adp_input_5,p=0.1)))
            # sp_adp_5 = torch.reshape(sp_adp_5, (-1, self.graph_dim))
            # sp_output_5 = F.relu(sp_gout_5) * torch.sigmoid(sp_adp_5) + sp_output_4 * (1 - torch.sigmoid(sp_adp_5))

            # [batch_size, num_nodes, output_dim, graph_dim]
            sp_output = torch.reshape(sp_output_4, (batch_size, self.node_num, self.output_dim, self.graph_dim))
            # sp_output = sp_output_4
            output_list[1] = sp_output

        if self.choice[2] == 1:
            dtw_gout_0 = self.seq_linear(dtw_gout_0) + dtw_gout_0

            dtw_learned_matrix = F.softmax(F.relu(torch.mm(self.dtw_source_embed, self.dtw_target_embed)), dim=1)

            dtw_gout_1 = self.dtw_gconv1(dtw_gout_0, dtw_edge_index)
            adp_input_1 = torch.reshape(dtw_gout_0, (-1, self.node_num, self.input_dim * self.seq_len))
            dtw_adp_1 = self.dtw_linear_1(dtw_learned_matrix.matmul(F.dropout(adp_input_1, p=0.1)))
            dtw_adp_1 = torch.reshape(dtw_adp_1, (-1, self.output_dim * self.graph_dim))
            dtw_origin = self.dtw_origin(dtw_gout_0)
            dtw_output_1 = torch.tanh(dtw_gout_1) * torch.sigmoid(dtw_adp_1) + \
                           dtw_origin * (1 - torch.sigmoid(dtw_adp_1))

            dtw_gout_2 = self.dtw_gconv2(torch.tanh(dtw_output_1), dtw_edge_index)
            adp_input_2 = torch.reshape(torch.tanh(dtw_output_1), (-1, self.node_num, self.output_dim * self.graph_dim))
            dtw_adp_2 = self.dtw_linear_2(dtw_learned_matrix.matmul(F.dropout(adp_input_2, p=0.1)))
            dtw_adp_2 = torch.reshape(dtw_adp_2, (-1, self.output_dim * self.graph_dim))
            dtw_output_2 = F.leaky_relu(dtw_gout_2) * torch.sigmoid(dtw_adp_2) + \
                           dtw_output_1 * (1 - torch.sigmoid(dtw_adp_2))

            dtw_gout_3 = self.dtw_gconv3(F.relu(dtw_output_2), dtw_edge_index)
            adp_input_3 = torch.reshape(F.relu(dtw_output_2), (-1, self.node_num, self.output_dim * self.graph_dim))
            dtw_adp_3 = self.dtw_linear_3(dtw_learned_matrix.matmul(F.dropout(adp_input_3, p=0.1)))
            dtw_adp_3 = torch.reshape(dtw_adp_3, (-1, self.output_dim * self.graph_dim))
            dtw_output_3 = F.relu(dtw_gout_3) * torch.sigmoid(dtw_adp_3) + dtw_output_2 * (1 - torch.sigmoid(dtw_adp_3))

            dtw_gout_4 = self.dtw_gconv4(F.relu(dtw_output_3), dtw_edge_index)
            adp_input_4 = torch.reshape(F.relu(dtw_output_3), (-1, self.node_num, self.output_dim * self.graph_dim))
            dtw_adp_4 = self.dtw_linear_4(dtw_learned_matrix.matmul(F.dropout(adp_input_4, p=0.1)))
            dtw_adp_4 = torch.reshape(dtw_adp_4, (-1, self.output_dim * self.graph_dim))
            # [batch_size*num_nodes, output_dim*graph_dim]
            dtw_output_4 = F.relu(dtw_gout_4) * torch.sigmoid(dtw_adp_4) + dtw_output_3 * (1 - torch.sigmoid(dtw_adp_4))

            # dtw_gout_5 = self.dtw_gconv5(F.relu(dtw_output_4), dtw_edge_index)
            # adp_input_5 = torch.reshape(F.relu(dtw_output_4), (-1, self.node_num, self.graph_dim))
            # dtw_adp_5 = self.dtw_linear_5(dtw_learned_matrix.matmul(F.dropout(adp_input_5,p=0.1)))
            # dtw_adp_5 = torch.reshape(dtw_adp_5, (-1, self.graph_dim))
            # dtw_output_5 = \
            # F.relu(dtw_gout_5) * torch.sigmoid(dtw_adp_5) + dtw_output_4 * (1 - torch.sigmoid(dtw_adp_5))

            # [batch_size, num_nodes, output_dim, graph_dim]
            dtw_output = torch.reshape(dtw_output_4, (batch_size, self.node_num, self.output_dim, self.graph_dim))
            # dtw_output = dtw_output_4
            output_list[2] = dtw_output

        # output_list[*]: [batch_size, num_nodes, output_dim, graph_dim]
        # cell_output: [batch_size, num_nodes, output_dim, sum(choice)*graph_dim]
        step = 0
        for i in range(len(self.choice)):
            if self.choice[i] == 1 and step == 0:
                cell_output = output_list[i]
                step += 1
            elif self.choice[i] == 1:
                cell_output = torch.cat((cell_output, output_list[i]), dim=3)

        # cell_output = self.jklayer([output_list[0], output_list[1], output_list[2]])
        # cell_output = self.out(cell_output)

        # cell_output = torch.reshape(cell_output, (-1, self.pred_len_raw))

        return cell_output
