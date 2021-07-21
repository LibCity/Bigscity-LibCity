import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
# from torch_geometric.nn import GATConv
from logging import getLogger
from libtraffic.model import loss
from libtraffic.model.abstract_traffic_state_model import AbstractTrafficStateModel
import numpy as np
import math

"""
这个模型只能处理输入维度是1，输出也是1
环境torch_geometric很难配置
"""


class STAGGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._scaler = self.data_feature.get('scaler')
        self.adj_mx = self.data_feature.get('adj_mx', 1)
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
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
        self.atten_head = config.get('atten_head', 5)
        self.choice = config.get('choice', [1, 1, 1])
        self.batch_size = config.get('batch_size', 64)

        self.edge_index = self.edge_index.to(self.device)
        self.dtw_edge_index = self.dtw_edge_index.to(self.device)

        self.model = STAGGCNModel(node_num=self.num_nodes,
                                  seq_len=self.input_window,
                                  pred_len=self.output_window,
                                  graph_dim=self.graph_dim,
                                  tcn_dim=self.tcn_dim,
                                  atten_head=self.atten_head,
                                  choice=self.choice).to(self.device)

    def forward(self, batch):
        x = batch['X']  # shape = (batch_size, input_length, num_nodes, feature_dim)
        x = x[:, :, :, 0].permute(0, 2, 1)  # shape = (batch_size, num_nodes, input_length)
        x = x.reshape(-1, x.shape[2])  # 将x维度变为 [batch*node_num, seq_len] 以适应模型的输入维度
        outputs = self.model(x, self.edge_index, self.dtw_edge_index)
        outputs = torch.reshape(outputs, (self.batch_size, self.output_window, self.num_nodes, 1))
        return outputs

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true)

    def predict(self, batch):
        return self.forward(batch)


class STAGGCNModel(nn.Module):
    def __init__(self, node_num=325, seq_len=12, pred_len=6, graph_dim=32,
                 tcn_dim=[10], atten_head=4, choice=[1, 1, 1]):
        super(STAGGCNModel, self).__init__()
        self.node_num = node_num
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.graph_dim = graph_dim
        # self.output_dim = seq_len + np.sum(choice) * graph_dim
        self.output_dim = np.sum(choice) * graph_dim
        self.STCell = STCell(node_num, seq_len, graph_dim, tcn_dim, choice=choice, atten_head=atten_head)
        self.output_linear = nn.Linear(in_features=self.output_dim, out_features=self.pred_len)
        # self.output_linear_0 = nn.Linear(in_features=self.graph_dim, out_features=256)
        # self.output_linear_1 = nn.Linear(in_features=256, out_features=self.pred_len)

    def forward(self, x, edge_index, dtw_edge_index):
        # x shape is [batch*node_num, seq_len]
        # st_output shape is [batch*node_num, 3*graph_dim]
        st_output = self.STCell(x, edge_index, dtw_edge_index)
        output = st_output

        output = self.output_linear(output)
        # output = F.relu(self.output_linear_0(output))
        # output = self.output_linear_1(output)
        # output = torch.reshape(output, (-1, self.node_num, self.pred_len))
        return output


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
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
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


class STCell(nn.Module):
    def __init__(self, node_num=524, seq_len=12, graph_dim=16, tcn_dim=[10], choice=[1, 1, 1], atten_head=2):
        super(STCell, self).__init__()
        self.node_num = node_num
        self.seq_len = seq_len
        self.graph_dim = graph_dim
        self.tcn_dim = tcn_dim
        self.output_dim = np.sum(choice) * graph_dim
        self.choice = choice
        # self.jklayer = JumpingKnowledge("max")
        # self.jklayer = JumpingKnowledge("lstm", self.graph_dim, 1)
        self.seq_linear = nn.Linear(in_features=self.seq_len, out_features=self.seq_len)

        if choice[0] == 1:
            print(f"[TCN]")
            print("node_num:", node_num, "\tatten_head:", atten_head)
            self.self_atten = nn.MultiheadAttention(embed_dim=node_num, num_heads=atten_head)
            self.tcn = TemporalConvNet(num_inputs=1, num_channels=self.tcn_dim)
            self.tlinear = nn.Linear(in_features=self.tcn_dim[-1] * self.seq_len, out_features=self.graph_dim)

        if choice[1] == 1:
            print(f"[SP]")
            self.sp_origin = nn.Linear(in_features=seq_len, out_features=graph_dim)
            self.sp_gconv1 = GATConv(seq_len, graph_dim, heads=3, concat=False)
            self.sp_gconv2 = GATConv(graph_dim, graph_dim, heads=3, concat=False)
            self.sp_gconv3 = GATConv(graph_dim, graph_dim, heads=3, concat=False)
            self.sp_gconv4 = GATConv(graph_dim, graph_dim, heads=1, concat=False)
            # self.sp_gconv5 = GATConv(graph_dim, graph_dim, heads = 1, concat = False)
            self.sp_source_embed = nn.Parameter(torch.Tensor(self.node_num, 12))
            self.sp_target_embed = nn.Parameter(torch.Tensor(12, self.node_num))
            self.sp_linear_1 = nn.Linear(self.seq_len, self.graph_dim)
            self.sp_linear_2 = nn.Linear(self.graph_dim, self.graph_dim)
            self.sp_linear_3 = nn.Linear(self.graph_dim, self.graph_dim)
            self.sp_linear_4 = nn.Linear(self.graph_dim, self.graph_dim)
            # self.sp_linear_5 = nn.Linear(self.graph_dim, self.graph_dim)
            # self.sp_jklayer = JumpingKnowledge("max")

            nn.init.xavier_uniform_(self.sp_source_embed)
            nn.init.xavier_uniform_(self.sp_target_embed)

        if choice[2] == 1:
            print(f"[DTW]")
            self.dtw_origin = nn.Linear(in_features=seq_len, out_features=graph_dim)
            self.dtw_gconv1 = GATConv(seq_len, graph_dim, heads=3, concat=False)
            self.dtw_gconv2 = GATConv(graph_dim, graph_dim, heads=3, concat=False)
            self.dtw_gconv3 = GATConv(graph_dim, graph_dim, heads=3, concat=False)
            self.dtw_gconv4 = GATConv(graph_dim, graph_dim, heads=3, concat=False)
            # self.dtw_gconv5 = GATConv(graph_dim, graph_dim, heads = 1, concat = False)
            self.dtw_source_embed = nn.Parameter(torch.Tensor(self.node_num, 12))
            self.dtw_target_embed = nn.Parameter(torch.Tensor(12, self.node_num))
            self.dtw_linear_1 = nn.Linear(self.seq_len, self.graph_dim)
            self.dtw_linear_2 = nn.Linear(self.graph_dim, self.graph_dim)
            self.dtw_linear_3 = nn.Linear(self.graph_dim, self.graph_dim)
            self.dtw_linear_4 = nn.Linear(self.graph_dim, self.graph_dim)
            # self.dtw_linear_5 = nn.Linear(self.graph_dim, self.graph_dim)
            # self.dtw_jklayer = JumpingKnowledge("max")

            nn.init.xavier_uniform_(self.dtw_source_embed)
            nn.init.xavier_uniform_(self.dtw_target_embed)

    def forward(self, x, edge_index, dtw_edge_index):
        # x shape is [batch*node_num, seq_len]
        # tcn/dtw/sp/adaptive output shape is [batch, node_num, graph_dim]
        output_list = [0, 0, 0]

        if self.choice[0] == 1:
            atten_input = torch.reshape(x, (-1, self.node_num, self.seq_len)).permute(2, 0, 1)
            atten_output, _ = self.self_atten(atten_input, atten_input, atten_input)
            atten_output = torch.tanh(atten_output + atten_input)
            atten_output = torch.reshape(atten_output.permute(1, 2, 0), (-1, self.seq_len))

            tcn_input = atten_output.unsqueeze(1)
            tcn_output = self.tcn(tcn_input)
            tcn_output = torch.reshape(tcn_output, (tcn_output.shape[0], self.tcn_dim[-1] * self.seq_len))
            tcn_output = self.tlinear(tcn_output)
            tcn_output = torch.reshape(tcn_output, (-1, self.node_num, self.graph_dim))
            output_list[0] = tcn_output

        if self.choice[1] == 1:
            x = self.seq_linear(x) + x

            sp_learned_matrix = F.softmax(F.relu(torch.mm(self.sp_source_embed, self.sp_target_embed)), dim=1)

            sp_gout_1 = self.sp_gconv1(x, edge_index)
            adp_input_1 = torch.reshape(x, (-1, self.node_num, self.seq_len))
            sp_adp_1 = self.sp_linear_1(sp_learned_matrix.matmul(F.dropout(adp_input_1, p=0.1)))
            sp_adp_1 = torch.reshape(sp_adp_1, (-1, self.graph_dim))
            sp_origin = self.sp_origin(x)
            sp_output_1 = torch.tanh(sp_gout_1) * torch.sigmoid(sp_adp_1) + sp_origin * (1 - torch.sigmoid(sp_adp_1))

            sp_gout_2 = self.sp_gconv2(torch.tanh(sp_output_1), edge_index)
            adp_input_2 = torch.reshape(torch.tanh(sp_output_1), (-1, self.node_num, self.graph_dim))
            sp_adp_2 = self.sp_linear_2(sp_learned_matrix.matmul(F.dropout(adp_input_2, p=0.1)))
            sp_adp_2 = torch.reshape(sp_adp_2, (-1, self.graph_dim))
            sp_output_2 = F.leaky_relu(sp_gout_2) * torch.sigmoid(sp_adp_2) \
                + sp_output_1 * (1 - torch.sigmoid(sp_adp_2))

            sp_gout_3 = self.sp_gconv3(F.relu(sp_output_2), edge_index)
            adp_input_3 = torch.reshape(F.relu(sp_output_2), (-1, self.node_num, self.graph_dim))
            sp_adp_3 = self.sp_linear_3(sp_learned_matrix.matmul(F.dropout(adp_input_3, p=0.1)))
            sp_adp_3 = torch.reshape(sp_adp_3, (-1, self.graph_dim))
            sp_output_3 = F.relu(sp_gout_3) * torch.sigmoid(sp_adp_3) + sp_output_2 * (1 - torch.sigmoid(sp_adp_3))

            sp_gout_4 = self.sp_gconv4(F.relu(sp_output_3), edge_index)
            adp_input_4 = torch.reshape(F.relu(sp_output_3), (-1, self.node_num, self.graph_dim))
            sp_adp_4 = self.sp_linear_4(sp_learned_matrix.matmul(F.dropout(adp_input_4, p=0.1)))
            sp_adp_4 = torch.reshape(sp_adp_4, (-1, self.graph_dim))
            sp_output_4 = F.relu(sp_gout_4) * torch.sigmoid(sp_adp_4) + sp_output_3 * (1 - torch.sigmoid(sp_adp_4))

            # sp_gout_5 = self.sp_gconv5(F.relu(sp_output_4), edge_index)
            # adp_input_5 = torch.reshape(F.relu(sp_output_4), (-1, self.node_num, self.graph_dim))
            # sp_adp_5 = self.sp_linear_5(sp_learned_matrix.matmul(F.dropout(adp_input_5,p=0.1)))
            # sp_adp_5 = torch.reshape(sp_adp_5, (-1, self.graph_dim))
            # sp_output_5 = F.relu(sp_gout_5) * torch.sigmoid(sp_adp_5) + sp_output_4 * (1 - torch.sigmoid(sp_adp_5))

            sp_output = torch.reshape(sp_output_4, (-1, self.node_num, self.graph_dim))
            # sp_output = sp_output_4
            output_list[1] = sp_output

        if self.choice[2] == 1:
            x = self.seq_linear(x) + x

            dtw_learned_matrix = F.softmax(F.relu(torch.mm(self.dtw_source_embed, self.dtw_target_embed)), dim=1)

            dtw_gout_1 = self.dtw_gconv1(x, dtw_edge_index)
            adp_input_1 = torch.reshape(x, (-1, self.node_num, self.seq_len))
            dtw_adp_1 = self.dtw_linear_1(dtw_learned_matrix.matmul(F.dropout(adp_input_1, p=0.1)))
            dtw_adp_1 = torch.reshape(dtw_adp_1, (-1, self.graph_dim))
            dtw_origin = self.dtw_origin(x)
            dtw_output_1 = torch.tanh(dtw_gout_1) * torch.sigmoid(dtw_adp_1) + \
                dtw_origin * (1 - torch.sigmoid(dtw_adp_1))

            dtw_gout_2 = self.dtw_gconv2(torch.tanh(dtw_output_1), dtw_edge_index)
            adp_input_2 = torch.reshape(torch.tanh(dtw_output_1), (-1, self.node_num, self.graph_dim))
            dtw_adp_2 = self.dtw_linear_2(dtw_learned_matrix.matmul(F.dropout(adp_input_2, p=0.1)))
            dtw_adp_2 = torch.reshape(dtw_adp_2, (-1, self.graph_dim))
            dtw_output_2 = F.leaky_relu(dtw_gout_2) * torch.sigmoid(dtw_adp_2) + \
                dtw_output_1 * (1 - torch.sigmoid(dtw_adp_2))

            dtw_gout_3 = self.dtw_gconv3(F.relu(dtw_output_2), dtw_edge_index)
            adp_input_3 = torch.reshape(F.relu(dtw_output_2), (-1, self.node_num, self.graph_dim))
            dtw_adp_3 = self.dtw_linear_3(dtw_learned_matrix.matmul(F.dropout(adp_input_3, p=0.1)))
            dtw_adp_3 = torch.reshape(dtw_adp_3, (-1, self.graph_dim))
            dtw_output_3 = F.relu(dtw_gout_3) * torch.sigmoid(dtw_adp_3) + dtw_output_2 * (1 - torch.sigmoid(dtw_adp_3))

            dtw_gout_4 = self.dtw_gconv4(F.relu(dtw_output_3), dtw_edge_index)
            adp_input_4 = torch.reshape(F.relu(dtw_output_3), (-1, self.node_num, self.graph_dim))
            dtw_adp_4 = self.dtw_linear_4(dtw_learned_matrix.matmul(F.dropout(adp_input_4, p=0.1)))
            dtw_adp_4 = torch.reshape(dtw_adp_4, (-1, self.graph_dim))
            dtw_output_4 = F.relu(dtw_gout_4) * torch.sigmoid(dtw_adp_4) + dtw_output_3 * (1 - torch.sigmoid(dtw_adp_4))

            # dtw_gout_5 = self.dtw_gconv5(F.relu(dtw_output_4), dtw_edge_index)
            # adp_input_5 = torch.reshape(F.relu(dtw_output_4), (-1, self.node_num, self.graph_dim))
            # dtw_adp_5 = self.dtw_linear_5(dtw_learned_matrix.matmul(F.dropout(adp_input_5,p=0.1)))
            # dtw_adp_5 = torch.reshape(dtw_adp_5, (-1, self.graph_dim))
            # dtw_output_5 = \
            # F.relu(dtw_gout_5) * torch.sigmoid(dtw_adp_5) + dtw_output_4 * (1 - torch.sigmoid(dtw_adp_5))

            dtw_output = torch.reshape(dtw_output_4, (-1, self.node_num, self.graph_dim))
            # dtw_output = dtw_output_4
            output_list[2] = dtw_output

        step = 0
        for i in range(len(self.choice)):
            if self.choice[i] == 1 and step == 0:
                cell_output = output_list[i]
                step += 1
            elif self.choice[i] == 1:
                cell_output = torch.cat((cell_output, output_list[i]), dim=2)

        # cell_output = self.jklayer([output_list[0], output_list[1], output_list[2]])
        # cell_output = self.out(cell_output)

        cell_output = torch.reshape(cell_output, (-1, self.output_dim))

        return cell_output
