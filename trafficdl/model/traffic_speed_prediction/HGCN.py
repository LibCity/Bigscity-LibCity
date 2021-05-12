from logging import getLogger
import torch
from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Conv2d, Parameter, BatchNorm1d


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        A = A.transpose(-1, -2)
        x = torch.einsum('ncvl,vw->ncwl', x, A)
        return x.contiguous()


class multi_gcn_time(nn.Module):
    def __init__(self, c_in, c_out, Kt, dropout, support_len=3, order=2):
        super(multi_gcn_time, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear_time(c_in, c_out, Kt)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        count = 0
        for a in support:
            count += 1
            a = a.to(x.device)
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class TATT_1(nn.Module):
    '''#时间注意力机制'''

    def __init__(self, c_in, num_nodes, tem_size):
        super(TATT_1, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)

        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn = BatchNorm1d(tem_size)
        self.c_in = c_in
        self.tem_size = tem_size

    def forward(self, seq):
        # print('seq.shape:', seq.shape)  # [64, 32, 10, 8] [64, 32, 10, 6]
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()  # b,l,n

        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        # print(c2.shape)
        f2 = self.conv2(c2).squeeze()  # b,c,n
        # print('f1.shape:', f1.shape)
        # print('self.w.shape:', self.w.shape)
        # print('torch.matmul(f1, self.w):', torch.matmul(f1, self.w).shape)
        # print('f2.shape:', f2.shape)
        # print('torch.matmul(torch.matmul(f1, self.w), f2):', torch.matmul(torch.matmul(f1, self.w), f2).shape)
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        logits = logits.permute(0, 2, 1).contiguous()
        logits = self.bn(logits).permute(0, 2, 1).contiguous()
        coefs = torch.softmax(logits, -1)
        return coefs


class linear_time(nn.Module):
    '''#线性层'''

    def __init__(self, c_in, c_out, Kt):
        super(linear_time, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class GCNPool(nn.Module):
    ''' #GCN      S-T Blocks'''

    def __init__(self, c_in, c_out, num_nodes, tem_size,
                 Kt, dropout, pool_nodes, support_len=3, order=2):
        super(GCNPool, self).__init__()
        self.time_conv = Conv2d(c_in, 2 * c_out, kernel_size=(1, Kt), padding=(0, 0),
                                stride=(1, 1), bias=True, dilation=2)

        self.multigcn = multi_gcn_time(c_out, 2 * c_out, Kt, dropout, support_len, order)

        self.num_nodes = num_nodes
        self.tem_size = tem_size
        self.TAT = TATT_1(c_out, num_nodes, tem_size)
        self.c_out = c_out
        # self.bn=LayerNorm([c_out,num_nodes,tem_size])
        self.bn = BatchNorm2d(c_out)

        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)

    def forward(self, x, support):
        # print('GCNPool x.shape:', x.shape)  # [64, 32, 10, 14] [64, 32, 10, 12] (,,,input_window)
        residual = self.conv1(x)
        # print('GCNPool residual.shape:', residual.shape)
        x = self.time_conv(x)
        # print('GCNPool x.shape:', x.shape)  # [64, 64, 10, 10] [64, 64, 10, 8](,,,input_window-kt*diadilation+2)
        x1, x2 = torch.split(x, [self.c_out, self.c_out], 1)
        x = torch.tanh(x1) * torch.sigmoid(x2)

        x = self.multigcn(x, support)
        # print('after multigcn:', x.shape) # [64, 64, 10, 8] [64, 64, 10, 6]
        x1, x2 = torch.split(x, [self.c_out, self.c_out], 1)
        x = torch.tanh(x1) * (torch.sigmoid(x2))
        # x=F.dropout(x,0.3,self.training)

        # print('TAT input x.shape:', x.shape)  # [64, 32, 10, 8] [64, 32, 10, 6]
        T_coef = self.TAT(x)
        T_coef = T_coef.transpose(-1, -2)
        x = torch.einsum('bcnl,blq->bcnq', x, T_coef)
        out = self.bn(x + residual[:, :, :, -x.size(3):])
        return out


class Transmit(nn.Module):
    '''#Transfer Blocks  交换层'''

    def __init__(self, c_in, tem_size, transmit, num_nodes, cluster_nodes):
        super(Transmit, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(tem_size, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(tem_size, c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(num_nodes, cluster_nodes), requires_grad=True)
        self.c_in = c_in
        self.transmit = transmit
        self.tem_size = tem_size

    def forward(self, seq, seq_cluster):
        # print('self.c_in:', self.c_in)
        # print('seq_cluster.shape:', seq_cluster.shape)
        c1 = seq
        # print('c1.shape:', c1.shape)
        f1 = self.conv1(c1).squeeze(1)  # b,n,l

        c2 = seq_cluster.permute(0, 3, 1, 2)  # b,c,n,l->b,l,n,c
        f2 = self.conv2(c2).squeeze(1)  # b,c,n
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        a = torch.mean(logits, 1, True)
        logits = logits - a
        logits = torch.sigmoid(logits)

        coefs = (logits) * self.transmit
        return coefs


class gate(nn.Module):
    def __init__(self, c_in):
        super(gate, self).__init__()
        self.conv1 = Conv2d(c_in, c_in // 2, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)

    def forward(self, seq, seq_cluster):
        # x=torch.cat((seq_cluster,seq),1)
        # gate=torch.sigmoid(self.conv1(x))
        out = torch.cat((seq, (seq_cluster)), 1)

        return out


class HGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        """
        构造模型
        :param config: 源于各种配置的配置字典
        :param data_feature: 从数据集Dataset类的`get_data_feature()`接口返回的必要的数据相关的特征
        """
        # 1.初始化父类（必须）
        super().__init__(config, data_feature)
        # 2.初始化device（必须）
        self.device = config.get('device', torch.device('cpu'))
        # 3.从data_feature获取想要的信息，注意不同模型使用不同的Dataset类，其返回的data_feature内容不同（必须）
        self._scaler = self.data_feature.get('scaler')  # 用于数据归一化
        self.num_nodes = self.data_feature.get('num_nodes')  # 节点个数
        self.feature_dim = self.data_feature.get('feature_dim')  # 输入维度
        self.output_dim = self.data_feature.get('output_dim')  # 输出维度
        self.transmit = self.data_feature.get('transmit').to(self.device)  # trans矩阵
        self.adj_mx = self.data_feature.get('adj_mx')  # 邻接矩阵
        self.adj_mx_cluster = self.data_feature.get('adj_mx_cluster').to(self.device)  # region邻接矩阵
        self.centers_ind_groups = self.data_feature.get('centers_ind_groups')  # 聚类后   区域标号[节点标号]
        # 4.初始化log用于必要的输出（必须）
        self._logger = getLogger()
        # 5.初始化输入输出时间步的长度（非必须）
        self.input_window = config.get('input_window')
        self.output_window = config.get('output_window')
        # 6.从config中取用到的其他参数，主要是用于构造模型结构的参数（必须）
        self.cluster_nodes = config['cluster_nodes']  # 聚类节点（区域）个数
        self.dropout = config['dropout']
        self.channels = config['channels']
        self.skip_channels = config['skip_channels']
        self.end_channels = config['end_channels']
        self.kernel_size = config['kernel_size']  # TODO unused
        self.K = config['K']  # TODO unused
        self.Kt = config['Kt']  # TODO unused
        # TODO 以上三个参数若加入模型则会带来维度不匹配的问题，即不是独立的参数。现在的状态是硬编码，反而不会有问题。
        # 7.构造深度模型的层次结构（必须
        self.supports = [torch.tensor(self.adj_mx)]
        self.supports_cluster = [self.adj_mx_cluster.clone().detach()]
        self.supports_len = torch.tensor(0, device=self.device)
        self.supports_len_cluster = torch.tensor(0, device=self.device)

        self.supports_len += len(self.supports)
        self.supports_len_cluster += len(self.supports_cluster)

        self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
                                    out_channels=self.channels,
                                    kernel_size=(1, 1))
        self.start_conv_cluster = nn.Conv2d(in_channels=self.feature_dim,
                                            out_channels=self.channels,
                                            kernel_size=(1, 1))

        self.h = Parameter(torch.zeros(self.num_nodes, self.num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.h_cluster = Parameter(torch.zeros(self.cluster_nodes, self.cluster_nodes), requires_grad=True)
        nn.init.uniform_(self.h_cluster, a=0, b=0.0001)
        self.supports_len += 1
        self.supports_len_cluster += 1
        self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10, self.num_nodes), requires_grad=True)
        self.nodevec1_c = nn.Parameter(torch.randn(self.cluster_nodes, 10), requires_grad=True)
        self.nodevec2_c = nn.Parameter(torch.randn(10, self.cluster_nodes), requires_grad=True)

        self.block1 = GCNPool(2 * self.channels, self.channels, self.num_nodes, self.input_window - 6, 3,
                              self.dropout, self.num_nodes,
                              self.supports_len)
        self.block2 = GCNPool(2 * self.channels, self.channels, self.num_nodes, self.input_window - 9, 2,
                              self.dropout, self.num_nodes,
                              self.supports_len)

        self.block_cluster1 = GCNPool(
            c_in=self.channels,
            c_out=self.channels,
            num_nodes=self.cluster_nodes,
            tem_size=self.input_window-6,
            Kt=3,
            dropout=self.dropout,
            pool_nodes=self.cluster_nodes,
            support_len=self.supports_len
        )
        self.block_cluster2 = GCNPool(
            c_in=self.channels,
            c_out=self.channels,
            num_nodes=self.cluster_nodes,
            tem_size=self.input_window - 9,
            Kt=2,
            dropout=self.dropout,
            pool_nodes=self.cluster_nodes,
            support_len=self.supports_len
        )

        self.skip_conv1 = Conv2d(2 * self.channels, self.skip_channels, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)
        self.skip_conv2 = Conv2d(2 * self.channels, self.skip_channels, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 3),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.output_window,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.bn = BatchNorm2d(self.feature_dim, affine=False)
        # self.conv_cluster1 = Conv2d(self.dilation_channels, self.out_dim, kernel_size=(1, 3),
        #                             stride=(1, 1), bias=True)
        self.bn_cluster = BatchNorm2d(self.feature_dim, affine=False)
        self.gate1 = gate(2 * self.channels)
        self.gate2 = gate(2 * self.channels)
        self.gate3 = gate(2 * self.channels)

        self.transmit1 = Transmit(self.channels, self.input_window, self.transmit, self.num_nodes,
                                  self.cluster_nodes)
        self.transmit2 = Transmit(self.channels, self.input_window - 6, self.transmit, self.num_nodes,
                                  self.cluster_nodes)
        self.transmit3 = Transmit(self.channels, self.input_window - 9, self.transmit, self.num_nodes,
                                  self.cluster_nodes)

    def get_input_cluster(self, input):
        # 得到batch的shape
        batch_size, input_length, feature_dim = input.shape[0], input.shape[1], input.shape[
            3]
        # 初始化batch_cluster
        input_cluster = torch.zeros([batch_size, input_length, self.cluster_nodes, feature_dim], dtype=torch.float,
                                    device=self.device)
        # input_cluster = np.zeros((batch_size, input_length, self.cluster_nodes, feature_dim), dtype=np.float32)
        # 更新batch_cluster
        # for i in range(batch_size):
        #     for j in range(input_length):
        #         for k in range(self.cluster_nodes):
        #             for m in range(feature_dim):
        #                 input_cluster[i, j, k, m] = input[i, j, self.centers_ind_groups[k], m].mean() + input[
        #                     i, j, self.centers_ind_groups[k], m].min()
        for k in range(self.cluster_nodes):
            input_cluster[:, :, k, :] = input[:, :, self.centers_ind_groups[k][0], :] + \
                                        input[:, :, self.centers_ind_groups[k][0], :]
        return input_cluster

    def forward(self, batch):
        """
        调用模型计算这个batch输入对应的输出，nn.Module必须实现的接口
        :param batch: 输入数据，类字典，可以按字典的方法取数据
        :return:
        """
        # 1.取数据，假设字典中有4类数据，X,y,X_ext,y_ext
        # 当然一般只需要取输入数据，例如X,X_ext，因为这个函数是用来计算输出的
        # 模型输入的数据的特征维度应该等于self.feature_dim
        # x = batch['X']  # shape = (batch_size, input_length, ..., feature_dim)
        # 例如: y = batch['y'] / X_ext = batch['X_ext'] / y_ext = batch['y_ext']]
        # 2.根据输入数据计算模型的输出结果
        # 模型输出的结果的特征维度应该等于self.output_dim
        # 模型输出的结果的其他维度应该跟batch['y']一致，只有特征维度可能不同（因为batch['y']可能包含一些外部特征）
        # 如果模型的单步预测，batch['y']是多步的数据，则时间维度也有可能不同
        # 例如: outputs = self.model(x)
        # 3.返回输出结果
        # 例如: return outputs

        # 1.取数据
        input = batch['X'].permute(0, 3, 2, 1)
        # 2.计算聚类batch_cluster
        input_cluster = self.get_input_cluster(input)
        # 3模型计算
        x = self.bn(input)
        x_cluster = self.bn_cluster(input_cluster)

        # nodes
        A = F.relu(torch.mm(self.nodevec1, self.nodevec2))
        d = 1 / (torch.sum(A, -1))
        D = torch.diag_embed(d)
        A = torch.matmul(D, A)

        new_supports = self.supports + [A]
        # region
        A_cluster = F.relu(torch.mm(self.nodevec1_c, self.nodevec2_c))
        d_c = 1 / (torch.sum(A_cluster, -1))
        D_c = torch.diag_embed(d_c)
        A_cluster = torch.matmul(D_c, A_cluster)

        new_supports_cluster = self.supports_cluster + [A_cluster]

        # network
        x = self.start_conv(x)
        x_cluster = self.start_conv_cluster(x_cluster)
        # print('x.shape:', x.shape)  # [64, 32, 207, 12] [64, 32, 207, 14]
        # print('x_cluster.shape:', x_cluster.shape)  # [64, 32, 10, 12] [64, 32, 10, 14]
        transmit1 = self.transmit1(x, x_cluster)
        # print('transmit1.shape:', transmit1.shape)  # [64, 207, 10]
        x_1 = (torch.einsum('bmn,bcnl->bcml', transmit1, x_cluster))

        x = self.gate1(x, x_1)

        skip = torch.tensor(0, device=self.device)
        # 1
        # print('before block_cluster1:', x_cluster.shape)  # [64, 32, 10, 14] [64, 32, 10, 12]
        x_cluster = self.block_cluster1(x_cluster, new_supports_cluster)
        # print('after block_cluster1:', x_cluster.shape)  # [64, 32, 10, 4] [64, 32, 10, 6]
        x = self.block1(x, new_supports)
        transmit2 = self.transmit2(x, x_cluster)
        x_2 = (torch.einsum('bmn,bcnl->bcml', transmit2, x_cluster))

        x = self.gate2(x, x_2)

        s1 = self.skip_conv1(x)
        skip = s1 + skip

        # 2
        # print('before block_cluster2:', x_cluster.shape)
        x_cluster = self.block_cluster2(x_cluster, new_supports_cluster)
        # print('after block_cluster2:', x_cluster.shape)  # [64, 32, 10, 3]
        x = self.block2(x, new_supports)
        transmit3 = self.transmit3(x, x_cluster)
        x_3 = (torch.einsum('bmn,bcnl->bcml', transmit3, x_cluster))

        x = self.gate3(x, x_3)

        s2 = self.skip_conv2(x)
        skip = skip[:, :, :, -s2.size(3):]
        skip = s2 + skip

        # output
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
        # return x, transmit3, A

    def calculate_loss(self, batch):
        """
        输入一个batch的数据，返回训练过程这个batch数据的loss，也就是需要定义一个loss函数。
        :param batch: 输入数据，类字典，可以按字典的方法取数据
        :return: training loss (tensor)
        """
        # 1.取出真值 ground_truth
        y_true = batch['y'].to(self.device)

        # 2.取出预测值
        output = self.predict(batch)
        y_predicted = output
        # 3.使用self._scaler将进行了归一化的真值和预测值进行反向归一化（必须）
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        # 4.调用loss函数计算真值和预测值的误差
        # trafficdl/model/loss.py中定义了常见的loss函数
        # 如果模型源码用到了其中的loss，则可以直接调用，以MSE为例:
        # res = loss.masked_mse_torch(y_predicted, y_true).to(self.device)
        res = torch.mean(abs(y_predicted - y_true))
        # 如果模型源码所用的loss函数在loss.py中没有，则需要自己实现loss函数
        # ...（自定义loss函数）
        # 5.返回loss的结果
        return res

    def predict(self, batch):
        return self.forward(batch)
