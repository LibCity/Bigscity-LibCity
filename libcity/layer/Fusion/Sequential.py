import torch
import torch.nn as nn

from libcity.layer.Spatial.atten import SpatialAttention
from libcity.layer.Spatial.gnn import spectralGCN
from libcity.layer.Temporal.atten import TemporalAttention, TemporalAttentionLayer
from libcity.layer.Temporal.cnn import TemporalConvLayer
from libcity.layer.utils import calculate_scaled_laplacian, calculate_cheb_poly, calculate_first_approx, GatedFusion

"""
    顺序的对spatial和temporal两个维度建模，交替对两个维度进行encode
    Examples ASTGCN、HGCN、ST-GCN、MTGNN
"""


class ST2Block(nn.Module):
    """
        similar to ASTGCN
        ST-block ：
        spatial gcn ->
        temporal conv/gate
    """

    def __init__(self, ks, kt, n, c, p, lk,timestep, device, residual=False):
        super(ST2Block, self).__init__()
        self.tconv = TemporalConvLayer(kt, c[0], c[1], "GLU")
        self.sconv = spectralGCN(K=ks, input_size=c[1], output_size=c[1], adj_mx=lk, device=device)
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

        self.residual = residual
        if self.residual:
            self.residual_conv = nn.Conv2d(c[0], c[2], kernel_size=(1, 1), stride=(1, timestep))

    def forward(self, x):  # x: (batch_size, feature_dim/c[0], input_length, num_nodes)
        x = self.sconv(x)
        x = self.tconv(x)

        # residual 模块
        if self.residual:
            x_res = self.residual_conv(x)
            x += x_res

        return self.dropout(x)


class ST3Block_impl1(nn.Module):
    """
        similar to ST-GCN
        ST-block ->
        temporal gate ->
        spatial gcn ->
        temporal gate
    """

    def __init__(self, ks, kt, n, c, p, lk, timestep,device, residual=False):
        super(ST3Block_impl1, self).__init__()
        self.tconv1 = TemporalConvLayer(kt, c[0], c[1], "GLU")
        self.sconv = spectralGCN(K=ks, input_size=c[1], output_size=c[1], adj_mx=lk, device=device)
        self.tconv2 = TemporalConvLayer(kt=kt, c_in=c[1], c_out=c[2])
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

        self.residual = residual
        if self.residual:
            self.residual_conv = nn.Conv2d(c[0], c[2], kernel_size=(1, 1), stride=(1, timestep))

    def forward(self, x):  # x: (batch_size, feature_dim/c[0], input_length, num_nodes)
        x_t1 = self.tconv1(x)  # (batch_size, c[1], input_length-kt+1, num_nodes)
        x_s = self.sconv(x_t1)  # (batch_size, c[1], input_length-kt+1, num_nodes)
        x_t2 = self.tconv2(x_s)  # (batch_size, c[2], input_length-kt+1-kt+1, num_nodes)

        # residual 模块
        if self.residual:
            x_res = self.residual_conv(x)
            x_t2 += x_res

        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)


class ST3Block_impl2(nn.Module):
    """
        similar to HGCN
        ST-block ->
        temporal gate ->
        spatial gcn ->
        temporal attn ->
    """

    def __init__(self, ks, kt, n, c, p, lk, timestep,device, residual=False):
        super(ST3Block_impl2, self).__init__()
        self.tconv1 = TemporalConvLayer(kt=kt, c_in=c[0], c_out=c[1], act="GLU")
        self.sconv = spectralGCN(K=ks, input_size=c[1], output_size=c[1], adj_mx=lk, device=device)
        self.tconv2 = TemporalAttentionLayer(num_of_timesteps=timestep,num_of_vertices=c[2],device=device,in_channels=c[1])
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

        self.residual = residual
        if self.residual:
            self.residual_conv = nn.Conv2d(c[0], c[2], kernel_size=(1, 1), stride=(1, timestep))

    def forward(self, x):  # x: (batch_size, feature_dim/c[0], input_length, num_nodes)
        x_t1 = self.tconv1(x)  # (batch_size, c[1], input_length-kt+1, num_nodes)
        x_s = self.sconv(x_t1)  # (batch_size, c[1], input_length-kt+1, num_nodes)
        x_t2 = self.tconv2(x_s)  # (batch_size, c[2], input_length-kt+1-kt+1, num_nodes)

        # residual 模块
        if self.residual:
            x_res = self.residual_conv(x)
            x_t2 += x_res

        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)


"""
    对spatial和temporal分别建模的
    Examples STAGGCN、GMAN
"""


class STParBlock(nn.Module):
    def __init__(self, K, D, bn, bn_decay, device, mask=True):
        super(STParBlock, self).__init__()
        self.K = K
        self.D = D
        self.d = self.D / self.K
        self.bn = bn
        self.bn_decay = bn_decay
        self.device = device
        self.mask = mask
        self.sp_att = SpatialAttention(num_heads=self.K, dim=self.D, bn=self.bn, bn_decay=self.bn_decay, device=self.device)
        self.temp_att = TemporalAttention(num_heads=self.K, input_size=self.D, bn=self.bn, bn_decay=self.bn_decay,
                                          device=self.device)
        self.gated_fusion = GatedFusion(dim=self.D, bn=self.bn, bn_decay=self.bn_decay, device=self.device)  # outputLayer

    def forward(self, x, ste):
        HS = self.sp_att(x, ste)
        HT = self.temp_att(x, ste)
        H = self.gated_fusion(HS, HT)
        return torch.add(x, H)


"""
    对spatial和temporal同时建模
    Examples stsgcn、Seq2
"""


class STSycBlock(nn.Module):
    # 同步时空卷积模块，捕获连续 3 个时间片的时空特征
    def __init__(self, filters, num_of_features, num_of_vertices, adj, activation):
        super().__init__()
        self.filters = filters
        self.num_of_features = num_of_features
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        self.layers = nn.ModuleList()
        for i in range(len(filters)):
            self.layers.append(spectralGCN(filters[i], num_of_features, num_of_vertices, adj, activation))
            num_of_features = filters[i]

    def forward(self, data):
        # 多个卷积层叠加，
        need_concat = []
        for i in range(len(self.layers)):
            data = self.layers[i](data)
            need_concat.append(torch.transpose(data, 1, 0))

        # 且每个卷积层的输出都以类似残差网络的形式输入聚合层
        need_concat = [
            torch.unsqueeze(
                i[self.num_of_vertices:2 * self.num_of_vertices, :, :],
                dim=0
            ) for i in need_concat
        ]

        # 聚合使用最大池化
        return torch.max(torch.cat(need_concat, dim=0), dim=0)[0]


class OutputLayer(nn.Module):
    def __init__(self, c, t, n, out_dim):
        super(OutputLayer, self).__init__()
        self.tconv1 = TemporalConvLayer(t, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = TemporalConvLayer(1, c, c, "sigmoid")  # kernel=1*1
        self.fc = FullyConvLayer(c, out_dim)

    def forward(self, x):
        # (batch_size, input_dim(c), T, num_nodes)
        x_t1 = self.tconv1(x)
        # (batch_size, input_dim(c), 1, num_nodes)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # (batch_size, input_dim(c), 1, num_nodes)
        x_t2 = self.tconv2(x_ln)
        # (batch_size, input_dim(c), 1, num_nodes)
        return self.fc(x_t2)
        # (batch_size, output_dim, 1, num_nodes)

class STsequential(nn.Module):
    def __init__(self, config):
        super(STsequential, self).__init__()
        self.config = config

        self.num_nodes = self.config.get('num_nodes', 1)
        self.feature_dim = self.config.get('feature_dim', 1)
        self.output_dim = self.config.get('output_dim', 1)

        self.Ks = config.get('Ks', 3)
        self.Kt = config.get('Kt', 3)
        self.blocks = config.get('blocks', [[1, 32, 64], [64, 32, 128]])
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.drop_prob = config.get('dropout', 0)

        self.graph_conv_type = config.get('graph_conv_type', 'chebconv')
        adj_mx = config.get('adj_mx')

        if self.graph_conv_type.lower() == 'chebconv':
            laplacian_mx = calculate_scaled_laplacian(adj_mx)
            self.Lk = calculate_cheb_poly(laplacian_mx, self.Ks)
            self._logger.info('Chebyshev_polynomial_Lk shape: ' + str(self.Lk.shape))
            self.Lk = torch.FloatTensor(self.Lk).to(self.device)
        elif self.graph_conv_type.lower() == 'gcnconv':
            self.Lk = calculate_first_approx(adj_mx)
            self._logger.info('First_approximation_Lk shape: ' + str(self.Lk.shape))
            self.Lk = torch.FloatTensor(self.Lk).to(self.device)
            self.Ks = 1  # 一阶近似保留到K0和K1，但是不是数组形式，只有一个n*n矩阵，所以是1（本质上是2）

        self.num_blocks = len(self.blocks)
        self.st_blocks = nn.ModuleList()

        for i in range(self.num_blocks):
            self.st_blocks.append(ST3Block_impl1(self.Ks, self.Kt, self.num_nodes,
                                                 self.blocks[i], self.drop_prob, self.Lk, self.device))

        self.output = OutputLayer(self.blocks[-1][2], self.input_window - len(self.blocks) * 2
                                  * (self.Kt - 1), self.num_nodes, self.output_dim)

    def forward(self, x, return_hidden=False):
        # return hidden 控制是否输出中间层结果
        # (batch_size, input_length, num_nodes, feature_dim)
        x = x.permute(0, 3, 1, 2)  # (batch_size, feature_dim, input_length, num_nodes)
        # x_st1 = self.st_conv1(x)  # (batch_size, c[2](64), input_length-kt+1-kt+1, num_nodes)
        # x_st2 = self.st_conv2(x_st1)  # (batch_size, c[2](128), input_length-kt+1-kt+1-kt+1-kt+1, num_nodes)

        if return_hidden:
            outputs = []
            for i in range(self.num_blocks):
                x = self.st_blocks[i](x)
                outputs.append(x)
            outputs = torch.cat(outputs, dim=1)
            return outputs
        else:
            for i in range(self.num_blocks):
                x = self.st_blocks[i](x)
            outputs = self.output(x)  # (batch_size, output_dim(1), output_length(1), num_nodes)
            outputs = outputs.permute(0, 2, 3, 1)  # (batch_size, output_length(1), num_nodes, output_dim)
            return outputs


