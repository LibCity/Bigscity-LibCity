import torch
import torch.nn as nn
import torch.nn.functional as F
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import numpy as np
from libcity.model import loss
import scipy.sparse as sp

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # adj_.dot(degree_mat_inv_sqrt)得到 SD^{-0.5}
    # adj_.dot(degree_mat_inv_sqrt).transpose()得到(D^{-0.5})^{T}S^{T}=D^{-0.5}S，因为D和S都是对称矩阵
    # adj_normalized即为D^{-0.5}SD^{-0.5}
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)  #取出稀疏矩阵的上三角部分的非零元素，返回的是coo_matrix类型
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    # 取除去节点自环的所有边（注意，由于adj_tuple仅包含原始邻接矩阵上三角的边，所以edges中的边虽
    # 然只记录了边<src,dis>，而不冗余记录边<dis,src>），shape=(边数,2)每一行记录一条边的起始节点和终点节点的编号
    edges_all = sparse_to_tuple(adj)[0]
    # 取原始graph中的所有边，shape=(边数,2)每一行记录一条边的起始节点和终点节点的编号
    num_test = int(np.floor(edges.shape[0] / 10.))#划分测试集
    num_val = int(np.floor(edges.shape[0] / 20.))#划分验证集

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]#划分验证集
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]#划分测试集
    test_edges = edges[test_edge_idx]
    #edges是除去节点自环的所有边（因为数据集中的边都是无向的，edges只是存储了<src,dis>,
    #没有存储<dis,src>，因为没必要浪费内存），shape=(边数,2)每一行记录一条边的起始节点和终点节点的编号
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    # np.vstack():在竖直方向上堆叠，np.hstack():在水平方向上平铺。
    # np.hstack([test_edge_idx, val_edge_idx])将两个list水平方向拼接成一维数组
    # np.delete的参数axis=0，表示删除多行，删除的行号由第一个参数确定

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        # np.round返回浮点数x的四舍五入值，第二参数是保留的小数的位数
        # b[:, None]使b从shape=(边数,2)变为shape=(边数,1,2)，而a是长度为2的list，a - b[:, None]触发numpy的广播机制
        # np.all()判断给定轴向上的所有元素是否都为True，axis=-1（此时等同于axis=2）表示3维数组最里层的2维数组的每一行的元素是否都为True
        return np.any(rows_close)
        # np.any()判断给定轴向上是否有一个元素为True,现在不设置axis参数则是判断所有元素中是否有一个True，有一个就返回True。
        # rows_close的shape=(边数,1)
        # 至此，可以知道，ismember( )方法用于判断随机生成的<a,b>这条边是否是已经真实存在的边，如果是，则返回True，否则返回False

    # test_edges_false = []
    # while len(test_edges_false) < len(test_edges):
    #     idx_i = np.random.randint(0, adj.shape[0])  #生成负样本
    #     idx_j = np.random.randint(0, adj.shape[0])
    #     if idx_i == idx_j:
    #         continue
    #     if ismember([idx_i, idx_j], edges_all):
    #         continue
    #     if test_edges_false:
    #         if ismember([idx_j, idx_i], np.array(test_edges_false)):
    #             continue
    #         if ismember([idx_i, idx_j], np.array(test_edges_false)):
    #             continue
    #     test_edges_false.append([idx_i, idx_j])

    # val_edges_false = []
    # while len(val_edges_false) < len(val_edges):
    #     idx_i = np.random.randint(0, adj.shape[0])
    #     idx_j = np.random.randint(0, adj.shape[0])
    #     if idx_i == idx_j:
    #         continue
    #     if ismember([idx_i, idx_j], train_edges):
    #         continue
    #     if ismember([idx_j, idx_i], train_edges):
    #         continue
    #     if ismember([idx_i, idx_j], val_edges):
    #         continue
    #     if ismember([idx_j, idx_i], val_edges):
    #         continue
    #     if val_edges_false:
    #         if ismember([idx_j, idx_i], np.array(val_edges_false)):
    #             continue
    #         if ismember([idx_i, idx_j], np.array(val_edges_false)):
    #             continue
    #     val_edges_false.append([idx_i, idx_j])

    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train
    #return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight) #初始化服从均匀分布

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)  #随机丢弃数据，防止过拟合
        support = torch.mm(input, self.weight)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        output = torch.spmm(adj, support)
        output = self.act(output) #relu激活
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    print("in loss_function")
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=torch.tensor(pos_weight))
    print(cost)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #cost = 0.0
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

class VAE(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.input_feat_dim = data_feature.get('feature_dim')
        self.hidden_dim1 = config.get('hidden1')
        self.hidden_dim2 = config.get('hidden2')
        self.adj = data_feature.get('adj_mx')
        self.device = config.get('device', torch.device('cpu'))
        self.num_nodes = data_feature.get('num_nodes')
        self.dropout = config.get('dropout')
        self.lr = config.get('lr')
        self.gc1 = GraphConvolution(self.input_feat_dim, self.hidden_dim1, self.dropout, act=F.relu)
        self.gc2 = GraphConvolution(self.hidden_dim1, self.hidden_dim2, self.dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(self.hidden_dim1, self.hidden_dim2, self.dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(self.dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)  #生成与std形状相同的随机张量
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self,batch):
        adj = self.adj
        x = batch['node_features']
        self.mu, self.logvar = self.encode(x, adj)
        z = self.reparameterize(self.mu, self.logvar)
        return self.dc(z), self.mu, self.logvar


    def predict(self,batch):
        x =batch['node_features']
        adj = self.adj
        return self.forward(batch)

    def calculate_loss(self, batch):
        """

        Args:
            batch: dict, need key 'node_features', 'node_labels', 'mask'

        Returns:

        """
        preds,mu,logvar = self.predict(batch)  # N, feature_dim
        print(preds.shape)
        # mu,logvar = self.mu,self.logvar
        n_nodes = self.num_nodes
        adj = self.adj
        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)#邻接矩阵对角线置0
        # Scipy的dia_matrix函数见1.其中offsets数组中0表示对角线，-1表示对角线下面，正数表示对角线上面
        # np.newaxis的作用是增加一个维度。[np.newaxis，：]是在np.newaxis这里增加1维。这样改变维度的作用往往是将一维的数据转变成一个矩阵
        #diagonal()是获得矩阵对角线
        #adj_orig.diagonal()[np.newaxis, :], [0]代码意思是先将对角线提取出来然后增加一维变为矩阵，方便后续计算
        adj_orig.eliminate_zeros()#将值为0的元素删除

        print("before mask")
        adj_train= mask_test_edges(adj)
        print("after mask")
        adj = adj_train #用于训练的邻接矩阵，类型为csr_matrix

        # Some preprocessing
        # adj_norm = preprocess_graph(adj) #返回D^{-0.5}SD^{-0.5}的coords(坐标), data, shape，其中S=A+I
        adj_label = adj_train + sp.eye(adj_train.shape[0]) #adj_train是用于训练的邻接矩阵，类型为csr_matrix sp.eye用于对角线置1
        # adj_label = sparse_to_tuple(adj_label)
        adj_label = torch.FloatTensor(adj_label.toarray())  
        print(adj_label.shape)
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        # labels =sparse_mx_to_torch_sparse_tensor(self.adj)  # N, feature_dim
        print("before loss_function")
        return loss_function(preds, adj_label, mu, logvar, n_nodes, norm, pos_weight)


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
