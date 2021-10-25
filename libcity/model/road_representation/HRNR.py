import torch.nn as nn
import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphEncoderTL(Module):
    def __init__(self, hparams, struct_assign, fnc_assign, struct_adj):
        super(GraphEncoderTL, self).__init__()
        self.hparams = hparams
        self.struct_assign = struct_assign
        self.fnc_assign = fnc_assign
        self.struct_adj = struct_adj

        self.node_emb_layer = nn.Embedding(hparams.node_num, hparams.node_dims).to(self.hparams.device)
        self.type_emb_layer = nn.Embedding(hparams.type_num, hparams.type_dims).to(self.hparams.device)
        self.length_emb_layer = nn.Embedding(hparams.length_num, hparams.length_dims).to(self.hparams.device)
        self.lane_emb_layer = nn.Embedding(hparams.lane_num, hparams.lane_dims).to(self.hparams.device)

        self.tl_layer_1 = GraphEncoderTLCore(hparams, self.struct_assign, self.fnc_assign)
        self.tl_layer_2 = GraphEncoderTLCore(hparams, self.struct_assign, self.fnc_assign)
        self.tl_layer_3 = GraphEncoderTLCore(hparams, self.struct_assign, self.fnc_assign)

        self.init_feat = None

    def forward(self, node_feature, type_feature, length_feature, lane_feature, adj):
        node_emb = self.node_emb_layer(node_feature)
        type_emb = self.type_emb_layer(type_feature)
        length_emb = self.length_emb_layer(length_feature)
        lane_emb = self.lane_emb_layer(lane_feature)
        raw_feat = torch.cat([lane_emb, type_emb, length_emb, node_emb], 1)
        self.init_feat = raw_feat
        #    for i in range(self.hparams.label_pred_gnn_layer):
        #      raw_feat = self.tl_layer_1(self.struct_adj, raw_feat, adj)
        raw_feat = self.tl_layer_1(self.struct_adj, raw_feat, adj)
        raw_feat = self.tl_layer_2(self.struct_adj, raw_feat, adj)
        #    raw_feat = self.tl_layer_3(self.struct_adj, raw_feat, adj)

        return raw_feat


class GraphEncoderTLCore(Module):
    def __init__(self, hparams, struct_assign, fnc_assign):
        super(GraphEncoderTLCore, self).__init__()
        self.hparams = hparams
        self.struct_assign = struct_assign
        self.fnc_assign = fnc_assign

        self.fnc_gcn = GraphConvolution(
            in_features=self.hparams.hidden_dims,
            out_features=self.hparams.hidden_dims,
            device=self.hparams.device).to(self.hparams.device)

        self.struct_gcn = GraphConvolution(
            in_features=self.hparams.hidden_dims,
            out_features=self.hparams.hidden_dims,
            device=self.hparams.device).to(self.hparams.device)

        self.node_gat = SPGCN(
            in_features=self.hparams.hidden_dims,
            out_features=self.hparams.hidden_dims,
            device=self.hparams.device).to(self.hparams.device)

        self.l_c = torch.nn.Linear(hparams.hidden_dims * 2, 1).to(self.hparams.device)

        self.l_s = torch.nn.Linear(hparams.hidden_dims * 2, 1).to(self.hparams.device)

        self.sigmoid = nn.Sigmoid()

    def forward(self, struct_adj, raw_feat, raw_adj):
        # forward
        self.raw_struct_assign = self.struct_assign
        self.raw_fnc_assign = self.fnc_assign

        self.struct_assign = self.struct_assign / (F.relu(torch.sum(self.struct_assign, 0) - 1.0) + 1.0)
        self.fnc_assign = self.fnc_assign / (F.relu(torch.sum(self.fnc_assign, 0) - 1.0) + 1.0)

        self.struct_emb = torch.mm(self.struct_assign.t(), raw_feat)
        self.fnc_emb = torch.mm(self.fnc_assign.t(), self.struct_emb)

        # backward
        ## F2F
        self.fnc_adj = F.sigmoid(torch.mm(self.fnc_emb, self.fnc_emb.t()))  # n_f * n_f
        self.fnc_adj = self.fnc_adj + torch.eye(self.fnc_adj.shape[0]).to(self.hparams.device) * 1.0
        self.fnc_emb = self.fnc_gcn(self.fnc_emb.unsqueeze(0), self.fnc_adj.unsqueeze(0)).squeeze()

        ## F2C
        fnc_message = torch.div(torch.mm(self.raw_fnc_assign, self.fnc_emb),
                                (F.relu(torch.sum(self.fnc_assign, 1) - 1.0) + 1.0).unsqueeze(1))

        self.r_f = self.sigmoid(self.l_c(torch.cat((self.struct_emb, fnc_message), 1)))
        self.struct_emb = self.struct_emb + 0.15 * fnc_message  # magic number: 0.15

        ## C2C
        struct_adj = F.relu(struct_adj - torch.eye(struct_adj.shape[1]).to(self.hparams.device) * 10000.0) + torch.eye(
            struct_adj.shape[1]).to(self.hparams.device) * 1.0
        self.struct_emb = self.struct_gcn(self.struct_emb.unsqueeze(0), struct_adj.unsqueeze(0)).squeeze()

        ## C2N
        struct_message = torch.mm(self.raw_struct_assign, self.struct_emb)
        self.r_s = self.sigmoid(self.l_s(torch.cat((raw_feat, struct_message), 1)))
        #    raw_feat = raw_feat + 0.5 * struct_message

        ## N2N
        raw_feat = self.node_gat(raw_feat, raw_adj)
        return raw_feat


# gcn_layers

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape,
                b):  # indices, value and shape define a sparse tensor, it will do mm() operation with b
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SPGCN(Module):
    def __init__(self, in_features, out_features, device, concat=True):
        super(SPGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.device = device

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        #    self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
        pass

    def forward(self, inputs, adj):
        inputs = inputs.squeeze()
        #   adj = adj.squeeze()
        dv = self.device
        #    self_loop = torch.eye(adj.shape[0]).to(self.device)
        #    adj = adj + self_loop
        N = inputs.size()[0]
        #    edge = adj.nonzero().t()
        edge = adj._indices()
        #    edge = edge[1:, :]
        h = torch.mm(inputs, self.W)
        # h: N x out
        assert not torch.isnan(h).any()
        #    edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E
        #    edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        edge_e = torch.ones(edge.shape[1], dtype=torch.float).to(self.device)
        assert not torch.isnan(edge_e).any()
        # edge_e: E
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1
        #    edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        if self.concat:  # if this layer is not last layer,
            return F.elu(h_prime)
        else:  # if this layer is last layer
            return h_prime


class GraphConvolution(Module):
    """
      Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, device, bias=True):
        super(GraphConvolution, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.5
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def norm(self, adj):
        node_num = adj.shape[-1]
        # add remaining self-loops
        self_loop = torch.eye(node_num).to(self.device)
        self_loop = self_loop.reshape((1, node_num, node_num))
        self_loop = self_loop.repeat(adj.shape[0], 1, 1)
        adj_post = adj + self_loop
        # signed adjacent matrix
        deg_abs = torch.sum(torch.abs(adj_post), dim=-1)
        deg_abs_sqrt = deg_abs.pow(-0.5)
        diag_deg = torch.diag_embed(deg_abs_sqrt, dim1=-2, dim2=-1)

        norm_adj = torch.matmul(torch.matmul(diag_deg, adj_post), diag_deg)
        return norm_adj

    def forward(self, inputs, adj):
        support = torch.matmul(inputs, self.weight)
        adj_norm = self.norm(adj)
        output = torch.matmul(support.transpose(1, 2), adj_norm.transpose(1, 2))
        output = output.transpose(1, 2)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

