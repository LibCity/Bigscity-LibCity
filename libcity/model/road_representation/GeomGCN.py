import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import dgl.function as fn
from dgl import DGLGraph
from logging import getLogger
import scipy.sparse as sp
import networkx as nx
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
from libcity.model import utils


class GeomGCNSingleChannel(nn.Module):
    def __init__(self, g, in_feats, out_feats, num_divisions, activation, dropout_prob, merge, device):
        super(GeomGCNSingleChannel, self).__init__()
        self.num_divisions = num_divisions
        self.in_feats_dropout = nn.Dropout(dropout_prob)
        self.device = device

        self.linear_for_each_division = nn.ModuleList().to(self.device)
        for i in range(self.num_divisions):
            self.linear_for_each_division.append(nn.Linear(in_feats, out_feats, bias=False).to(self.device))

        for i in range(self.num_divisions):
            nn.init.xavier_uniform_(self.linear_for_each_division[i].weight.to(self.device))

        self.activation = activation
        self.g = g
        self.subgraph_edge_list_of_list = self.get_subgraphs(self.g)
        self.subgraph_node_list_of_list = self.get_node_subgraphs(self.g)
        self.merge = merge
        self.out_feats = out_feats

    def get_subgraphs(self, g):
        subgraph_edge_list = [[] for _ in range(self.num_divisions)]
        u, v, eid = g.all_edges(form='all')
        for i in range(g.number_of_edges()):
            subgraph_edge_list[g.edges[u[i], v[i]].data['subgraph_idx']].append(eid[i])

        return subgraph_edge_list

    def get_node_subgraphs(self, g):
        subgraph_node_list = [[] for _ in range(self.num_divisions)]
        u, v, eid = g.all_edges(form='all')
        for i in range(g.number_of_edges()):
            subgraph_node_list[g.edges[u[i], v[i]].data['subgraph_idx']].append(u[i])
            subgraph_node_list[g.edges[u[i], v[i]].data['subgraph_idx']].append(v[i])

        return subgraph_node_list

    def forward(self, feature):
        in_feats_dropout = self.in_feats_dropout(feature).to(self.device)    # 使输入挂上dropout
        self.g.ndata['h'] = in_feats_dropout.to(self.device)  # 使数据挂上dropout；ndata代表特征；加入关于h的索引；

        for i in range(self.num_divisions):
            subgraph = self.g.subgraph(self.subgraph_node_list_of_list[i])
            self.linear_for_each_division[i].to(self.device)
            temp = self.linear_for_each_division[i](subgraph.ndata['h'])
            subgraph.ndata[f'Wh_{i}'] = temp * subgraph.ndata['norm']
            subgraph.update_all(message_func=fn.copy_u(u=f'Wh_{i}', out=f'm_{i}'),
                                reduce_func=fn.sum(msg=f'm_{i}', out=f'h_{i}'))
            subgraph.ndata.pop(f'Wh_{i}')

        self.g.ndata.pop('h')

        results_from_subgraph_list = []
        for i in range(self.num_divisions):
            if f'h_{i}' in self.g.node_attr_schemes():
                results_from_subgraph_list.append(self.g.ndata.pop(f'h_{i}'))
            else:
                results_from_subgraph_list.append(
                    torch.zeros((feature.size(0), self.out_feats), dtype=torch.float32, device=feature.device))

        if self.merge == 'cat':
            h_new = torch.cat(results_from_subgraph_list, dim=-1).to(self.device)
        else:
            h_new = torch.mean(torch.stack(results_from_subgraph_list, dim=-1), dim=-1).to(self.device)
        h_new = h_new * self.g.ndata['norm']
        h_new = self.activation(h_new)
        return h_new


class GeomGCNNet(nn.Module):
    def __init__(self, g, in_feats, out_feats, num_divisions, activation, num_heads, dropout_prob, ggcn_merge,
                 channel_merge, device):
        super(GeomGCNNet, self).__init__()
        self.attention_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.attention_heads.append(
                GeomGCNSingleChannel(g, in_feats, out_feats, num_divisions,
                                     activation, dropout_prob, ggcn_merge, device))
        self.channel_merge = channel_merge
        self.g = g

    def forward(self, feature):
        all_attention_head_outputs = [head(feature) for head in self.attention_heads]
        if self.channel_merge == 'cat':
            return torch.cat(all_attention_head_outputs, dim=1)
        else:
            return torch.mean(torch.stack(all_attention_head_outputs), dim=0)


class GeomGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.device = config.get('device')
        g, num_input_features, num_output_classes, num_hidden, num_divisions, \
        num_heads_layer_one, num_heads_layer_two, dropout_rate, layer_one_ggcn_merge, \
        layer_one_channel_merge, layer_two_ggcn_merge, layer_two_channel_merge = self.get_input(config, data_feature)

        self.geomgcn1 = GeomGCNNet(g, num_input_features, num_hidden, num_divisions, F.relu, num_heads_layer_one,
                                   dropout_rate, layer_one_ggcn_merge, layer_one_channel_merge, self.device)

        if layer_one_ggcn_merge == 'cat':
            layer_one_ggcn_merge_multiplier = num_divisions
        else:
            layer_one_ggcn_merge_multiplier = 1

        if layer_one_channel_merge == 'cat':
            layer_one_channel_merge_multiplier = num_heads_layer_one
        else:
            layer_one_channel_merge_multiplier = 1

        self.geomgcn2 = GeomGCNNet(g, num_hidden * layer_one_ggcn_merge_multiplier * layer_one_channel_merge_multiplier,
                                   num_output_classes, num_divisions, lambda x: x,
                                   num_heads_layer_two, dropout_rate, layer_two_ggcn_merge,
                                   layer_two_channel_merge, self.device)

        self.geomgcn3 = GeomGCNNet(g, num_output_classes,
                                   num_hidden * layer_one_ggcn_merge_multiplier * layer_one_channel_merge_multiplier,
                                   num_divisions, lambda x: x,
                                   num_heads_layer_two, dropout_rate, layer_two_ggcn_merge,
                                   layer_two_channel_merge, self.device)

        self.geomgcn4 = GeomGCNNet(g, num_hidden * layer_one_ggcn_merge_multiplier * layer_one_channel_merge_multiplier,
                                   num_input_features,  num_divisions, F.relu, num_heads_layer_one,
                                   dropout_rate, layer_two_ggcn_merge, layer_two_channel_merge, self.device)

        self.g = g
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')
        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.output_dim = config.get('output_dim', 8)

    def get_input(self, config, data_feature):
        num_input_features = data_feature.get('feature_dim', 1)
        num_output_classes = config.get('output_dim', 8)
        num_hidden = config.get('hidden_dim', 144)
        num_divisions = config.get('divisions_dim', 2)
        num_heads_layer_one = config.get('num_heads_layer_one', 1)
        num_heads_layer_two = config.get('num_heads_layer_two', 1)
        dropout_rate = config.get('dropout_rate', 0.5)
        layer_one_ggcn_merge = config.get('layer_one_ggcn_merge', 'cat')
        layer_two_ggcn_merge = config.get('layer_two_ggcn_merge', 'mean')
        layer_one_channel_merge = config.get('layer_one_channel_merge', 'cat')
        layer_two_channel_merge = config.get('layer_two_channel_merge', 'mean')
        adj_mx = data_feature.get('adj_mx')

        G = nx.DiGraph(adj_mx)

        for (node1, node2) in G.edges:
            G.remove_edge(node1, node2)
            G.add_edge(node1, node2, subgraph_idx=0)

        for node in sorted(G.nodes):
            if G.has_edge(node, node):
                G.remove_edge(node, node)
            G.add_edge(node, node, subgraph_idx=1)
        adj = nx.adjacency_matrix(G, sorted(G.nodes))
        g = DGLGraph(adj)
        g = g.to(self.device)

        for u, v, feature in G.edges(data='subgraph_idx'):
            if (feature is not None) and g.has_edge_between(u, v):
                g.edges[g.edge_id(u, v)].data['subgraph_idx'] = torch.tensor([feature]).to(self.device)

        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        g.ndata['norm'] = norm.unsqueeze(1).to(self.device).requires_grad_()

        return g, num_input_features, num_output_classes, num_hidden, num_divisions, \
            num_heads_layer_one, num_heads_layer_two, dropout_rate, layer_one_ggcn_merge, \
            layer_one_channel_merge, layer_two_ggcn_merge, layer_two_channel_merge

    def forward(self, batch):
        """
        自回归任务

        Args:
            batch: dict, need key 'node_features' contains tensor shape=(N, feature_dim)

        Returns:
            torch.tensor: N, output_classes
        """
        inputs = batch['node_features']
        x = self.geomgcn1(inputs)
        encoder_state = self.geomgcn2(x)
        np.save('./libcity/cache/evaluate_cache/embedding_{}_{}_{}.npy'
                .format(self.model, self.dataset, self.output_dim),
                encoder_state.detach().cpu().numpy())
        x = self.geomgcn3(encoder_state)
        output = self.geomgcn4(x)
        return output

    def calculate_loss(self, batch):
        """
        Args:
            batch: dict, need key 'node_features', 'node_labels', 'mask'

        Returns:

        """
        y_true = batch['node_labels']
        y_predicted = self.predict(batch)
        mask = batch['mask']
        return loss.masked_mse_torch(y_predicted[mask], y_true[mask])

    def predict(self, batch):
        """
        Args:
            batch: dict, need key 'node_features'

        Returns:
            torch.tensor

        """
        return self.forward(batch)
