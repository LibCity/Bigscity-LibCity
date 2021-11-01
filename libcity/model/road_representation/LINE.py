import torch.nn as nn

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class LINE(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.device = config.get('device')

        self.order = config.get('order')
        self.negative_ratio = config.get('negative_ratio')
        self.embedding_size = config.get('embedding_size')
        self.num_nodes = data_feature.get("num_nodes")
        self.num_edges = data_feature.get("num_edges")
        self.batch_size = data_feature.get("batch_size")

        self.samples_per_epoch = self.num_edges * (1 + self.negative_ratio)

        self.node_emb = nn.Embedding(self.num_nodes, self.embedding_size)

        if self.order == 'second':
            self.context_emb = nn.Embedding(self.num_nodes, self.embedding_size)
        elif self.order != 'first':
            raise ValueError("order mode must be first or second")

    def forward(self, vi, vj, neg):
        """
        Args:
            vi : origin vector of node i
            vj : origin vector of node j
            neg: origin vectors of nodes neg
        """
        vi = self.node_emb(vi)
        if self.order == 'first':
            vj = self.node_emb(vj)
            neg = self.node_emb(neg)
        else:
            vj = self.context_emb(vj)
            negs = self.context_emb(neg)
