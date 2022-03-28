import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class LINE_FIRST(nn.Module):
    def __init__(self, num_nodes, output_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.node_emb = nn.Embedding(self.num_nodes, self.output_dim)

    def forward(self, i, j):
        """
        Args:
            i: indices of i; (B,)
            j: indices of j; (B,)
        Return:
            v_i^T * v_j; (B,)
        """
        vi = self.node_emb(i)
        vj = self.node_emb(j)
        return (vi * vj).sum(dim=-1)

    def get_embeddings(self):
        return self.node_emb.weight.data


class LINE_SECOND(nn.Module):
    def __init__(self, num_nodes, output_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.node_emb = nn.Embedding(self.num_nodes, self.output_dim)
        self.context_emb = nn.Embedding(self.num_nodes, self.output_dim)

    def forward(self, I, J):
        """
        Args:
            I: indices of i; (B,)
            J: indices of j; (B,)
        Return:
            [v_i^T * u_j for (i,j) in zip(I,J)]; (B,)
        """
        vi = self.node_emb(I)
        vj = self.context_emb(J)
        return (vi * vj).sum(dim=-1)

    def get_embeddings(self):
        return self.node_emb.weight.data


class LINE(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.device = config.get('device')

        self.order = config.get('order')
        self.output_dim = config.get('output_dim')
        self.num_nodes = data_feature.get("num_nodes")
        self.num_edges = data_feature.get("num_edges")

        if self.order == 'first':
            self.embed = LINE_FIRST(self.num_nodes, self.output_dim)
        elif self.order == 'second':
            self.embed = LINE_SECOND(self.num_nodes, self.output_dim)
        else:
            raise ValueError("order mode must be first or second")

        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)

    def calculate_loss(self, batch):
        I, J, is_neg = batch['I'], batch['J'], batch['Neg']
        dot_product = self.forward(I, J)
        return -(F.logsigmoid(dot_product * is_neg)).mean()

    def forward(self, I, J):
        """
        Args:
            I : origin indices of node i ; (B,)
            J : origin indices of node j ; (B,)
        Return:
            if order == 'first':
                [u_j^T * u_i for (i,j) in zip(I, J)]; (B,)
            elif order == 'second':
                [u'_j^T * v_i for (i,j) in zip(I, J)]; (B,)
        """
        np.save('./libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.npy'
                .format(self.exp_id, self.model, self.dataset, self.output_dim),
                self.embed.get_embeddings().cpu().numpy())
        return self.embed(I, J)
