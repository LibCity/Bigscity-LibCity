import torch
import torch.nn as nn
import numpy as np
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
from libcity.model import utils


class ChebConvModule(nn.Module):
    """
    路网表征模型的基类并不统一
    图卷积，将N*C的输入矩阵映射成N*F的输出矩阵，其中邻接矩阵形状N*N。
    """
    def __init__(self, num_nodes, max_diffusion_step, adj_mx,
                 device, input_dim, output_dim, filter_type):
        """
        K阶切比雪夫估计
        Args:
            num_nodes: 节点个数n
            max_diffusion_step: K阶
            adj_mx: list of 拉普拉斯矩阵
            device: 设备
            input_dim: 输入维度
            output_dim: 输出维度
        """
        super().__init__()
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        # 计算拉普拉斯
        supports = utils.get_supports_matrix(adj_mx=adj_mx, filter_type=filter_type)
        results = []
        for support in supports:
            results.append(utils.build_sparse_matrix(device, support))
        self._supports = results
        self._device = device
        self._ks = len(self._supports) * self._max_diffusion_step + 1  # Ks
        self._input_dim = input_dim
        self._output_dim = output_dim
        shape = (self._input_dim * self._ks, self._output_dim)
        self.weight = torch.nn.Parameter(torch.empty(*shape, device=self._device))
        self.biases = torch.nn.Parameter(torch.empty(self._output_dim, device=self._device))
        torch.nn.init.xavier_normal_(self.weight)
        torch.nn.init.constant_(self.biases, 0)

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, x):
        """
        GONV
        :param x: (N, input_dim)
        :return:  (N, output_dim)
        """
        num_nodes, input_dim = x.shape
        # T(0)=I x(0)=T(0)*x=x
        x0 = x  # (N, C)
        x = torch.unsqueeze(x0, 0)  # (1, N, C)

        # 3阶[T0,T1,T2] Chebyshev多项式近似g(theta)
        for support in self._supports:
            x1 = torch.sparse.mm(support, x0)  # supports: N*N; x0: (N, C) --> (N, C)
            x = self._concat(x, x1)  # (2, N, C)
            for k in range(2, self._max_diffusion_step + 1):
                x2 = 2 * torch.sparse.mm(support, x1) - x0
                x = self._concat(x, x2)  # (3, N, C)
                x1, x0 = x2, x1  # 循环
        # x.shape (Ks, N, C)
        # Ks = len(supports) * self._max_diffusion_step + 1

        x = torch.reshape(x, shape=[self._ks, self._num_nodes, input_dim])  # (Ks, N, C)
        x = x.permute(1, 2, 0)  # (N, C, Ks)
        x = torch.reshape(x, shape=[self._num_nodes, input_dim * self._ks])  # (N, Ks * C)
        # (N, Ks * C) * (Ks * C, F)  --> (N, F)
        x = torch.matmul(x, self.weight)  # (N, F)
        x += self.biases  # (N, F)
        return x


class ChebConv(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.adj_mx = data_feature.get('adj_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        config['num_nodes'] = self.num_nodes
        config['feature_dim'] = self.feature_dim
        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        self.max_diffusion_step = config.get('max_diffusion_step', 2)
        self.output_dim = config.get('output_dim', 32)
        self.filter_type = config.get('filter_type', 'dual_random_walk')
        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)

        self.encoder = ChebConvModule(num_nodes=self.num_nodes, max_diffusion_step=self.max_diffusion_step,
                                      adj_mx=self.adj_mx, device=self.device, input_dim=self.feature_dim,
                                      output_dim=self.output_dim, filter_type=self.filter_type)
        self.decoder = ChebConvModule(num_nodes=self.num_nodes, max_diffusion_step=self.max_diffusion_step,
                                      adj_mx=self.adj_mx, device=self.device, input_dim=self.output_dim,
                                      output_dim=self.feature_dim, filter_type=self.filter_type)

    def forward(self, batch):
        """
        自回归任务

        Args:
            batch: dict, need key 'node_features' contains tensor shape=(N, feature_dim)

        Returns:
            torch.tensor: N, feature_dim

        """
        inputs = batch['node_features']
        encoder_state = self.encoder(inputs)  # N, output_dim
        np.save('./libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.npy'
                .format(self.exp_id, self.model, self.dataset, self.output_dim),
                encoder_state.detach().cpu().numpy())
        output = self.decoder(encoder_state)  # N, feature_dim
        return output

    def calculate_loss(self, batch):
        """

        Args:
            batch: dict, need key 'node_features', 'node_labels', 'mask'

        Returns:

        """
        y_true = batch['node_labels']  # N, feature_dim
        y_predicted = self.predict(batch)  # N, feature_dim
        y_true = self._scaler.inverse_transform(y_true)
        y_predicted = self._scaler.inverse_transform(y_predicted)
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
