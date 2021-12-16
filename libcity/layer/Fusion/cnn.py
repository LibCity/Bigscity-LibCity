import torch
import torch.nn as nn


class TemporalCONV(nn.Module):
    def __init__(self, config,gcn):
        super(TemporalCONV, self).__init__()
        self.input_fileter = config.get("input_fileter")
        self.out_time_fileter = config.get("out_time_fileter")
        self.time_strides = config.get("time_strides")
        self.kernel_size = config.get("kernel_size",(1,3))
        self.gcn=gcn(config['gcn'])
        self.time_conv = nn.Conv2d(self.input_fileter, self.out_time_fileter, kernel_size=self.kernel_size,
                                   stride=(1, self.time_strides), padding=(0, 1))

    def forward(self, inputs):
        """
        (B, T, N , F_in) -> (B, F_in, N, T) 用(1,3)的卷积核去做->(B, F_out', N, T')
        Args:
            inputs: (B, T, N , F_in)

        Returns:
            tensor : (B, T', N, F_out)
        """
        inputs = self.gcn(inputs)

        time_conv_output = self.time_conv(inputs.permute(0, 3, 2, 1))

        time_conv_output = time_conv_output.permute(0, 3, 2, 1)

        return time_conv_output


class CONVGCN(nn.Module):
    """
        gcn + conv3d
    """
    def __init__(self, config, gcn):
        super(CONVGCN, self).__init__()
        self.device = config.get('device', torch.device('cpu'))
        self.num_nodes = config.get('num_nodes', 1)
        self.feature_dim = self.config.get('feature_dim', 1)  # 输入维度
        self.output_dim = self.config.get('output_dim', 1)  # 输出维度
        self.conv_depth = config.get('conv_depth', 5)
        self.conv_height = config.get('conv_height', 3)
        self.hidden_size = config.get('hidden_size', 16)
        self.time_lag = config.get('time_lag', 1)
        self.output_window = config.get('output_window', 1)
        """
            config['gcn'] = {adj, num_nodes, dim_in, dim_out, ...}
        """
        self.gcn = gcn(config['gcn'])
        self.Conv = nn.Conv3d(
            in_channels=self.num_nodes,
            out_channels=self.num_nodes,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(
            self.num_nodes * self.conv_depth * self.conv_height * self.feature_dim,
            self.num_nodes * self.output_window * self.output_dim
        )
    def forward(self, inputs):

        out = self.gcn(inputs)
        out = torch.reshape(
            out,
            (-1, self.num_nodes, self.conv_depth, self.conv_height, self.feature_dim)
        )
        out = self.relu(self.Conv(out))
        out = out.view(-1, self.num_nodes * self.conv_depth * self.conv_height * self.feature_dim)
        out = self.fc(out)
        out = torch.reshape(out, [-1, self.output_window, self.num_nodes, self.output_dim])
        return out
