import torch
import torch.nn as nn
from libcity.layer.utils import Align


# Temporal CNN
class TemporalConvLayer(nn.Module):
    def __init__(self, input_size, output_size, time_strides, kernel_size=(1, 3)):
        """
        Args:
            input_size=d
            output_size=d'
            time_strides:
            kernel_size:
        """

        self.input_size = input_size
        self.output_size = output_size
        self.time_strides = time_strides
        self.temporalCNN = nn.Conv2d(input_size, output_size, kernel_size, stride=(1, time_strides), padding=(0, 1))

    def forward(self, inputs):
        """
        Args:
            inputs:(b,t,n,d)
        Returns:
            (b,t‘,n,d')
            t'= (time_strides * t + 2 * pad - ker) / time_strides + 1
        """
        output = self.temporalCNN(inputs.permute(0, 2, 1, 3))
        output = output.permute(0, 2, 1, 3)
        return output


# gate cnn for temporal STGCN
class TemporalGatedConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(TemporalGatedConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        """

        :param x: (batch_size, feature_dim(c_in), input_length, num_nodes)
        :return: (batch_size, c_out, input_length-kt+1, num_nodes)
        """
        x_in = self.align(x)[:, :, self.kt - 1:, :]  # (batch_size, c_out, input_length-kt+1, num_nodes)
        if self.act == "GLU":
            # x: (batch_size, c_in, input_length, num_nodes)
            x_conv = self.conv(x)
            # x_conv: (batch_size, c_out * 2, input_length-kt+1, num_nodes)  [P Q]
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
            # return P * sigmoid(Q) shape: (batch_size, c_out, input_length-kt+1, num_nodes)
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)  # residual connection
        return torch.relu(self.conv(x) + x_in)  # residual connection


