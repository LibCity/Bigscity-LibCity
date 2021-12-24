from collections import OrderedDict

import torch
import torch.nn as nn


class SpatialViewConv(nn.Module):
    def __init__(self, inp_channel, oup_channel, kernel_size, stride=1, padding=0):
        super(SpatialViewConv, self).__init__()
        self.inp_channel = inp_channel
        self.oup_channel = oup_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels=inp_channel, out_channels=oup_channel,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch = nn.BatchNorm2d(oup_channel)
        self.relu = nn.ReLU()

    def forward(self, inp):
        """

        Args:
            inp:

        Returns:

        """
        return self.relu(self.batch(self.conv(inp)))


# 两个cnn，一个用relu，一个用sigmod然后两个相乘 STDN
class SpatialGatedCNN(nn.Module):
    def __init__(self,input_size,output_size,kernal_size=(3,3),padding=1):
        self.input_size=input_size
        self.output_size=output_size
        self.cnn1=nn.Conv2d(input_size,output_size,kernal_size,padding=padding)
        self.cnn2 = nn.Conv2d(input_size, output_size, kernal_size, padding=padding)

    def forward(self, inputs):
        """
        Args:
            inputs: shape = (b,t,c_in,h,w)
        Returns:
            [b,t,c_out,h,w]
        """
        gate=nn.Sigmoid(nn.ReLU(self.cnn1(inputs)))
        value=nn.ReLU(self.cnn2)

        return torch.matmul(gate,value)


class BnReluConv(nn.Module):
    def __init__(self, nb_filter, kernel_size=(3,3),stride=1,bn=False):
        super(BnReluConv, self).__init__()
        self.has_bn = bn
        self.bn1 = nn.BatchNorm2d(nb_filter)
        self.relu = torch.relu
        self.conv1 = nn.Conv2d(nb_filter, nb_filter, kernel_size=kernel_size,
                     stride=stride, padding=1, bias=True)

    def forward(self, x):
        if self.has_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        return x


class ResidualUnit(nn.Module):
    def __init__(self, nb_filter, bn=False):
        super(ResidualUnit, self).__init__()
        self.bn_relu_conv1 = BnReluConv(nb_filter, bn)
        self.bn_relu_conv2 = BnReluConv(nb_filter, bn)

    def forward(self, x):
        residual = x
        out = self.bn_relu_conv1(x)
        out = self.bn_relu_conv2(out)
        out += residual  # short cut
        return out


# 两个 cnn 中间 relu 最后Residual ST-ResNet
class SpatialResCNN(nn.Module):
    def __init__(self,dim,bn=False):
        self.model=ResidualUnit(dim,bn)

    def forward(self,inputs):
        return self.model(inputs)


