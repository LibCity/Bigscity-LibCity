import random
from decimal import Decimal
from logging import getLogger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from tqdm import tqdm



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
        return self.relu(self.batch(self.conv(inp)))



