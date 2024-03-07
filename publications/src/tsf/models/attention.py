import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('./publications/')

from src.seq2seq.models.hyperconv import hyperConv


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, weight_dim=8, style_dim=64):
        super(ChannelAttention, self).__init__()
        self.ndims = 2
        self.weight_dim = weight_dim
        self.style_dim = style_dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc1 = hyperConv(self.style_dim, in_planes, in_planes // ratio, ksize=1, stride=1, padding=0, weight_dim=self.weight_dim, ndims=self.ndims)
        self.fc2 = nn.LeakyReLU(0.2, True)
        self.fc3 = hyperConv(self.style_dim, in_planes // ratio, in_planes, ksize=1, stride=1, padding=0, weight_dim=self.weight_dim, ndims=self.ndims)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, s):
        avg_out = self.fc3(self.fc2(self.fc1(self.avg_pool(x), s)), s)
        max_out = self.fc3(self.fc2(self.fc1(self.max_pool(x), s)), s)
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, weight_dim=8, style_dim=64):
        super(SpatialAttention, self).__init__()
        self.ndims = 2
        self.weight_dim = weight_dim
        self.style_dim = style_dim

        self.conv1 = hyperConv(self.style_dim, 2, 1, ksize=kernel_size, stride=1, padding=kernel_size//2, weight_dim=self.weight_dim, ndims=self.ndims, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, s):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x, s)
        return self.sigmoid(x)