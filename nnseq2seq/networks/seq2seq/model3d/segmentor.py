import torch
import torch.nn as nn
import torch.nn.functional as F

from nnseq2seq.networks.seq2seq.model3d.convnext import LayerNorm


class Segmentor(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.c_in = args['latent_space_dim']
        self.num_classes = args['num_classes']
        self.upsample = args['upsample_scale']

        self.head = DeepLabHead(self.c_in, self.num_classes)
        self.up = nn.Upsample(scale_factor=self.upsample, mode='trilinear', align_corners=True)

    def forward(self, x):
        x = self.head(x)
        x = self.up(x)
        return x


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36], 256),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            LayerNorm(256, eps=1e-6, data_format="channels_first"),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(256, out_channels, 1)
        )


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rate, out_channels=256):
        super(ASPP, self).__init__()

        modules = [nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
            nn.LeakyReLU(0.01, inplace=True)
        )]

        rates = tuple(atrous_rate)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv3d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, rate):
        super(ASPPConv, self).__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=False),
            LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
            nn.LeakyReLU(0.01, inplace=True),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, x):
        size = x.shape[-3:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='trilinear', align_corners=False)