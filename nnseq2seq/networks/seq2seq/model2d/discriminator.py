import torch
import torch.nn as nn
import torch.nn.functional as F

from nnseq2seq.networks.seq2seq.model2d.convnext import LayerNorm, hyperAttnResBlock


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, args):
        super(NLayerDiscriminator, self).__init__()

        self.c_in = args['in_channels']
        self.c_enc = args['conv_channels']
        self.layer_scale_init_value = args['layer_scale_init_value']
        self.hyper_dim = args['hyper_conv_dim']
        self.style_dim = args['style_dim']

        ndf = self.c_enc[0]
        kw = 4
        padw = 1
        
        sequence = [nn.Conv2d(self.c_in, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nc_pre = ndf
        for nc in self.c_enc[1:]:  # gradually increase the number of filters
            sequence += [
                nn.Conv2d(nc_pre, nc, kernel_size=kw, stride=2, padding=padw),
                LayerNorm(nc, eps=1e-6, data_format="channels_first"),
                nn.LeakyReLU(0.2, True),
            ]
            nc_pre = nc

        self.encoder = nn.Sequential(*sequence)
        self.hyperblocks = hyperAttnResBlock(nc, self.style_dim, 2, latent_dim=self.hyper_dim, kernel_size=3, padding=1, layer_scale_init_value=self.layer_scale_init_value, use_attn=True)
        self.conv_out = nn.Conv2d(nc, 1, kernel_size=kw, stride=1, padding=padw)  # output 1 channel prediction map

    def forward(self, input, s):
        x = self.encoder(input)
        x = self.hyperblocks(x, s)
        x = self.conv_out(x)

        return x