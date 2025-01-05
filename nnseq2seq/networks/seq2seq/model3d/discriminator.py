import torch
import torch.nn as nn
import torch.nn.functional as F

from nnseq2seq.networks.seq2seq.model3d.convnext import LayerNorm, hyperAttnResBlock


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, args):
        super(NLayerDiscriminator, self).__init__()

        c_in = args['in_channels']
        ndf = args['ndf']
        n_layers = args['n_layers']
        kw = args['kw']
        padw = args['padw']
        hyper_dim = args['hyper_conv_dim']
        style_dim = args['style_dim']
        layer_scale_init_value = args['layer_scale_init_value']

        sequence = [nn.Conv3d(c_in, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                LayerNorm(ndf * nf_mult, eps=1e-6, data_format="channels_first"),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            LayerNorm(ndf * nf_mult, eps=1e-6, data_format="channels_first"),
            nn.LeakyReLU(0.2, True)
        ]

        #sequence += [
        #    nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)
        self.hyper = hyperAttnResBlock(ndf*nf_mult, style_dim, 2, hyper_dim, 1, 0, layer_scale_init_value=layer_scale_init_value, use_attn=True)

    def forward(self, input, s):
        """Standard forward."""
        x = self.main(input)
        x = self.hyper(x, s)
        return x