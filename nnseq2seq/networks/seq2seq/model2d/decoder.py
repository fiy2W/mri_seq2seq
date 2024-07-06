import numpy as  np

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnseq2seq.networks.seq2seq.model2d.convnext import LayerNorm, AttnResBlock, hyperAttnResBlock, hyperConv


class HyperImageDecoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        
        self.c_out = args['out_channels']
        self.c_enc = args['conv_channels']
        self.k_enc = args['conv_kernel']
        self.s_enc = args['conv_stride']
        self.n_res = args['resblock_n']
        self.k_res = args['resblock_kernel']
        self.p_res = args['resblock_padding']
        self.layer_scale_init_value = args['layer_scale_init_value']
        self.hyper_dim = args['hyper_conv_dim']
        self.latent_space_dim = args['latent_space_dim']
        self.style_dim = args['style_dim']
        self.deep_supervision = args['deep_supervision']

        self.down_layers = nn.ModuleList()
        self.midres_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.deep_layers = nn.ModuleList()
        up_scale = np.prod(self.s_enc)
        for i, (ce, ke, se, nr, kr, pr) in enumerate(zip(self.c_enc, self.k_enc, self.s_enc, self.n_res, self.k_res, self.p_res)):
            self.down_layers.append(
                hyperConv(self.style_dim, self.latent_space_dim if i==0 else self.latent_space_dim+ce, ce, ksize=3, padding=1, weight_dim=self.hyper_dim, ndims=2),
            )
            self.midres_layers.append(
                hyperAttnResBlock(ce, self.style_dim, nr, self.hyper_dim, kr, pr, layer_scale_init_value=self.layer_scale_init_value, use_attn=True if up_scale>=4 else False)
            )
            self.deep_layers.append(nn.Sequential(
                LayerNorm(ce, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(ce, out_channels=self.c_out, kernel_size=3, padding=1, stride=1, padding_mode='zeros'),
                nn.LeakyReLU(0.01, inplace=True),
            ))
            if i==(len(self.c_enc)-1):
                self.up_layers.append(nn.Identity())
            else:
                self.up_layers.append(nn.Sequential(
                    LayerNorm(ce, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(ce, out_channels=self.c_enc[i+1], kernel_size=3, padding=1, stride=1, padding_mode='zeros'),
                    nn.Upsample(scale_factor=se, mode='nearest'),
                ))
                up_scale = up_scale//se
            
    def forward(self, zqs, s):
        outputs = []
        up_scale = np.prod(self.s_enc)
        for i, (z, down, midr, deep, up) in enumerate(zip(zqs, self.down_layers, self.midres_layers, self.deep_layers, self.up_layers)):
            if i==0:
                x = down(z, s)
            else:
                x = down(torch.cat([z, x], dim=1), s)
            
            x = midr(x, s)
            outputs.append(deep(x))
            x = up(x)
            up_scale = up_scale//self.s_enc[i]
            
        outputs = outputs[::-1]

        if not self.deep_supervision:
            return outputs[0]
        else:
            return outputs