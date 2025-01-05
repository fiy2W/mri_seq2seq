import numpy as  np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnseq2seq.networks.seq2seq.model2d.convnext import LayerNorm, hyperAttnResBlock, hyperConv


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
        self.focal_mode = args['focal_mode']

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
        
        self.image_fusion = nn.Sequential(
            nn.Conv2d(self.latent_space_dim, out_channels=3, kernel_size=1, padding=0, stride=1, padding_mode='zeros'),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.image_fusion_inv = nn.Sequential(
            nn.Conv2d(3, out_channels=self.latent_space_dim, kernel_size=1, padding=0, stride=1, padding_mode='zeros'),
        )
            
    def forward(self, zqs, s, latent_focal=None):
        outputs = []
        up_scale = np.prod(self.s_enc)

        if latent_focal is None:
            focal_mode = self.focal_mode
        else:
            focal_mode = latent_focal
        
        if focal_mode=='dispersion':
            zid = 0
        elif focal_mode=='focal_x1':
            zid = -1
        elif focal_mode=='focal_x2':
            zid = -2
        elif focal_mode=='focal_x4':
            zid = -3
        elif focal_mode=='focal_x8':
            zid = -4
        elif focal_mode=='focal_x16':
            zid = -5
        elif focal_mode=='focal_mix':
            zid = random.choice([0, 0, 0, 0, 0, -1, -2, -3, -4, -5])
        else:
            raise

        for i, (z0, down, midr, deep, up) in enumerate(zip(zqs, self.down_layers, self.midres_layers, self.deep_layers, self.up_layers)):
            if zid==0:
                z = z0
            else:
                z = F.interpolate(zqs[zid], size=z0.shape[2:], mode='bilinear')
            if i==0:
                x = down(z, s)
            else:
                x = down(torch.cat([z, x], dim=1), s)
            
            x = midr(x, s)
            outputs.append(deep(x))
            x = up(x)
            up_scale = up_scale//self.s_enc[i]
            
        outputs = outputs[::-1]

        image_fusion = self.image_fusion(zqs[-1].detach())
        latent_inv = self.image_fusion_inv(image_fusion)

        if not self.deep_supervision:
            return outputs[0], [image_fusion, latent_inv]
        else:
            return outputs, [image_fusion, latent_inv]