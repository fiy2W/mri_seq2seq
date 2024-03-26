import torch
import torch.nn as nn
import torch.nn.functional as F

from nnseq2seq.networks.seq2seq.model2d.convnext import Block, LayerNorm, ResBlock, hyperResBlock


class ImageEncoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        
        self.c_in = args['in_channels']
        self.c_enc = args['conv_channels']
        self.k_enc = args['conv_kernel']
        self.s_enc = args['conv_stride']
        self.n_res = args['resblock_n']
        self.k_res = args['resblock_kernel']
        self.p_res = args['resblock_padding']
        self.layer_scale_init_value = args['layer_scale_init_value']
        self.latent_space_dim = args['latent_space_dim']

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        c_pre = self.c_in
        up_scale = 1
        up_channel = None
        for i, (ce, ke, se, nr, kr, pr) in enumerate(zip(self.c_enc, self.k_enc, self.s_enc, self.n_res, self.k_res, self.p_res)):
            if i==0:
                block = [
                    nn.Conv2d(in_channels=c_pre, out_channels=ce, kernel_size=se, padding=0, stride=se),
                ]
                if nr!=0:
                    block.append(LayerNorm(ce, eps=1e-6, data_format="channels_first"))
                up_channel = ce
            else:
                block = [
                    LayerNorm(c_pre, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(in_channels=c_pre, out_channels=ce, kernel_size=se, padding=0, stride=se),
                ]
                up_scale = up_scale * se
            block.extend([Block(dim=ce, kernel_size=kr, padding=pr, layer_scale_init_value=self.layer_scale_init_value) for _ in range(nr)])
            self.down_layers.append(nn.Sequential(*block))
            c_pre = ce

            if up_scale==1:
                self.up_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels=ce, out_channels=up_channel, kernel_size=1, padding=0, stride=1),
                    LayerNorm(up_channel, eps=1e-6, data_format="channels_first"),
                ))
            else:
                self.up_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels=ce, out_channels=up_channel, kernel_size=1, padding=0, stride=1),
                    LayerNorm(up_channel, eps=1e-6, data_format="channels_first"),
                    nn.Upsample(scale_factor=up_scale, mode='nearest'),
                ))
        self.conv_latent = nn.Conv2d(in_channels=up_channel*len(self.c_enc), out_channels=self.latent_space_dim, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        features = []
        for down, up in zip(self.down_layers, self.up_layers):
            x = down(x)
            f = up(x)
            features.append(f)
        features = torch.cat(features, dim=1)
        z = self.conv_latent(features)
        return z