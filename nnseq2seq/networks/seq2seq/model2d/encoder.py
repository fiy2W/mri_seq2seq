import torch
import torch.nn as nn
import torch.nn.functional as F

from nnseq2seq.networks.seq2seq.model2d.convnext import LayerNorm, AttnResBlock, hyperConv, hyperAttnResBlock
from nnseq2seq.networks.seq2seq.model2d.quantize import VectorQuantizer2 as VectorQuantizer


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
        self.hyper_dim = args['hyper_conv_dim']
        self.latent_space_dim = args['latent_space_dim']
        self.style_dim = args['style_dim']
        self.vq_beta = args['vq_beta']
        self.vq_n_embed = args['vq_n_embed']

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.conv_latent = nn.ModuleList()
        self.quantize = VectorQuantizer(self.vq_n_embed, self.latent_space_dim, beta=self.vq_beta)
        self.latent_fusion = nn.Sequential(
            nn.Linear(self.style_dim, self.style_dim*4),
            nn.Linear(self.style_dim*4, self.style_dim),
        )
        c_pre = self.c_in
        up_scale = 1
        for i, (ce, ke, se, nr, kr, pr) in enumerate(zip(self.c_enc, self.k_enc, self.s_enc, self.n_res, self.k_res, self.p_res)):
            if i==0:
                block = [
                    nn.Conv2d(in_channels=c_pre, out_channels=ce, kernel_size=ke, padding=(ke-se)//2, stride=se),
                ]
                if nr!=0:
                    block.append(LayerNorm(ce, eps=1e-6, data_format="channels_first"))
            else:
                block = [
                    LayerNorm(c_pre, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(in_channels=c_pre, out_channels=ce, kernel_size=ke, padding=(ke-se)//2, stride=se),
                ]
                
            up_scale = up_scale * se
            block.append(AttnResBlock(dim=ce, n_layer=nr, kernel_size=kr, padding=pr, layer_scale_init_value=self.layer_scale_init_value, use_attn=True if up_scale>=4 else False))
            self.down_layers.append(nn.Sequential(*block))
            c_pre = ce

            if se==1:
                self.up_layers.append(nn.Identity())
            else:
                self.up_layers.append(nn.Sequential(
                    nn.Upsample(scale_factor=se, mode='nearest'),
                ))
            self.conv_latent.append(nn.Conv2d(in_channels=ce if up_scale==2**(len(self.c_enc)-1) else ce+self.latent_space_dim, out_channels=self.latent_space_dim, kernel_size=3, padding=1, stride=1))

    def tsf(self, z, s, weight, level, b, n):
        _, c0, w0, h0 = z.shape
        z = z.reshape(b, n, c0, w0, h0)*s.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        z = torch.sum(z*weight, dim=1)
        return z, c0

    def forward(self, x, s_all, s_subgroup):
        features = []
        zqs_all = []
        zqs_subgroup = []
        vq_losses = 0

        b, n, w, h = x.shape
        x = x.reshape(-1, 1, w, h)
        for down in self.down_layers:
            x = down(x)
            features.append(x)
        
        weight_all = (self.latent_fusion(s_all)*s_all).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        weight_sub = (self.latent_fusion(s_subgroup)*s_subgroup).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        for i in reversed(range(len(features))):
            if i==len(features)-1:
                z1 = self.conv_latent[i](x)
                z2 = z1
            else:
                z1 = self.conv_latent[i](torch.cat([x1, features[i]], dim=1))
                z2 = self.conv_latent[i](torch.cat([x2, features[i]], dim=1))

            z1, c0 = self.tsf(z1, s_all, weight_all, i, b, n)
            z2, _ = self.tsf(z2, s_subgroup, weight_sub, i, b, n)
            
            zq1, vq_loss1, _ = self.quantize(z1)
            zq2, vq_loss2, _ = self.quantize(z2)
            zqs_all.append(zq1)
            zqs_subgroup.append(zq2)
            vq_losses += (vq_loss1 + vq_loss2 + self.vq_beta * torch.mean((zq1.detach()-z2)**2))
            
            x1 = torch.tile(self.up_layers[i](zq1).unsqueeze(1), (1,n,1,1,1))
            x2 = torch.tile(self.up_layers[i](zq2).unsqueeze(1), (1,n,1,1,1))
            w0, h0 = x1.shape[3:]
            x1 = x1.reshape(-1,c0,w0,h0)
            x2 = x2.reshape(-1,c0,w0,h0)
        
        return zqs_all, zqs_subgroup, vq_losses