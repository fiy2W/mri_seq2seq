import torch
import torch.nn as nn
import torch.nn.functional as F

from nnseq2seq.networks.seq2seq.model3d.convnext import LayerNorm, AttnResBlock
from nnseq2seq.networks.seq2seq.model3d.quantize import VectorQuantizer2 as VectorQuantizer


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
            #nn.Softmax(1),
            nn.Sigmoid(),
        )
        c_pre = self.c_in
        up_scale = 1
        for i, (ce, ke, se, nr, kr, pr) in enumerate(zip(self.c_enc, self.k_enc, self.s_enc, self.n_res, self.k_res, self.p_res)):
            if i==0:
                block = [
                    nn.Conv3d(in_channels=c_pre, out_channels=ce, kernel_size=ke, padding=(ke-se)//2, stride=se),
                ]
                if nr!=0:
                    block.append(LayerNorm(ce, eps=1e-6, data_format="channels_first"))
            else:
                block = [
                    LayerNorm(c_pre, eps=1e-6, data_format="channels_first"),
                    nn.Conv3d(in_channels=c_pre, out_channels=ce, kernel_size=ke, padding=(ke-se)//2, stride=se),
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
            self.conv_latent.append(nn.Conv3d(in_channels=ce if up_scale==2**(len(self.c_enc)-1) else ce+self.latent_space_dim, out_channels=self.latent_space_dim, kernel_size=3, padding=1, stride=1))

    def tsf(self, z, s, weight, level, b, n):
        _, c0, d0, w0, h0 = z.shape
        s = s.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        z = z.reshape(b, n, c0, d0, w0, h0)
        z_mean = torch.sum(z*weight, dim=1)/(torch.sum(weight*s, dim=1)+1e-5)
        tsf_loss = torch.mean(((z-z_mean.unsqueeze(1).detach())*s)**2)
        return z_mean, tsf_loss

    def forward(self, x, s_all, s_subgroup):
        features = []
        zqs_all = []
        zqs_subgroup = []
        zs_all_seg = []
        zs_subgroup_seg = []
        vq_losses = 0

        b, n, d, w, h = x.shape
        x = x.reshape(-1, 1, d, w, h)
        for down in self.down_layers:
            x = down(x)
            features.append(x)
        
        weight_all = (self.latent_fusion(s_all)*s_all).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        weight_sub = (self.latent_fusion(s_subgroup)*s_subgroup).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        for i in reversed(range(len(features))):
            f1, tsf_loss = self.tsf(features[i], s_all, weight_all, i, b, n)
            f2, _ = self.tsf(features[i], s_subgroup, weight_sub, i, b, n)

            if i==len(features)-1:
                z1 = self.conv_latent[i](f1)
                z2 = self.conv_latent[i](f2)
            else:
                f1 = torch.cat([x1, f1], dim=1)
                f2 = torch.cat([x2, f2], dim=1)
                z1 = self.conv_latent[i](f1)
                z2 = self.conv_latent[i](f2)

            zq1, vq_loss1, _ = self.quantize(z1)
            zq2, vq_loss2, _ = self.quantize(z2)
            zqs_all.append(zq1)
            zqs_subgroup.append(zq2)
            zs_all_seg.append(f1)
            zs_subgroup_seg.append(f2)
            vq_losses += (vq_loss1 + vq_loss2 + tsf_loss + torch.mean((z1.detach()-z2)**2) + torch.mean((z2.detach()-z1)**2))
            
            x1 = self.up_layers[i](zq1)
            x2 = self.up_layers[i](zq2)
        
        return zqs_all, zqs_subgroup, zs_all_seg, zs_subgroup_seg, vq_losses