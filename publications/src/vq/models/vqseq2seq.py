import torch
import torch.nn as nn
import torch.nn.functional as F

from src.seq2seq.models.encoder import Encoder
from src.seq2seq.models.decoder import hyperDecoder
from src.vq.models.quantize import VectorQuantizer2 as VectorQuantizer
from src.vq.loss import SupConLoss


class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ndims = args['seq2seq']['ndims']
        assert self.ndims in [2]
        
        style_dim = args['seq2seq']['c_s']
        self.c_dim = args['seq2seq']['c_enc'][-1]
        self.embed_dim = args['seq2seq']['embed_dim']
        self.n_embed = args['seq2seq']['n_embed']
        self.beta = args['seq2seq']['beta']
        
        self.encoder = Encoder(args)
        self.decoder = hyperDecoder(args)
        
        self.style_fc = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.quantize_conv = nn.Conv2d(in_channels=self.c_dim, out_channels=self.embed_dim, kernel_size=1)
        self.quantize = VectorQuantizer(self.n_embed, self.embed_dim, beta=self.beta)
        self.quantize_deconv = nn.Conv2d(in_channels=self.embed_dim, out_channels=self.c_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None and hasattr(m.bias, 'data'):
                nn.init.constant_(m.bias.data, 0.0)
    
    def enc(self, x):
        x = self.encoder(x)
        z = self.quantize_conv(x)
        zq, vq_loss, vq_info = self.quantize(z)
        return zq, vq_loss, vq_info
    
    def dec(self, zq, s):
        s = self.style_fc(s)
        x = self.quantize_deconv(zq)
        x = self.decoder(x, s)
        return x

    def forward_single(self, x, s):
        zq, vq_loss, vq_info = self.enc(x)
        y = self.dec(zq, s)
        return y, zq, vq_loss, vq_info
    
    def forward(self, x_src, x_tgt, s_src, s_tgt, z_sample=False):
        rec_src2tgt, zq_src, vq_losses_src, _ = self.forward_single(x_src, s_tgt)
        rec_tgt2src, zq_tgt, vq_losses_tgt, _ = self.forward_single(x_tgt, s_src)

        rec_src2src = self.dec(zq_src, s_src)
        rec_tgt2tgt = self.dec(zq_tgt, s_tgt)

        if z_sample:
            zq_mean = (zq_src + zq_tgt) / 2
            zq_std = (zq_src-zq_mean)**2 + (zq_tgt-zq_mean)**2
            zq_rand = torch.randn_like(zq_mean) * zq_std + zq_mean

            rec_rand2src = self.dec(zq_rand, s_src)
            rec_rand2tgt = self.dec(zq_rand, s_tgt)

            return rec_src2tgt, rec_tgt2src, rec_src2src, rec_tgt2tgt, rec_rand2src, rec_rand2tgt, zq_src, zq_tgt, vq_losses_src + vq_losses_tgt
        else:
            return rec_src2tgt, rec_tgt2src, rec_src2src, rec_tgt2tgt, zq_src, zq_tgt, vq_losses_src + vq_losses_tgt
    
    def con_loss(self, c1, c2, img1, img2, downsample=4):
        contrastive_loss = SupConLoss()
        mask1 = (((img1<=-1)+(img2<=-1))<0.5).to(device=img1.device, dtype=torch.float32).reshape(-1, *img1.shape[1:]).mean(dim=1, keepdim=True)
        mask1 = F.interpolate(mask1, scale_factor=1/downsample, mode='nearest')>0.5

        loss_mse = (nn.MSELoss()(c1*mask1, c2.detach()*mask1) + nn.MSELoss()(c1.detach()*mask1, c2*mask1))
        
        bs, c = c1.shape[0:2]
        c1 = c1 / c1.norm(dim=1, keepdim=True)
        c2 = c2 / c2.norm(dim=1, keepdim=True)

        c1 = c1.permute(0,2,3,1).reshape(bs, -1, c)
        c2 = c2.permute(0,2,3,1).reshape(bs, -1, c)
        mask1 = mask1.permute(0,2,3,1).reshape(bs, -1, 1)

        A = torch.sum(torch.softmax(torch.bmm(c1*mask1, (c1*mask1).permute(0,2,1)), dim=1), dim=2)
        
        loss_contrast = 0
        for bi in range(bs):
            _, index = torch.sort(A[bi])
            index = index[:index.shape[0]//8]
            sp_z1 = c1[bi][index]
            sp_z2 = c2[bi][index]

            loss_contrast += contrastive_loss(torch.stack([sp_z1, sp_z2], dim=1))
            loss_contrast += contrastive_loss(torch.stack([sp_z2, sp_z1], dim=1))
        loss_contrast = loss_contrast/bs
        return loss_mse, loss_contrast/bs