import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from nnseq2seq.networks.seq2seq.model2d.encoder import ImageEncoder as ImageEncoder2d
from nnseq2seq.networks.seq2seq.model2d.decoder import HyperImageDecoder as HyperImageDecoder2d
from nnseq2seq.networks.seq2seq.model2d.tsf import TSF_attention as TSF_attention2d
from nnseq2seq.networks.seq2seq.model2d.segmentor import Segmentor as Segmentor2d
from nnseq2seq.networks.seq2seq.model2d.discriminator import NLayerDiscriminator as NLayerDiscriminator2d

from nnseq2seq.networks.seq2seq.model3d.encoder import ImageEncoder as ImageEncoder3d
from nnseq2seq.networks.seq2seq.model3d.decoder import HyperImageDecoder as HyperImageDecoder3d
from nnseq2seq.networks.seq2seq.model3d.tsf import TSF_attention as TSF_attention3d
from nnseq2seq.networks.seq2seq.model3d.segmentor import Segmentor as Segmentor3d
from nnseq2seq.networks.seq2seq.model3d.discriminator import NLayerDiscriminator as NLayerDiscriminator3d

from nnseq2seq.training.loss.contrastive_loss import SupConLoss


class Seq2Seq2d(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.image_encoder = ImageEncoder2d(args['image_encoder'])
        self.hyper_decoder = HyperImageDecoder2d(args['image_decoder'])
        self.tsf = TSF_attention2d(args['image_decoder'])
        self.segmentor = Segmentor2d(args['segmentor'])
        self.discriminator = NLayerDiscriminator2d(args['discriminator'])

        self.init()
    
    def forward(self, x_src, domain_tgt, with_latent=False):
        z, vq_loss = self.image_encoder(x_src)
        outputs = self.hyper_decoder(z, domain_tgt)
        if with_latent:
            return outputs, z, vq_loss
        else:
            return outputs
    
    def init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def reparameterization(self, zqs):
        zqs = torch.stack(zqs, dim=1)
        zq_mean = torch.mean(zqs, dim=1, keepdim=True)
        zq_std = torch.mean(torch.square(zqs-zq_mean), dim=1, keepdim=True) / (zqs.shape[1] - 1) * zqs.shape[1]
        zq_rand = torch.randn_like(zq_mean) * zq_std + zq_mean
        return zq_rand[:,0]
    
    def compute_conv_feature_map_size(self, input_size):
        output = self.compute_single_size(input_size) * 3

        # discriminator
        for c in self.discriminator.c_enc:
            input_size = [insize/2 for insize in input_size]
            output += (np.prod([c, *input_size], dtype=np.int64)*2)
        output += (np.prod([c, *input_size], dtype=np.int64)*(2+5*2) + \
            (np.prod([*input_size, *input_size], dtype=np.int64) + np.prod([c, *input_size], dtype=np.int64)*5) + \
            np.prod([1, *input_size], dtype=np.int64))
        
        # perceptual
        for s, c, n in zip(
            [2, 2, 2, 2, 2],
            [64, 128, 256, 512, 512],
            [3, 3, 5, 5, 5]):
            output += (np.prod([c, *input_size], dtype=np.int64)*n)
            input_size = [insize/s for insize in input_size]

        return output
    
    def compute_single_size(self, input_size):
        output = np.prod(input_size, dtype=np.int64)

        strides = self.image_encoder.s_enc
        channels = self.image_encoder.c_enc
        nres = self.image_encoder.n_res
        zdim = self.image_encoder.latent_space_dim
        for i, (s, c, n) in enumerate(zip(strides, channels, nres)):
            if i==0:
                n = n + 2
            input_size = [insize/s for insize in input_size]
            feature_map_layer = np.prod([c, *input_size], dtype=np.int64)
            attention_map = np.prod([*input_size, *input_size], dtype=np.int64)
            if i==0:
                z_size = input_size
                feature_map_layer_z = feature_map_layer
            output += (feature_map_layer*(2+5*n)+(attention_map+feature_map_layer*5)*np.max([0, n-1]))
        output += (feature_map_layer_z*(len(strides)-1))
        output += np.prod([zdim, *z_size], dtype=np.int64)

        down = self.hyper_decoder.d_enc
        strides = self.hyper_decoder.s_enc
        channels = self.hyper_decoder.c_enc
        nres = self.hyper_decoder.n_res
        for i, (d, s, c, n) in enumerate(zip(down, strides, channels, nres)):
            d_size = [zs/d for zs in z_size]
            u_size = [ds*s for ds in d_size]
            output += (np.prod([c, *d_size], dtype=np.int64)*(2+5*n) + \
                (np.prod([*d_size, *d_size], dtype=np.int64) + np.prod([c, *d_size], dtype=np.int64)*5)*np.max([0, n-1]) + \
                np.prod([c, *u_size], dtype=np.int64) + \
                np.prod(u_size, dtype=np.int64))
        output += (np.prod([c, *u_size], dtype=np.int64)*5)
        u_size = [us*2 for us in u_size]
        output += (np.prod([c+c/2+c/2+1, *u_size], dtype=np.int64))
        return output
    
    def contrastive_loss(self, z1, z2, mask):
        contrastive = SupConLoss()

        b = z1.shape[0]

        z1 = z1 * mask
        z2 = z2 * mask
        
        p_z1 = torch.flatten(self.image_encoder.p(z1), 2, 3).permute(0,2,1) # b, wh, c
        p_z2 = torch.flatten(self.image_encoder.p(z2), 2, 3).permute(0,2,1) # b, wh, c

        p_z1 = p_z1 / p_z1.norm(dim=2, keepdim=True)
        p_z2 = p_z2 / p_z2.norm(dim=2, keepdim=True)

        A = torch.sum(torch.softmax(torch.bmm(p_z1, p_z1.permute(0,2,1)), dim=1), dim=2)
        
        loss_contrast = 0
        for bi in range(b):
            _, index = torch.sort(A[bi])
            index = index[:index.shape[0]//4]
            sp_z1 = p_z1[bi][index]
            sp_z2 = p_z2[bi][index]

            loss_contrast += contrastive(torch.stack([sp_z1, sp_z2], dim=1))
            loss_contrast += contrastive(torch.stack([sp_z2, sp_z1], dim=1))
        loss_contrast = loss_contrast/b
        return loss_contrast


class Seq2Seq3d(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.image_encoder = ImageEncoder3d(args['image_encoder'])
        self.hyper_decoder = HyperImageDecoder3d(args['image_decoder'])
        self.tsf = TSF_attention3d(args['image_decoder'])
        self.segmentor = Segmentor3d(args['segmentor'])
        self.discriminator = NLayerDiscriminator3d(args['discriminator'])
        
        self.init()
    
    def forward(self, x_src, domain_tgt, with_latent=False):
        z, vq_loss = self.image_encoder(x_src)
        outputs = self.hyper_decoder(z, domain_tgt)
        if with_latent:
            return outputs, z, vq_loss
        else:
            return outputs
    
    def init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def reparameterization(self, zqs):
        zqs = torch.stack(zqs, dim=1)
        zq_mean = torch.mean(zqs, dim=1, keepdim=True)
        zq_std = torch.mean(torch.square(zqs-zq_mean), dim=1, keepdim=True) / (zqs.shape[1] - 1) * zqs.shape[1]
        zq_rand = torch.randn_like(zq_mean) * zq_std + zq_mean
        return zq_rand[:,0]
    
    def compute_conv_feature_map_size(self, input_size):
        output = self.compute_single_size(input_size) * 3

        # discriminator
        for c in self.discriminator.c_enc:
            input_size = [insize/2 for insize in input_size]
            output += (np.prod([c, *input_size], dtype=np.int64)*2)
        output += (np.prod([c, *input_size], dtype=np.int64)*(2+5*2) + \
            (np.prod([*input_size, *input_size], dtype=np.int64) + np.prod([c, *input_size], dtype=np.int64)*5) + \
            np.prod([1, *input_size], dtype=np.int64))
        
        # perceptual
        for input_size in [
                input_size[1:],
                [input_size[0], input_size[2]],
                input_size[:2]
            ]:
            for s, c, n in zip(
                [2, 2, 2, 2, 2],
                [64, 128, 256, 512, 512],
                [3, 3, 5, 5, 5]):
                output += (np.prod([c, *input_size], dtype=np.int64)*n)
                input_size = [insize/s for insize in input_size]

        return output
    
    def compute_single_size(self, input_size):
        output = np.prod(input_size, dtype=np.int64)

        strides = self.image_encoder.s_enc
        channels = self.image_encoder.c_enc
        nres = self.image_encoder.n_res
        zdim = self.image_encoder.latent_space_dim
        for i, (s, c, n) in enumerate(zip(strides, channels, nres)):
            if i==0:
                n = n + 2
            input_size = [insize/s for insize in input_size]
            feature_map_layer = np.prod([c, *input_size], dtype=np.int64)
            attention_map = np.prod([*input_size, *input_size], dtype=np.int64)
            if i==0:
                z_size = input_size
                feature_map_layer_z = feature_map_layer
            output += (feature_map_layer*(2+5*n)+(attention_map+feature_map_layer*5)*np.max([0, n-1]))
        output += (feature_map_layer_z*(len(strides)-1))
        output += np.prod([zdim, *z_size], dtype=np.int64)

        down = self.hyper_decoder.d_enc
        strides = self.hyper_decoder.s_enc
        channels = self.hyper_decoder.c_enc
        nres = self.hyper_decoder.n_res
        for i, (d, s, c, n) in enumerate(zip(down, strides, channels, nres)):
            d_size = [zs/d for zs in z_size]
            u_size = [ds*s for ds in d_size]
            output += (np.prod([c, *d_size], dtype=np.int64)*(2+5*n) + \
                (np.prod([*d_size, *d_size], dtype=np.int64) + np.prod([c, *d_size], dtype=np.int64)*5)*np.max([0, n-1]) + \
                np.prod([c, *u_size], dtype=np.int64) + \
                np.prod(u_size, dtype=np.int64))
        output += (np.prod([c, *u_size], dtype=np.int64)*3)
        u_size = [us*2 for us in u_size]
        output += (np.prod([c+c/2+c/2+1, *u_size], dtype=np.int64))
        return output
    
    def contrastive_loss(self, z1, z2, mask):
        contrastive = SupConLoss()

        b = z1.shape[0]

        z1 = z1 * mask
        z2 = z2 * mask
        
        p_z1 = torch.flatten(self.image_encoder.p(z1), 2, 4).permute(0,2,1)
        p_z2 = torch.flatten(self.image_encoder.p(z2), 2, 4).permute(0,2,1)
        
        p_z1 = p_z1 / p_z1.norm(dim=2, keepdim=True)
        p_z2 = p_z2 / p_z2.norm(dim=2, keepdim=True)

        A = torch.sum(torch.softmax(torch.bmm(p_z1, p_z1.permute(0,2,1)), dim=1), dim=2)
        
        loss_contrast = 0
        for bi in range(b):
            _, index = torch.sort(A[bi])
            index = index[:index.shape[0]//8]
            sp_z1 = p_z1[bi][index]
            sp_z2 = p_z2[bi][index]

            loss_contrast += contrastive(torch.stack([sp_z1, sp_z2], dim=1))
            loss_contrast += contrastive(torch.stack([sp_z2, sp_z1], dim=1))
        loss_contrast = loss_contrast/b
        return loss_contrast