import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from nnseq2seq.networks.seq2seq.model2d.encoder import ImageEncoder as ImageEncoder2d
from nnseq2seq.networks.seq2seq.model2d.decoder import HyperImageDecoder as HyperImageDecoder2d
from nnseq2seq.networks.seq2seq.model2d.segmentor import Segmentor as Segmentor2d
from nnseq2seq.networks.seq2seq.model2d.discriminator import NLayerDiscriminator as NLayerDiscriminator2d

from nnseq2seq.networks.seq2seq.model3d.encoder import ImageEncoder as ImageEncoder3d
from nnseq2seq.networks.seq2seq.model3d.decoder import HyperImageDecoder as HyperImageDecoder3d
from nnseq2seq.networks.seq2seq.model3d.segmentor import Segmentor as Segmentor3d
from nnseq2seq.networks.seq2seq.model3d.discriminator import NLayerDiscriminator as NLayerDiscriminator3d


class Seq2Seq2d(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ndim = 2
        self.image_encoder = ImageEncoder2d(args['image_encoder'])
        self.hyper_decoder = HyperImageDecoder2d(args['image_decoder'])
        self.segmentor = Segmentor2d(args['segmentor'])
        self.discriminator = NLayerDiscriminator2d(args['discriminator'])

        self.init()
    
    def forward(self, x_src, domain_src_all, domain_src_subgroup, domain_tgt, with_latent=False, latent_focal=None):
        z_all, z_subgroup, z_all_seg, z_subgroup_seg, vq_loss = self.image_encoder(x_src, domain_src_all, domain_src_subgroup)
        outputs_all, fusion_all = self.hyper_decoder(z_all, domain_tgt, latent_focal=latent_focal)
        outputs_subgroup, fusion_subgroup = self.hyper_decoder(z_subgroup, domain_tgt, latent_focal=latent_focal)
        if with_latent:
            return outputs_all, outputs_subgroup, z_all, z_subgroup, z_all_seg, z_subgroup_seg, vq_loss, fusion_all, fusion_subgroup
        else:
            return outputs_all, outputs_subgroup, fusion_all, fusion_subgroup
    
    def infer(self, x_src, domain_src_all, domain_src_subgroup, domain_tgt):
        z_all, z_subgroup, z_all_seg, z_subgroup_seg, _ = self.image_encoder(x_src, domain_src_all, domain_src_subgroup)
        outputs_subgroup, _ = self.hyper_decoder(z_subgroup, domain_tgt, latent_focal='dispersion' if self.hyper_decoder.focal_mode=='focal_mix' else None)
        fusion_all = self.hyper_decoder.image_fusion(z_all[-1].detach())
        return outputs_subgroup, z_all, z_all_seg, fusion_all
    
    def init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def compute_conv_feature_map_size(self, input_size):
        output = self.compute_single_size(input_size)
        
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

        num_input_channels = self.image_encoder.style_dim
        strides = self.image_encoder.s_enc
        channels = self.image_encoder.c_enc
        nres = self.image_encoder.n_res
        zdim = self.image_encoder.latent_space_dim
        up_scale = 1
        for i, (s, c, n) in enumerate(zip(strides, channels, nres)):
            up_scale = up_scale*s
            input_size = [insize/s for insize in input_size]
            feature_map_layer = np.prod([c, *input_size], dtype=np.int64)
            output += (feature_map_layer*(1+5*n))
            if up_scale>=4:
                attention_map = np.prod([*input_size, *input_size], dtype=np.int64)
                output += (attention_map + 5*feature_map_layer)
            output += np.prod([zdim, *input_size], dtype=np.int64)*(1+1)
        output = output * num_input_channels

        rep = 6 # 2 dec + 2 seg + 2 dec
        strides = self.hyper_decoder.s_enc
        channels = self.hyper_decoder.c_enc
        nres = self.hyper_decoder.n_res
        for i, (s, c, n) in enumerate(zip(strides, channels, nres)):
            input_size = [insize/s for insize in input_size]
            feature_map_layer = np.prod([c, *input_size], dtype=np.int64)
            output += (feature_map_layer*(1+5*n) + np.prod(input_size, dtype=np.int64))*rep
            if up_scale>=4:
                attention_map = np.prod([*input_size, *input_size], dtype=np.int64)
                output += (attention_map + 5*feature_map_layer)*rep
            up_scale = up_scale//s

        return output


class Seq2Seq3d(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ndim = 3
        self.image_encoder = ImageEncoder3d(args['image_encoder'])
        self.hyper_decoder = HyperImageDecoder3d(args['image_decoder'])
        self.segmentor = Segmentor3d(args['segmentor'])
        self.discriminator = NLayerDiscriminator3d(args['discriminator'])
        
        self.init()
    
    def forward(self, x_src, domain_src_all, domain_src_subgroup, domain_tgt, with_latent=False, latent_focal=None):
        z_all, z_subgroup, z_all_seg, z_subgroup_seg, vq_loss = self.image_encoder(x_src, domain_src_all, domain_src_subgroup)
        outputs_all, fusion_all = self.hyper_decoder(z_all, domain_tgt, latent_focal=latent_focal)
        outputs_subgroup, fusion_subgroup = self.hyper_decoder(z_subgroup, domain_tgt, latent_focal=latent_focal)
        if with_latent:
            return outputs_all, outputs_subgroup, z_all, z_subgroup, z_all_seg, z_subgroup_seg, vq_loss, fusion_all, fusion_subgroup
        else:
            return outputs_all, outputs_subgroup, fusion_all, fusion_subgroup
    
    def infer(self, x_src, domain_src_all, domain_src_subgroup, domain_tgt):
        z_all, z_subgroup, z_all_seg, z_subgroup_seg, _ = self.image_encoder(x_src, domain_src_all, domain_src_subgroup)
        outputs_subgroup, _ = self.hyper_decoder(z_subgroup, domain_tgt, latent_focal='dispersion' if self.hyper_decoder.focal_mode=='focal_mix' else None)
        fusion_all = self.hyper_decoder.image_fusion(z_all[-1].detach())
        return outputs_subgroup, z_all, z_all_seg, fusion_all
    
    def init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def compute_conv_feature_map_size(self, input_size):
        output = self.compute_single_size(input_size)
        
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

        num_input_channels = self.image_encoder.style_dim
        strides = self.image_encoder.s_enc
        channels = self.image_encoder.c_enc
        nres = self.image_encoder.n_res
        zdim = self.image_encoder.latent_space_dim
        up_scale = 1
        for i, (s, c, n) in enumerate(zip(strides, channels, nres)):
            up_scale = up_scale*s
            input_size = [insize/s for insize in input_size]
            feature_map_layer = np.prod([c, *input_size], dtype=np.int64)
            output += (feature_map_layer*(1+5*n))
            if up_scale>=4:
                attention_map = np.prod([*input_size, *input_size], dtype=np.int64)
                output += (attention_map + 5*feature_map_layer)
            output += np.prod([zdim, *input_size], dtype=np.int64)*(1+1)
        output = output * num_input_channels

        rep = 6 # 2 dec + 2 seg + 2 dec
        strides = self.hyper_decoder.s_enc
        channels = self.hyper_decoder.c_enc
        nres = self.hyper_decoder.n_res
        for i, (s, c, n) in enumerate(zip(strides, channels, nres)):
            input_size = [insize/s for insize in input_size]
            feature_map_layer = np.prod([c, *input_size], dtype=np.int64)
            output += (feature_map_layer*(1+5*n) + np.prod(input_size, dtype=np.int64))*rep
            if up_scale>=4:
                attention_map = np.prod([*input_size, *input_size], dtype=np.int64)
                output += (attention_map + 5*feature_map_layer)*rep
            up_scale = up_scale//s

        return output