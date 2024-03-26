import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from nnseq2seq.networks.seq2seq.model2d.encoder import ImageEncoder as ImageEncoder2d
from nnseq2seq.networks.seq2seq.model2d.decoder import HyperImageDecoder as HyperImageDecoder2d

from nnseq2seq.networks.seq2seq.model3d.encoder import ImageEncoder as ImageEncoder3d
from nnseq2seq.networks.seq2seq.model3d.decoder import HyperImageDecoder as HyperImageDecoder3d


class Seq2Seq2d(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.image_encoder = ImageEncoder2d(args['image_encoder'])
        self.hyper_decoder = HyperImageDecoder2d(args['image_decoder'])

        self.init()
    
    def forward(self, x_src, domain_tgt):
        z = self.image_encoder(x_src)
        outputs = self.hyper_decoder(z, domain_tgt)
        return outputs
    
    def init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def compute_conv_feature_map_size(self, input_size):
        output = self.compute_single_size(input_size) * 2
        
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
            input_size = [insize/s for insize in input_size]
            feature_map_layer = np.prod([c, *input_size], dtype=np.int64)
            if i==0:
                z_size = input_size
                feature_map_layer_z = feature_map_layer
            output += (feature_map_layer*(2+5*n))
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
                np.prod([c, *u_size], dtype=np.int64) + \
                np.prod(u_size, dtype=np.int64))
        output += (np.prod([c, *u_size], dtype=np.int64)*5)
        u_size = [us*2 for us in u_size]
        output += (np.prod([c+c/2+c/2+1, *u_size], dtype=np.int64))
        return output


class Seq2Seq3d(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.image_encoder = ImageEncoder3d(args['image_encoder'])
        self.hyper_decoder = HyperImageDecoder3d(args['image_decoder'])

        self.init()
    
    def forward(self, x_src, domain_tgt):
        z = self.image_encoder(x_src)
        outputs = self.hyper_decoder(z, domain_tgt)
        return outputs
    
    def init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def compute_conv_feature_map_size(self, input_size):
        output = self.compute_single_size(input_size) * 2
        
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
            input_size = [insize/s for insize in input_size]
            feature_map_layer = np.prod([c, *input_size], dtype=np.int64)
            if i==0:
                z_size = input_size
                feature_map_layer_z = feature_map_layer
            output += (feature_map_layer*(2+5*n))
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
                np.prod([c, *u_size], dtype=np.int64) + \
                np.prod(u_size, dtype=np.int64))
        output += (np.prod([c, *u_size], dtype=np.int64)*3)
        u_size = [us*2 for us in u_size]
        output += (np.prod([c+c/2+c/2+1, *u_size], dtype=np.int64))
        return output