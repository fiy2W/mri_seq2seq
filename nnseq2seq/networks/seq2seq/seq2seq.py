import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from nnseq2seq.networks.seq2seq.model2d.model import DualSeq2Seq as DualSeq2Seq2d
from nnseq2seq.networks.seq2seq.model3d.model import DualSeq2Seq as DualSeq2Seq3d


class Seq2Seq2d(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ndim = 2
        self.model = DualSeq2Seq2d(args)
        self.init()
    
    def init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x1, x2, domain_src, is_training=False):
        x1 = x1 * 2 - 1
        x2 = x2 * 2 - 1
        output_rec, seg, vq_loss = self.model(x1, x2, domain_src)
        output_rec = [rec * 0.5 + 0.5 for rec in output_rec] if self.model.deep_supervision else output_rec * 0.5 + 0.5
        if is_training:
            return output_rec, seg, vq_loss
        return output_rec, seg
    
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

        return output


class Seq2Seq3d(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ndim = 3
        self.model = DualSeq2Seq3d(args)
        self.init()
    
    def init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x1, x2, domain_src, is_training=False):
        x1 = x1 * 2 - 1
        x2 = x2 * 2 - 1
        output_rec, seg, vq_loss = self.model(x1, x2, domain_src)
        output_rec = [rec * 0.5 + 0.5 for rec in output_rec] if self.model.deep_supervision else output_rec * 0.5 + 0.5
        if is_training:
            return output_rec, seg, vq_loss
        return output_rec, seg
    
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

        return output