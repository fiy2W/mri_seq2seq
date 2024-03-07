import torch
import torch.nn as nn

from src.seq2seq.models.hyperconv import hyperConv


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, ndims=2):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.ndims = ndims
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (LeakyReLU))
        """
        ReflectionPad = getattr(nn, 'ReflectionPad%dd' % self.ndims)
        ReplicationPad = getattr(nn, 'ReplicationPad%dd' % self.ndims)
        Conv = getattr(nn, 'Conv%dd' % self.ndims)

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [ReflectionPad(1)]
        elif padding_type == 'replicate':
            conv_block += [ReplicationPad(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        if norm_layer is not None:
            conv_block += [Conv(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.LeakyReLU(0.2, True)]
        else:
            conv_block += [Conv(dim, dim, kernel_size=3, padding=p, bias=use_bias), nn.LeakyReLU(0.2, True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [ReflectionPad(1)]
        elif padding_type == 'replicate':
            conv_block += [ReplicationPad(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [Conv(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        if norm_layer is not None:
            conv_block += [norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class hyperResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, style_dim, dim, padding_type, norm_layer, use_bias, weight_dim=8, ndims=2):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(hyperResnetBlock, self).__init__()
        self.ndims = ndims
        ReflectionPad = getattr(nn, 'ReflectionPad%dd' % self.ndims)
        ReplicationPad = getattr(nn, 'ReplicationPad%dd' % self.ndims)

        p = 0
        if padding_type == 'reflect':
            self.pad1 = ReflectionPad(1)
        elif padding_type == 'replicate':
            self.pad1 = ReplicationPad(1)
        elif padding_type == 'zero':
            self.pad1 = nn.Identity()
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        self.conv1 = hyperConv(style_dim, dim, dim, ksize=3, padding=p, bias=use_bias, weight_dim=weight_dim, ndims=self.ndims)
        if norm_layer is not None:
            self.norm1 = nn.Sequential(norm_layer(dim), nn.LeakyReLU(0.2, True))
        else:
            self.norm1 = nn.LeakyReLU(0.2, True)

        p = 0
        if padding_type == 'reflect':
            self.pad2 = ReflectionPad(1)
        elif padding_type == 'replicate':
            self.pad2 = ReplicationPad(1)
        elif padding_type == 'zero':
            self.pad2 = nn.Identity()
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        self.conv2 = hyperConv(style_dim, dim, dim, ksize=3, padding=p, bias=use_bias, weight_dim=weight_dim, ndims=self.ndims)
        if norm_layer is not None:
            self.norm2 = norm_layer(dim)
        else:
            self.norm2 = nn.Identity()

    def forward(self, x, s):
        """Forward function (with skip connections)"""
        y = self.norm1(self.conv1(self.pad1(x), s))
        y = self.norm2(self.conv2(self.pad2(y), s))
        out = x + y  # add skip connections
        return out