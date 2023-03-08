import torch
import torch.nn as nn

from models.seq2seq.hyperconv import hyperConv
from models.seq2seq.resblock import hyperResnetBlock


class hyperDecoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.ndims = args['seq2seq']['ndims']
        self.c_dec = args['seq2seq']['c_dec']
        self.k_dec = args['seq2seq']['k_dec']
        self.s_dec = args['seq2seq']['s_dec']
        self.nres_dec = args['seq2seq']['nres_dec']
        self.style_dim = args['seq2seq']['c_s']
        self.weight_dim = args['seq2seq']['c_w']

        norm = args['seq2seq']['norm']
        self.norm = getattr(nn, '%s%dd' % (norm, self.ndims)) if norm is not None else None
        ReflectionPad = getattr(nn, 'ReflectionPad%dd' % self.ndims)
        
        c_pre = args['seq2seq']['c_enc'][-1]
        self.res = nn.ModuleList()
        for _ in range(self.nres_dec):
            self.res.append(hyperResnetBlock(self.style_dim, c_pre, padding_type='reflect', norm_layer=self.norm, use_bias=True, weight_dim=self.weight_dim, ndims=self.ndims))
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear' if self.ndims==2 else 'trilinear')

        self.pads = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for c, k, s in zip(self.c_dec[:-1], self.k_dec[:-1], self.s_dec[:-1]):
            self.pads.append(ReflectionPad((k-1)//2))
            self.convs.append(hyperConv(self.style_dim, c_pre, c, ksize=k, stride=s, padding=0, weight_dim=self.weight_dim, ndims=self.ndims))
            if self.norm is not None:
                self.norms.append(nn.Sequential(self.norm(c), nn.LeakyReLU(0.2, True)))
            else:
                self.norms.append(nn.LeakyReLU(0.2, True))
            c_pre = c

        self.pad_last = ReflectionPad((self.k_dec[-1]-1)//2)
        self.conv_last = hyperConv(self.style_dim, dim_in=c_pre, dim_out=self.c_dec[-1], ksize=self.k_dec[-1], padding=0, weight_dim=self.weight_dim, ndims=self.ndims)
        self.act_last = nn.Tanh()

    def forward(self, x, s):
        for res in self.res:
            x = res(x, s)
        
        for pad, conv, norm in zip(self.pads, self.convs, self.norms):
            x = self.up(x)
            x = norm(conv(pad(x), s))
        
        x = self.act_last(self.conv_last(self.pad_last(x), s))
        return x