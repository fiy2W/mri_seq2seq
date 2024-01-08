import torch
import torch.nn as nn

from seq2seq.models.resblock import ResnetBlock


class Encoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.ndims = args['seq2seq']['ndims']
        self.c_in = args['seq2seq']['c_in']
        self.c_enc = args['seq2seq']['c_enc']
        self.k_enc = args['seq2seq']['k_enc']
        self.s_enc = args['seq2seq']['s_enc']
        self.nres_enc = args['seq2seq']['nres_enc']

        norm = args['seq2seq']['norm']
        self.norm = getattr(nn, '%s%dd' % (norm, self.ndims)) if norm is not None else None
        Conv = getattr(nn, 'Conv%dd' % self.ndims)

        self.conv = nn.ModuleList()
        c_pre = self.c_in
        for c, k, s in zip(self.c_enc, self.k_enc, self.s_enc):
            block = [Conv(in_channels=c_pre, out_channels=c, kernel_size=k, padding=(k-1)//2, stride=s, padding_mode='reflect')]
            if self.norm is not None:
                block.append(self.norm(c))
            block.append(nn.LeakyReLU(0.2, True))
            self.conv.append(nn.Sequential(*block))
            c_pre = c

        res = [
            ResnetBlock(c_pre, padding_type='reflect', norm_layer=self.norm, use_dropout=False, use_bias=True, ndims=self.ndims)
            for _ in range(self.nres_enc)
        ]
        self.res = nn.Sequential(*res)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        x = self.res(x)
        return x