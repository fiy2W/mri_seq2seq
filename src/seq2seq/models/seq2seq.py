import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq.models.convlstm import ConvLSTM2d, ConvLSTM3d
from seq2seq.models.encoder import Encoder
from seq2seq.models.decoder import hyperDecoder


class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ndims = args['seq2seq']['ndims']
        assert self.ndims in [2, 3]
        ConvLSTM = ConvLSTM2d if self.ndims == 2 else ConvLSTM3d
        
        dim_out = args['seq2seq']['c_lstm']
        style_dim = args['seq2seq']['c_s']
        
        self.encoder = Encoder(args)
        self.decoder = hyperDecoder(args)
        
        self.enc_convlstm = ConvLSTM(
            input_dim=dim_out,
            hidden_dim=[dim_out, dim_out, dim_out],
            kernel_size=(3, 3) if self.ndims==2 else (3,3,3),
            num_layers=3,
            batch_first=True,
            bias=True,
            return_all_layers=False)
        
        self.dec_convlstm = ConvLSTM(
            input_dim=dim_out,
            hidden_dim=[dim_out, dim_out, dim_out],
            kernel_size=(3, 3) if self.ndims==2 else (3,3,3),
            num_layers=3,
            batch_first=True,
            bias=True,
            return_all_layers=False)
        
        self.style_fc = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None and hasattr(m.bias, 'data'):
                nn.init.constant_(m.bias.data, 0.0)
    
    def enc(self, x_5d):
        b, t = x_5d.shape[:2]
        img_shape = x_5d.shape[2:]
        x = x_5d.reshape((-1, *img_shape))

        x = self.encoder(x)
        
        f_shape = x.shape[1:]
        x = x.reshape((b, t, *f_shape))
        x = self.enc_convlstm(x)[0][0]
        x = torch.mean(x, dim=1, keepdim=True)
        return x
    
    def dec(self, x_5d_shape, x, s, n_outseq=1):
        s = self.style_fc(s)
        b, t = x_5d_shape[:2]
        img_shape = x_5d_shape[2:]
        
        f_shape = x.shape[2:]
        if self.ndims == 2:
            x = x.tile((1, n_outseq, 1, 1, 1))
        else:
            x = x.tile((1, n_outseq, 1, 1, 1, 1))
        x = self.dec_convlstm(x)[0][0]

        x = x.reshape((-1, *f_shape))
        x = self.decoder(x, s)
        
        return x.reshape((b, n_outseq, *img_shape))

    def forward(self, x_5d, s, n_outseq=1):
        x = self.enc(x_5d)
        y = self.dec(x_5d.shape, x, s, n_outseq=n_outseq)
        return y