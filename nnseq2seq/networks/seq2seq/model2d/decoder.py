import torch
import torch.nn as nn
import torch.nn.functional as F

from nnseq2seq.networks.seq2seq.model2d.convnext import Block, LayerNorm, ResBlock, hyperResBlock, hyperAttnResBlock


class HyperImageDecoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        
        self.c_out = args['out_channels']
        self.c_enc = args['conv_channels']
        self.k_enc = args['conv_kernel']
        self.s_enc = args['conv_stride']
        self.d_enc = args['conv_down']
        self.n_res = args['resblock_n']
        self.k_res = args['resblock_kernel']
        self.p_res = args['resblock_padding']
        self.layer_scale_init_value = args['layer_scale_init_value']
        self.hyper_dim = args['hyper_conv_dim']
        self.latent_space_dim = args['latent_space_dim']
        self.style_dim = args['style_dim']
        self.deep_supervision = args['deep_supervision']

        self.down_layers = nn.ModuleList()
        self.midconv_layers = nn.ModuleList()
        self.midres_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.deep_layers = nn.ModuleList()
        for i, (ce, ke, se, de, nr, kr, pr) in enumerate(zip(self.c_enc, self.k_enc, self.s_enc, self.d_enc, self.n_res, self.k_res, self.p_res)):
            self.down_layers.append(nn.Sequential(
                nn.Conv2d(in_channels=self.latent_space_dim, out_channels=ce, kernel_size=de, padding=0, stride=de),
            ))
            if i==0:
                self.midconv_layers.append(nn.Identity())
            else:
                self.midconv_layers.append(nn.Sequential(
                    nn.Conv2d(c_pre+ce, out_channels=ce, kernel_size=1, padding=0, stride=1),
                ))
            self.midres_layers.append(
                #hyperResBlock(ce, self.style_dim, nr, self.hyper_dim, kr, pr, layer_scale_init_value=self.layer_scale_init_value)
                hyperAttnResBlock(ce, self.style_dim, nr, self.hyper_dim, kr, pr, layer_scale_init_value=self.layer_scale_init_value, use_attn=True)
                )
            self.up_layers.append(nn.Sequential(
                    #nn.ConvTranspose2d(ce, out_channels=ce, kernel_size=ke, padding=se//2, stride=se),
                    nn.Upsample(scale_factor=se, mode='bilinear', align_corners=True),
                ))
            self.deep_layers.append(nn.Sequential(
                nn.Conv2d(ce, out_channels=self.c_out, kernel_size=3, padding=1, stride=1, padding_mode='zeros'),
                nn.LeakyReLU(0.01, inplace=True),
            ))
            c_pre = ce
        
        self.up_out = nn.Sequential(
            ResBlock(c_pre, n_layer=1, kernel_size=3, padding=1, layer_scale_init_value=self.layer_scale_init_value),
            #nn.ConvTranspose2d(c_pre, out_channels=c_pre//2, kernel_size=4, padding=1, stride=2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(c_pre, out_channels=c_pre//2, kernel_size=3, padding=1, stride=1, padding_mode='zeros'),
            LayerNorm(c_pre//2, eps=1e-6, data_format="channels_first"),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(c_pre//2, out_channels=self.c_out, kernel_size=3, padding=1, stride=1, padding_mode='zeros'),
            nn.LeakyReLU(0.01, inplace=True),
        )
            
    def forward(self, z, s):
        outputs = []
        for i, (down, midc, midr, up, deep) in enumerate(zip(self.down_layers, self.midconv_layers, self.midres_layers, self.up_layers, self.deep_layers)):
            x = down(z)
            if i!=0:
                x = torch.cat([x, x_pre], dim=1)
            x = midc(x)
            x = midr(x, s)
            x_pre = up(x)
            outputs.append(deep(x_pre))
        x = self.up_out(x_pre)
        outputs.append(x)

        outputs = outputs[::-1]

        if not self.deep_supervision:
            return outputs[0]
        else:
            return outputs