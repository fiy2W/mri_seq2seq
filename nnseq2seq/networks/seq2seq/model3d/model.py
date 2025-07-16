from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed
from timm.layers import Format, _assert

from nnseq2seq.networks.seq2seq.hyperconv import hyperConv
from nnseq2seq.networks.seq2seq.model3d.quantize import SoftVectorQuantizer


class DualSeq2Seq(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        
        self.c_in = args['in_channels']
        self.c_enc = args['conv_channels']
        self.group = self.c_enc[0] // 2
        self.num_classes = args['num_classes']
        self.hyper_dim = args['hyper_conv_dim']
        self.style_dim = args['style_dim']
        self.deep_supervision = args['deep_supervision']

        
        self.encoder = Encoder(self.c_in, self.c_enc, self.group, self.style_dim, self.hyper_dim)

        self.syn_output_conv_x4 = nn.Sequential(
            nn.GroupNorm(num_groups=self.group, num_channels=self.c_enc[2], affine=True),
            nn.SiLU(),
            nn.Conv3d(in_channels=self.c_enc[2], out_channels=self.c_in, kernel_size=3, padding=1, stride=1)
        )
        self.seg_output_conv_x4 = nn.Sequential(
            nn.GroupNorm(num_groups=self.group, num_channels=self.c_enc[2], affine=True),
            nn.SiLU(),
            nn.Conv3d(in_channels=self.c_enc[2], out_channels=self.num_classes, kernel_size=3, padding=1, stride=1)
        )

        self.up_31 = nn.Sequential(
            ResBlock(self.c_enc[2]+self.c_enc[1], self.c_enc[1], stride=1, conv_shortcut=False, groups=self.group, up=False, down=False),
            ResBlock(self.c_enc[1], self.c_enc[1], stride=1, conv_shortcut=False, groups=self.group, up=False, down=False),
        )
        self.up_32 = nn.Sequential(
            ResBlock(self.c_enc[2]+self.c_enc[1], self.c_enc[1], stride=1, conv_shortcut=False, groups=self.group, up=False, down=False),
            ResBlock(self.c_enc[1], self.c_enc[1], stride=1, conv_shortcut=False, groups=self.group, up=False, down=False),
        )

        self.syn_output_conv_x2 = nn.Sequential(
            nn.GroupNorm(num_groups=self.group, num_channels=self.c_enc[1], affine=True),
            nn.SiLU(),
            nn.Conv3d(in_channels=self.c_enc[1], out_channels=self.c_in, kernel_size=3, padding=1, stride=1)
        )
        self.seg_output_conv_x2 = nn.Sequential(
            nn.GroupNorm(num_groups=self.group, num_channels=self.c_enc[1], affine=True),
            nn.SiLU(),
            nn.Conv3d(in_channels=self.c_enc[1], out_channels=self.num_classes, kernel_size=3, padding=1, stride=1)
        )

        self.syn_output_conv_x1 = nn.Sequential(
            ResBlock(self.c_enc[1]+self.c_enc[0], self.c_enc[0], stride=1, conv_shortcut=False, groups=self.group, up=False, down=False),
            nn.GroupNorm(num_groups=self.group, num_channels=self.c_enc[0], affine=True),
            nn.SiLU(),
            nn.Conv3d(in_channels=self.c_enc[0], out_channels=self.c_in, kernel_size=3, padding=1, stride=1)
        )
        self.seg_output_conv_x1 = nn.Sequential(
            ResBlock(self.c_enc[1]+self.c_enc[0], self.c_enc[0], stride=1, conv_shortcut=False, groups=self.group, up=False, down=False),
            nn.GroupNorm(num_groups=self.group, num_channels=self.c_enc[0], affine=True),
            nn.SiLU(),
            nn.Conv3d(in_channels=self.c_enc[0], out_channels=self.num_classes, kernel_size=3, padding=1, stride=1)
        )


    def forward(self, x1, x2, s):
        x1, x2, f1, f2, z1, z2, vq_loss = self.encoder(x1, x2, s)

        
        rec2 = self.syn_output_conv_x4(z1)
        seg2 = self.seg_output_conv_x4(z2)

        f_x2_1 = self.up_31(torch.cat([F.interpolate(z1, scale_factor=2, mode='nearest'), f1], dim=1))
        f_x2_2 = self.up_32(torch.cat([F.interpolate(z2, scale_factor=2, mode='nearest'), f2], dim=1))
        rec1 = self.syn_output_conv_x2(f_x2_1)
        seg1 = self.seg_output_conv_x2(f_x2_2)

        rec0 = self.syn_output_conv_x1(torch.cat([F.interpolate(f_x2_1, scale_factor=2, mode='nearest'), x1], dim=1))
        seg0 = self.seg_output_conv_x1(torch.cat([F.interpolate(f_x2_2, scale_factor=2, mode='nearest'), x2], dim=1))
        
        output_rec = [rec0, rec1, rec2]
        output_seg = [seg0, seg1, seg2]

        if not self.deep_supervision:
            return output_rec[0], output_seg[0], vq_loss
        else:
            return output_rec, output_seg, vq_loss


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        stride: int = 1,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        groups: int = 32,
        groups_out: Optional[int] = None,
        eps: float = 1e-5,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_3d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.stride = stride

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        conv_3d_out_channels = conv_3d_out_channels or out_channels
        self.conv2 = nn.Conv3d(out_channels, conv_3d_out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = nn.Upsample(scale_factor=self.stride, mode='nearest')
        elif self.down:
            self.downsample = nn.AvgPool3d(kernel_size=self.stride, stride=self.stride)

        self.use_in_shortcut = self.in_channels != conv_3d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv3d(
                in_channels,
                conv_3d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor.contiguous())

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class AttnBlock(nn.Module):
    def __init__(self, in_channels, groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=in_channels, affine=True)
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w, d = q.shape
        q = rearrange(q, 'b c h w d -> b (h w d) c')
        k = rearrange(k, 'b c h w d -> b (h w d) c')
        v = rearrange(v, 'b c h w d -> b (h w d) c')
        
        # q = q.reshape(b,c,h*w)
        # q = q.permute(0,2,1)   # b,hw,c
        # k = k.reshape(b,c,h*w) # b,c,hw
        
        # w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # w_ = w_ * (int(c)**(-0.5))
        # w_ = F.softmax(w_, dim=2)

        # # attend to values
        # v = v.reshape(b,c,h*w)
        # w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        # h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)
        h_ = rearrange(h_, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
        # h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels, c_enc, group, style_dim, hyper_dim
    ):
        super().__init__()
        self.c_in = in_channels
        self.c_enc = c_enc
        self.group = group
        self.style_dim = style_dim
        self.hyper_dim = hyper_dim
        self.latent_dim = 3
        self.n_emb_syn = 256
        self.n_emb_seg = 16

        self.conv_in1 = hyperConv(self.style_dim, self.c_in, self.c_enc[0], weight_dim=self.hyper_dim, ksize=3, padding=1, stride=1, ndims=3)
        self.conv_in2 = hyperConv(self.style_dim, self.c_in, self.c_enc[0], weight_dim=self.hyper_dim, ksize=3, padding=1, stride=1, ndims=3)

        self.fusion = nn.Sequential(
            ResBlock(self.c_enc[0]*2, self.c_enc[1]*2, stride=2, conv_shortcut=False, groups=self.group, up=False, down=True),
            ResBlock(self.c_enc[1]*2, self.c_enc[1]*2, stride=1, conv_shortcut=False, groups=self.group, up=False, down=False),
            ResBlock(self.c_enc[1]*2, self.c_enc[2]*2, stride=2, conv_shortcut=False, groups=self.group, up=False, down=True),
            AttnBlock(self.c_enc[2]*2, groups=self.group),
            ResBlock(self.c_enc[2]*2, self.c_enc[2]*2, stride=1, conv_shortcut=False, groups=self.group, up=False, down=False),
            nn.GroupNorm(num_groups=self.group, num_channels=self.c_enc[2]*2, affine=True),
            nn.SiLU(),
        )
        self.fusion_conv_syn = nn.Conv3d(self.c_enc[2]*2, self.latent_dim, kernel_size=3, stride=1, padding=1)
        self.fusion_conv_seg = nn.Conv3d(self.c_enc[2]*2, self.latent_dim, kernel_size=3, stride=1, padding=1)
        self.vq_syn = SoftVectorQuantizer(self.n_emb_syn, self.latent_dim)
        self.vq_seg = SoftVectorQuantizer(self.n_emb_seg, self.latent_dim)
        self.fusion_deconv_syn = nn.Conv3d(self.latent_dim, self.c_enc[2], kernel_size=3, stride=1, padding=1)
        self.fusion_deconv_seg = nn.Conv3d(self.latent_dim, self.c_enc[2], kernel_size=3, stride=1, padding=1)

        self.down_1 = nn.Sequential(
            ResBlock(self.c_enc[0], self.c_enc[1], stride=2, conv_shortcut=False, groups=self.group, up=False, down=True),
            ResBlock(self.c_enc[1], self.c_enc[1], stride=1, conv_shortcut=False, groups=self.group, up=False, down=False),
        )
        self.down_2 = nn.Sequential(
            ResBlock(self.c_enc[0], self.c_enc[1], stride=2, conv_shortcut=False, groups=self.group, up=False, down=True),
            ResBlock(self.c_enc[1], self.c_enc[1], stride=1, conv_shortcut=False, groups=self.group, up=False, down=False),
        )

        
    def forward(self, x1, x2, s):
        x1 = self.conv_in1(x1, s)
        x2 = self.conv_in2(x2, s)
        f1 = self.down_1(x1)
        f2 = self.down_2(x2)

        x = torch.cat([x1, x2], dim=1)
        f = self.fusion(x)
        z_syn = self.fusion_conv_syn(f)
        z_seg = self.fusion_conv_seg(f)
        z_q_soft_syn, vq_loss_syn, _ = self.vq_syn(z_syn)
        z_q_soft_seg, vq_loss_seg, _ = self.vq_seg(z_seg)
        z1 = self.fusion_deconv_syn(z_q_soft_syn)
        z2 = self.fusion_deconv_seg(z_q_soft_seg)

        return x1, x2, f1, f2, z1, z2, vq_loss_syn+vq_loss_seg