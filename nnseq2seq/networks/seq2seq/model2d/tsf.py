import torch
import torch.nn as nn
import torch.nn.functional as F

from nnseq2seq.networks.seq2seq.model2d.convnext import Block, LayerNorm, ResBlock, hyperResBlock, hyperAttnResBlock


class TSF_attention(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        
        self.num_channel = args['style_dim']
        self.layer_scale_init_value = args['layer_scale_init_value']
        self.hyper_dim = args['hyper_conv_dim']
        self.latent_space_dim = args['latent_space_dim']
        self.style_dim = args['style_dim'] * 2 + 1

        self.fc_w = nn.Sequential(
            nn.Linear(self.style_dim, self.num_channel),
            nn.Softmax(dim=1)
        )

        self.attn_layer = hyperAttnResBlock(
            self.latent_space_dim*self.num_channel, self.style_dim, 2, self.hyper_dim, 3, 1, layer_scale_init_value=self.layer_scale_init_value, use_attn=True)
        
        self.out_layer = nn.Sequential(
            LayerNorm(self.latent_space_dim*self.num_channel, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(self.latent_space_dim*self.num_channel, out_channels=self.latent_space_dim, kernel_size=3, padding=1, stride=1, padding_mode='zeros'),
        )
            
    def forward(self, zs, s, eps=1e-5):
        res = zs
        seq_in = s[:, :self.num_channel].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        param = self.fc_w(s).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + eps  # + eps to avoid dividing 0
        x = torch.stack(zs.chunk(self.num_channel, dim=1), dim=1)
        x = torch.sum(x * seq_in * param, dim=1) / torch.sum(seq_in * param)
        
        res = self.attn_layer(res, s)
        res = self.out_layer(res)
        finetune = x + res
        return x, finetune
    
    def infer_contribution(self, s, eps=1e-5):
        seq_in = s[:, :self.num_channel].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        param = self.fc_w(s).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + eps  # + eps to avoid dividing 0
        w = (seq_in * param) / torch.sum(seq_in * param)
        return w