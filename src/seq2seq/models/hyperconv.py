import torch
import torch.nn as nn
import torch.nn.functional as F


class hyperConv(nn.Module):
    def __init__(
        self,
        style_dim,
        dim_in,
        dim_out,
        ksize,
        stride=1,
        padding=None,
        bias=True,
        weight_dim=8,
        ndims=2,
    ):
        super().__init__()
        assert ndims in [2, 3]
        self.ndims = ndims
        self.dim_out = dim_out
        self.stride = stride
        self.bias = bias
        self.weight_dim = weight_dim
        self.fc = nn.Linear(style_dim, weight_dim)
        self.kshape = [dim_out, dim_in, ksize, ksize] if self.ndims==2 else [dim_out, dim_in, ksize, ksize, ksize]
        self.padding = (ksize-1)//2 if padding is None else padding

        self.param = nn.Parameter(torch.randn(*self.kshape, weight_dim).type(torch.float32))
        nn.init.kaiming_normal_(self.param, a=0, mode='fan_in')
        
        if self.bias is True:
            self.fc_bias = nn.Linear(style_dim, weight_dim)
            self.b = nn.Parameter(torch.randn(self.dim_out, weight_dim).type(torch.float32))
            nn.init.constant_(self.b, 0.0)

        self.conv = getattr(F, 'conv%dd' % self.ndims)
            
    def forward(self, x, s):
        kernel = torch.matmul(self.param, self.fc(s).view(self.weight_dim, 1)).view(*self.kshape)
        if self.bias is True:
            bias = torch.matmul(self.b, self.fc_bias(s).view(self.weight_dim,1)).view(self.dim_out)
            return self.conv(x, weight=kernel, bias=bias, stride=self.stride, padding=self.padding)
        else:
            return self.conv(x, weight=kernel, stride=self.stride, padding=self.padding)