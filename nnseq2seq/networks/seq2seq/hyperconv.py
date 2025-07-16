import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'hyperConv', 'hyperConvTranspose',
]


def batch_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1):
    batch_size, c1, width = input.shape
    weight = weight.permute(-1, 0, 1, 2)
    _, c2, _, kernel_width = weight.shape

    input = input.view(1, batch_size * c1, width)
    weight = weight.reshape(batch_size * c2, c1, kernel_width)
    output = F.conv1d(input, weight, bias=None, stride=stride, padding=padding, dilation=dilation, groups=batch_size)
    output = output.view(batch_size, c2, output.shape[2])
    if bias is not None:
        bias = bias.permute(-1, 0).unsqueeze(-1)
        output = output + bias

    return output


def batch_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1):
    batch_size, c1, height, width = input.shape
    weight = weight.permute(-1, 0, 1, 2, 3)
    _, c2, _, kernel_height, kernel_width = weight.shape

    input = input.reshape((1, batch_size * c1, height, width))
    weight = weight.reshape(batch_size * c2, c1, kernel_height, kernel_width)
    output = F.conv2d(input, weight, bias=None, stride=stride, padding=padding, dilation=dilation, groups=batch_size)
    output = output.reshape((batch_size, c2, output.shape[2], output.shape[3]))
    if bias is not None:
        bias = bias.permute(-1, 0).unsqueeze(-1).unsqueeze(-1)
        output = output + bias

    return output


def batch_conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1):
    batch_size, c1, depth, height, width = input.shape
    weight = weight.permute(-1, 0, 1, 2, 3, 4)
    _, c2, _, kernel_depth, kernel_height, kernel_width = weight.shape

    input = input.view(1, batch_size * c1, depth, height, width)
    weight = weight.reshape(batch_size * c2, c1, kernel_depth, kernel_height, kernel_width)
    output = F.conv3d(input, weight, bias=None, stride=stride, padding=padding, dilation=dilation, groups=batch_size)
    output = output.view(batch_size, c2, output.shape[2], output.shape[3], output.shape[4])
    if bias is not None:
        bias = bias.permute(-1, 0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        output = output + bias

    return output


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
        dilation=1,
        groups=1,
        weight_dim=8,
        ndims=2,
    ):
        super().__init__()
        assert ndims in [1, 2, 3]
        groups = 1
        self.ndims = ndims
        self.dim_out = dim_out
        self.stride = stride
        self.bias = bias
        self.weight_dim = weight_dim
        self.fc = nn.Sequential(
            nn.Linear(style_dim, weight_dim*4),
            nn.Linear(weight_dim*4, weight_dim)
        )
        self.kshape = [dim_out, dim_in//groups, ksize, ksize] if self.ndims==2 else [dim_out, dim_in//groups, ksize, ksize, ksize]
        self.padding = (ksize-1)//2 if padding is None else padding
        self.groups = groups
        self.dilation = dilation


        self.param = nn.Parameter(torch.randn(*self.kshape, weight_dim).type(torch.float32))
        nn.init.kaiming_normal_(self.param, a=0, mode='fan_in')
        
        if self.bias is True:
            self.fc_bias = nn.Sequential(
                nn.Linear(style_dim, weight_dim*4),
                nn.Linear(weight_dim*4, weight_dim)
            )
            self.b = nn.Parameter(torch.randn(self.dim_out, weight_dim).type(torch.float32))
            nn.init.constant_(self.b, 0.0)

        self.conv = getattr(F, 'conv%dd' % self.ndims)
            
    def forward(self, x, s):
        if s.shape[0]==1:
            return self.forward_bs1(x, s)
        elif s.shape[0]==x.shape[0]:
            return self.forward_bsn(x, s)
    
    def forward_bs1(self, x, s):
        kernel = torch.matmul(self.param, self.fc(s).view(self.weight_dim, 1)).view(*self.kshape)
        if self.bias is True:
            bias = torch.matmul(self.b, self.fc_bias(s).view(self.weight_dim,1)).view(self.dim_out)
            return self.conv(x, weight=kernel, bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        else:
            return self.conv(x, weight=kernel, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
    
    def forward_bsn(self, x, s):
        bs, n = s.shape
        kernel = torch.matmul(self.param, self.fc(s).permute(1,0))
        
        if self.bias is True:
            bias = torch.matmul(self.b, self.fc_bias(s).permute(1,0))
        else:
            bias = None
        if self.ndims==1:
            out = batch_conv1d(x, kernel, bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
        elif self.ndims==2:
            out = batch_conv2d(x, kernel, bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
        elif self.ndims==3:
            out = batch_conv3d(x, kernel, bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation)

        return out
    

class hyperConvTranspose(nn.Module):
    def __init__(
        self,
        style_dim,
        dim_in,
        dim_out,
        ksize,
        stride=1,
        padding=0,
        output_padding=0,
        bias=True,
        dilation=1,
        groups=1,
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
        self.kshape = [dim_in, dim_out//groups, ksize, ksize] if self.ndims==2 else [dim_in, dim_out//groups, ksize, ksize, ksize]
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation


        self.param = nn.Parameter(torch.randn(*self.kshape, weight_dim).type(torch.float32))
        nn.init.kaiming_normal_(self.param, a=0, mode='fan_in')
        
        if self.bias is True:
            self.fc_bias = nn.Linear(style_dim, weight_dim)
            self.b = nn.Parameter(torch.randn(self.dim_out, weight_dim).type(torch.float32))
            nn.init.constant_(self.b, 0.0)

        self.conv = getattr(F, 'conv_transpose%dd' % self.ndims)
            
    def forward(self, x, s):
        if s.shape[0]==1:
            return self.forwart_bs1(x, s)
        elif s.shape[0]==x.shape[0]:
            out = []
            for i in range(s.shape[0]):
                out.append(self.forwart_bs1(x[i:i+1], s[i:i+1]))
            out = torch.cat(out, dim=0)
            return out
    
    def forwart_bs1(self, x, s):
        kernel = torch.matmul(self.param, self.fc(s).view(self.weight_dim, 1)).view(*self.kshape)
        if self.bias is True:
            bias = torch.matmul(self.b, self.fc_bias(s).view(self.weight_dim,1)).view(self.dim_out)
            return self.conv(x, weight=kernel, bias=bias, stride=self.stride, padding=self.padding, output_padding=self.output_padding, dilation=self.dilation, groups=self.groups)
        else:
            return self.conv(x, weight=kernel, stride=self.stride, padding=self.padding, output_padding=self.output_padding, dilation=self.dilation, groups=self.groups)