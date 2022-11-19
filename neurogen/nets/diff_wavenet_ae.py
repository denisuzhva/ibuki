import numpy as np
import math
import torch
from torch import nn
from utils import activation_func



class CausalConv1d(nn.Conv1d):
    def __init__(self, 
                 in_channels, out_channels, 
                 kernel_size, stride=1, dilation=1, 
                 groups=1, 
                 bias=False):
        self.__padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super().forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result


class DiffWaveNetBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, 
                 kernel_size, dilation,
                 time_emb_dim):
        super().__init__()
        self._dilated = CausalConv1d(res_channels, res_channels, kernel_size, dilation=dilation)
        self._time_mlp =  nn.Linear(time_emb_dim, res_channels)
        self._res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self._skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self._tanh = activation_func('tanh')
        self._sigmoid = activation_func('sigmoid')
        self._relu = activation_func('relu')

    def forward(self, x, t, skip_size):
        # Dilated conv
        x_dil = self._dilated(x)
        # Time embedding
        time_emb = self._relu(self._time_mlp(t))
        time_emb = time_emb[(..., ) + (None, )]
        x_dil_timemb = x_dil + time_emb
        # Gating
        x_tanh = self._tanh(x_dil_timemb)
        x_sigm = self._sigmoid(x_dil_timemb)
        x_gated = x_tanh * x_sigm
        # Residual out
        y_res = self._res_conv(x_gated)
        y_res = y_res + x[..., -y_res.size(2):]
        # Skip out
        y_skip = self._skip_conv(x)
        #y_skip = y_skip[..., -skip_size:]
        return y_res, y_skip


class DiffWaveNetBlockStack(nn.Module):

    def __init__(self,  
                 res_channels, skip_channels, 
                 dilation_depth, repeats,
                 kernel_size,
                 time_emb_dim):
        super().__init__()
        dilations = [2 ** d for d in range(dilation_depth)]
        dilations_repeated = [dilations for _ in range(repeats)]
        self._res_blocks = []
        for rdx, dilations_local in enumerate(dilations_repeated):
            for ddx, dilation in enumerate(dilations_local):
                res_block = DiffWaveNetBlock(res_channels, skip_channels, 
                                             kernel_size, dilation,
                                             time_emb_dim)
                self.add_module(f'WaveNetBlock_{rdx}_{ddx}', res_block) # Add modules manually
                self._res_blocks.append(res_block)

    def forward(self, x, t, skip_size):
        y_res = x
        y_skip_list = []
        for res_block in self._res_blocks:
            y_res, y_skip = res_block(y_res, t, skip_size)
            y_skip_list.append(y_skip)
        return y_res, y_skip_list


class DiffWaveNet(nn.Module):

    def __init__(self, 
                 in_channels, 
                 res_channels, 
                 out_channels, 
                 dilation_depth, 
                 repeats, 
                 time_emb_dim,
                 kernel_size=2,
                 last_act='tanh'):
        super().__init__()
        self._receptive_field_diff = np.sum([(kernel_size - 1) * (2 ** ddx) 
                                             for ddx in range(dilation_depth)] * repeats)
        self._init_conv = CausalConv1d(in_channels, res_channels, kernel_size, dilation=1)
        self._res_stack = DiffWaveNetBlockStack(res_channels, out_channels, 
                                                dilation_depth, repeats,
                                                kernel_size,
                                                time_emb_dim)
        self._out_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self._act = activation_func(last_act)

    def forward(self, x, t):
        x = self._init_conv(x)
        skip_size = int(x.size(2)) - self._receptive_field_diff
        _, y_skip_list = self._res_stack(x, t, skip_size)
        y_skip = torch.sum(torch.stack(y_skip_list), dim=0)
        dense = self._act(self._out_conv(y_skip))
        return dense


class DiffUNetBlock1d(nn.Module):
    
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False):
        super().__init__()
        self._time_mlp =  nn.Linear(time_emb_dim, out_channels)
        if up:
            self._conv1 = nn.Conv1d(2*in_channels, out_channels, 3, padding=1)
            self._transform = nn.ConvTranspose1d(out_channels, out_channels, 4, 2, 1)
        else:
            self._conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
            self._transform = nn.Conv1d(out_channels, out_channels, 4, 2, 1)
        self._conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self._bnorm1 = nn.BatchNorm1d(out_channels)
        self._bnorm2 = nn.BatchNorm1d(out_channels)
        self._act = activation_func('relu')
        
    def forward(self, x, t):
        # First Conv
        h = self._bnorm1(self._act(self._conv1(x)))
        # Time embedding
        time_emb = self._act(self._time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, )]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self._bnorm2(self._act(self._conv2(h)))
        # Down or Upsample
        return self._transform(h)


class DiffPosEncoding(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self._dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
