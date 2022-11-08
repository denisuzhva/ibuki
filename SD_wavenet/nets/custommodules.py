# customlayers.py

import numpy as np
import torch
from torch import nn
from nets.utils_dcnn import calc_conv1d_shape



def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.1, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['sigmoid', nn.Sigmoid()],
        ['tanh', nn.Tanh()],
        ['tanhshrink', nn.Tanhshrink()],
        ['none', nn.Identity()]
    ])[activation]


class CausalConv1d(nn.Conv1d):
    def __init__(self, 
                 in_channels, out_channels, 
                 kernel_size, stride=1, dilation=1, 
                 groups=1, 
                 bias=True):
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


class WaveNetBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, kernel_size, dilation):
        super().__init__()
        self._dilated = CausalConv1d(res_channels, res_channels, kernel_size, dilation=dilation)
        self._res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self._skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self._tanh = activation_func('tanh')
        self._sigmoid = activation_func('sigmoid')

    def forward(self, x, skip_size):
        x_dil = self._dilated(x)
        x_tanh = self._tanh(x_dil)
        x_sigm = self._sigmoid(x_dil)
        x_gated = x_tanh * x_sigm
        y_res = self._res_conv(x_gated)
        y_res = y_res + x[..., -y_res.size(2):]
        y_skip = self._skip_conv(x)
        y_skip = y_skip[..., -skip_size:]
        return y_res, y_skip


class StackOfResBlocks(nn.Module):

    def __init__(self, dilation_depth, repeats, res_channels, skip_channels, kernel_size):
        super().__init__()
        dilations = [2 ** d for d in range(dilation_depth)]
        dilations_repeated = [dilations for _ in range(repeats)]
        self._res_blocks = []
        for rdx, dilations_local in enumerate(dilations_repeated):
            for ddx, dilation in enumerate(dilations_local):
                res_block = WaveNetBlock(res_channels, skip_channels, kernel_size, dilation)
                self.add_module(f'WaveNetBlock_{rdx}_{ddx}', res_block) # Add modules manually
                self._res_blocks.append(res_block)

    def forward(self, x, skip_size):
        y_res = x
        ys_skip = []
        for res_block in self._res_blocks:
            y_res, y_skip = res_block(y_res, skip_size)
            ys_skip.append(y_skip)
        return y_res, ys_skip


class WaveNet(nn.Module):

    # Stack 
    def _conv_stack(self, dilations, in_channels, out_channels, kernel_size):
        """
        Create stack of dilated convolutional layers, outlined in WaveNet paper:
        https://arxiv.org/pdf/1609.03499.pdf
        """
        return nn.ModuleList(
            [
                CausalConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dilation=d,
                    kernel_size=kernel_size,
                )
                for _, d in enumerate(dilations)
            ]
        )

    # Main init
    def __init__(self, n_channels, dilation_depth, repeats, kernel_size=2):
        super().__init__()
        dilations = [2 ** d for d in range(dilation_depth)] * repeats
        self.__hidden = self._conv_stack(dilations, n_channels, n_channels, kernel_size)
        self.__residuals = self._conv_stack(dilations, n_channels, n_channels, 1)

    def forward(self, x):
        out = x
        skips = []

        for hidden, residual in zip(self.__hidden, self.__residuals):
            x = out
            out_hidden = hidden(x)

            # gated activation
            out = torch.tanh(out_hidden) * torch.sigmoid(out_hidden)

            skips.append(out)

            out = residual(out)
            out = out + x[:, :, -out.size(2) :]

        # modified "postprocess" step:
        out = torch.cat([s[:, :, -out.size(2) :] for s in skips], dim=1)
        return out


class UNetBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv1d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose1d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv1d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm1d(out_ch)
        self.bnorm2 = nn.BatchNorm1d(out_ch)
        self.act = activation_func('relu')
        
    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.act(self.conv1(x)))
        # Time embedding
        time_emb = self.act(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.act(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class DiffusionPosEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
