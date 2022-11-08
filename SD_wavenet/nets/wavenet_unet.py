# wavenet_unet.py

import torch
from torch import nn
from custommodules import (activation_func, 
                           WaveNet, 
                           UNetBlock1d, 
                           DiffusionPosEncoding)




class EncoderNet1d_WNv1(nn.Module):
    """
    EncoderNet1d_WNv1(l_in : int, in_channels : int, params : dict)
        WaveNet encoder.

        Parameters
        ----------
        l_in :                      Length of input data
        in_channels :               Number of input channels
        params :                    Set of parameters
            channels :              List of numbers of kernels for each convolution layer
            dilation_depth :        Depth of dilations
            repeats :               Number of dilation layer repeats
            pools :                 List of pooling kernel size values for each convolution layer
                OR
            downsampling_sizes :    List of downsampling sizes
    """
    def __init__(self, in_channels, params):
        super().__init__()
        channels = params['channels']
        dilation_depth = params['dilation_depth']
        repeats = params['repeats']
        if 'pools' in params.keys():
            pools_over_downsamplings = True
            pools = params['pools']
        elif 'downsampling_sizes' in params.keys():
            pools_over_downsamplings = False
            pools = params['downsampling_sizes']
        else:
            raise NotImplementedError("Unable to find pooling parameters")

        act = activation_func('relu')

        channels = channels[:1] + \
                   [channels[0] * dilation_depth * repeats] + \
                   channels[1:]
        self._wavenet_block = nn.Sequential([
            nn.Conv1d(in_channels, channels[0], kernel_size=1),
            act,
            WaveNet(channels[0], dilation_depth, repeats),
            #act,
        ])

        self._downsample_blocks = nn.ModuleList([
            nn.Sequential([
                nn.Conv1d(channels[idx+1], channels[idx+2], kernel_size=1),
                nn.MaxPool1d(p) if pools_over_downsamplings else nn.AdaptiveMaxPool1d(p),
                act,
            ])
            for idx, p in enumerate(pools)
        ])

    def forward(self, x):
        block_outs = []
        out = self._wavenet_block(x)
        block_outs.append(out)
        
        for module in self._downsample_blocks:
            out = module(out) 
            block_outs.append(out)
        
        return block_outs

    def get_l_out(self):
        return 0

         
class DecoderNet1d_WNv1(nn.Module):
    """
    DecoderNet1d_WNv1(l_in : int, in_channels : int, params : dict)
        WaveNet decoder for compressive sensing.

        Parameters
        ----------
        l_in :                      Length of input data
        in_channels :               Number of input channels
        params :                    Set of parameters
            channels :              List of numbers of kernels for each convolution layer
            dilation_depth :        Depth of dilations
            repeats :               Number of dilation layer repeats
            upsampling_sizes :      List of upsampling sizes
            last_act :              Last activation function
    """
    def __init__(self, in_channels, params):
        super().__init__()
        channels = params['channels']
        dilation_depth = params['dilation_depth']
        repeats = params['repeats']
        upsampling_sizes = params['upsampling_sizes']
        last_act = params['last_act']

        act1 = activation_func('relu')
        act2 = activation_func(last_act)
        layers = []
        channels = [in_channels] + channels

        for idx, _ in enumerate(upsampling_sizes):
            layers.append(nn.Conv1d(channels[idx], channels[idx+1], kernel_size=1))
            layers.append(act1)
            layers.append(nn.Upsample(upsampling_sizes[idx], mode='linear', align_corners=False))
        
        start_idx = len(upsampling_sizes)
        layers.append(WaveNet(channels[start_idx], dilation_depth, repeats))

        layers.append(nn.Conv1d(channels[start_idx] * dilation_depth * repeats, channels[start_idx+1], kernel_size=1))
        layers.append(act2)

        layers.append(nn.Conv1d(channels[start_idx+1], channels[start_idx+2], kernel_size=1))
        layers.append(act2)

        self.__net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.__net(x)
        return out


class UWUNet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self,
                 wav_channels,
                 down_channels,
                 up_channels,
                 wn_dilation_depth,
                 wn_repeats):
        super().__init__()
        
        # Add the number of channels after WaveNet
        down_channels = down_channels[:1] + \
                        [down_channels[0] * wn_dilation_depth * wn_repeats] + \
                        down_channels[1:]
        
        #down_channels = (64, 128, 256, 512, 1024)
        #up_channels = (1024, 512, 256, 128, 64)
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                DiffusionPosEncoding(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Additional functions
        act = activation_func('relu')
        
        ###########
        # Encoder #
        ###########
        
        # Initial WaveNet block
        
        self._wavenet_block = nn.Sequential([
            nn.Conv1d(wav_channels, down_channels[0], kernel_size=1),
            act,
            WaveNet(down_channels[0], wn_dilation_depth, wn_repeats),
            #act,
        ])

        # Downsample
        self._downs = nn.ModuleList([UNetBlock1d(down_channels[i], 
                                                 down_channels[i+1],
                                                 time_emb_dim)
                                     for i in range(1, len(down_channels)-1)])
        
        ###########
        # Decoder #
        ###########
        
        # Upsample
        self.ups = nn.ModuleList([UNetBlock1d(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        self.output = nn.Conv2d(up_channels[-1], 3, out_dim)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)




        
