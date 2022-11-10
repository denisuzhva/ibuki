# wavenet_unet.py

import torch
from torch import nn
from nets.custommodules import (activation_func, 
                                DiffWaveNet, 
                                DiffUNetBlock1d, 
                                DiffPosEncoding)



class UWUNet_v1(nn.Module):
    """
    A UNet with WaveNet wrapping
    """
    def __init__(self,
                 wav_channels,
                 down_channels,
                 up_channels,
                 time_emb_dim,
                 wn_dilation_depth,
                 wn_repeats):
        super().__init__()

        # Time embedding
        self._time_emb = nn.Sequential(
                DiffPosEncoding(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU(),
            )
        
        ###########
        # Encoder #
        ###########
        
        # Initial WaveNet block
        self._init_wavenet = DiffWaveNet(wav_channels, 
                                         down_channels[0], 
                                         down_channels[1], 
                                         wn_dilation_depth, 
                                         wn_repeats,
                                         time_emb_dim,)

        # Downsample
        self._downs = nn.ModuleList([
            DiffUNetBlock1d(down_channels[i], 
                            down_channels[i+1],
                            time_emb_dim)
            for i in range(1, len(down_channels)-1)
        ])
        
        ###########
        # Decoder #
        ###########
        
        # Upsample
        self._ups = nn.ModuleList([
            DiffUNetBlock1d(up_channels[i], 
                            up_channels[i+1], 
                            time_emb_dim, 
                            up=True)
            for i in range(len(up_channels)-2)
        ])
        
        # Final WaveNetBlock
        self._final_wavenet = DiffWaveNet(up_channels[-2], 
                                          up_channels[-1], 
                                          wav_channels,
                                          wn_dilation_depth, wn_repeats,
                                          time_emb_dim,)

    def forward(self, x, timestep):
        # Embedd time
        t = self._time_emb(timestep)
        # UNet
        residual_inputs = []
        init_wn_out = self._init_wavenet(x, t)
        down_out = init_wn_out
        for ddx, down in enumerate(self._downs):
            down_out = down(down_out, t)
            residual_inputs.append(down_out)
        up_out = down_out
        for udx, up in enumerate(self._ups):
            residual_x = residual_inputs.pop()
            up_out = torch.cat((up_out, residual_x), dim=1)           
            up_out = up(up_out, t)
        final_wn_out = self._final_wavenet(up_out, t)
        return final_wn_out




        
