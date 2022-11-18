import torch
from torch import nn
from neurowhore.nets.diff_wavenet_ae import (
    DiffWaveNet, 
    DiffUNetBlock1d, 
    DiffPosEncoding
)



class NoiseScheduler(nn.Module):
    
    def __init__(self,
                 timesteps,
                 start=0.0001,
                 end=0.02,
                 distrib_type='uniform',):
        super().__init__()
        self._distrib_type = distrib_type
        betas = torch.linspace(start, end, timesteps)
        alphas = 1. - betas
        self._alphas_cumprod = torch.cumprod(alphas, axis=0)

    @staticmethod
    def get_index_from_list(vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward(self, x_0, t):
        """ 
        Takes a sample and a timestep as input and 
        returns the noisy version of it
        """
        if self._distrib_type == 'uniform':
            noise = torch.rand_like(x_0) * 2 - 1
            alphas_cumprod_t = self.get_index_from_list(self._alphas_cumprod, t, x_0.shape)
            one_minus_alphas_cumprod_t = self.get_index_from_list(
                1. - self._alphas_cumprod, t, x_0.shape
            )
            x_noised = alphas_cumprod_t * x_0 + one_minus_alphas_cumprod_t * noise
        elif self._distrib_type == 'gauss':
            noise = torch.randn_like(x_0)
            sqrt_alphas_cumprod = torch.sqrt(self._alphas_cumprod)
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self._alphas_cumprod)
            sqrt_alphas_cumprod_t = self.get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
            sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
                sqrt_one_minus_alphas_cumprod, t, x_0.shape
            )
            x_noised = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        else:
            raise NotImplementedError("Unknown diffusion distribution")
        
        return x_noised, noise


class UWUNet_v1(nn.Module):
    """
    A UNet with WaveNet wrapping
    """
    def __init__(self, params):
        super().__init__()

        # Unpack model parameters
        wav_channels = params['wav_channels']
        down_channels = params['down_channels']
        up_channels = params['up_channels']
        time_emb_dim = params['time_emb_dim']
        wn_dilation_depth = params['wn_dilation_depth']
        wn_repeats = params['wn_repeats']

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




        
