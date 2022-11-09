import torch
from torch import nn
import torch.nn.functional as F



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

