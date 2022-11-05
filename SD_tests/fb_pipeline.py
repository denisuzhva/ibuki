import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from IPython.display import Audio

from datasets import SimpleWavHandler
from torch.utils.data import DataLoader

from noise_sched import linear_beta_schedule, get_index_from_list, forward_diffusion_sample



if __name__ == '__main__':

    device = torch.device('cpu')
    sr = 16000
    sample_size = 2**12
    batch_size = 1
    wav_path = './datasets/ww_the_deep.wav'
    wav_dataset = SimpleWavHandler(wav_path, sr, mono=True,
                                   sample_size=sample_size, 
                                   unfolding_step=sample_size//2)
    wav_dataloader = DataLoader(wav_dataset, batch_size=batch_size, shuffle=False)

    t_max = 200
    betas = linear_beta_schedule(timesteps=t_max)

    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    
    wav_sample = next(iter(wav_dataloader))[0]
    print(wav_sample.shape)

    plt.figure(figsize=(12, 6))
    plt.axis('off')
    num_images = 5
    stepsize = int(t_max/num_images)

    for idx in range(0, t_max, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
        wav_sample_n, noise = forward_diffusion_sample(wav_sample, t, alphas_cumprod)
        plt.plot(wav_sample_n)
    plt.show()
        