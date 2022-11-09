import torch
from nets.wavenet_unet import UWUNet_v1



wav_channels = 1
sample_len = 256
down_channels = [2, 4, 8]
up_channels = [8, 4, 2]
time_emb_dim = 32
dilation_depth = 2
repeats = 1
with torch.no_grad():
    model = UWUNet_v1(wav_channels, down_channels, up_channels, 
                    time_emb_dim, dilation_depth, repeats)
    model.eval()

    sample = torch.ones((1, wav_channels, sample_len))
    t = torch.tensor([2])

