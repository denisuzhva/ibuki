import numpy as np
import torch
from torch import nn



def init_weights_xavier(m):
    """Xavier weight initializer."""
    if type(m) == (nn.Conv2d or nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def init_weights_kaiming(m):
    """Kaiming weight initializer."""
    if type(m) == (nn.Conv2d or nn.Conv1d):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


def fft_l1_norm(fft_size=4096):
    def get_norm(data, rec_data):
        l = data.shape[-1]
        pad_size = fft_size - l
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left
        data_padded = nn.functional.pad(data, (pad_left, pad_right))
        data_fft = torch.abs(torch.fft.fft(data_padded))
        norm_value = nn.functional.smooth_l1_loss(data_fft, torch.zeros_like(data_fft).to(data_fft.device))
        return norm_value
    return get_norm


def simple_quantizer(x, quantization_channels=256):
    mav = quantization_channels - 1
    if isinstance(x, np.ndarray):
        x_q = x * mav
        x_q = np.around(x_q) / mav
    elif isinstance(x, (torch.Tensor, torch.LongTensor)):

        if isinstance(x, torch.LongTensor):
            x = x.float()
        mav = torch.FloatTensor([mav])
        x_q = x * mav
        x_q = np.around(x_q) / mav
    return x_q

    
def mulaw(x, quantization_channels=256):
    mu = quantization_channels - 1
    if isinstance(x, np.ndarray):
        y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    elif isinstance(x, (torch.Tensor, torch.LongTensor)):
        if isinstance(x, torch.LongTensor):
            x = x.float()
        mu = torch.FloatTensor([mu])
        y = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
    return y


def mulraw_inverse(y, quantization_channels=256):
    mu = quantization_channels - 1
    if isinstance(y, np.ndarray):
        x = np.sign(y) * (np.power(1 + mu, np.abs(y)) - 1) / mu
    elif isinstance(y, (torch.Tensor, torch.LongTensor)):
        if isinstance(y, torch.LongTensor):
            x = x.float()
        mu = torch.FloatTensor([mu])
        x = torch.sign(y) * (torch.pow(1 + mu, torch.abs(y)) - 1) / mu
    return x 
