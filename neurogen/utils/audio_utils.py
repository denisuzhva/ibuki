import numpy as np
import torch



def simple_quantizer(x, quantization_channels=256):
    mav = quantization_channels - 1
    if isinstance(x, np.ndarray):
        x_q = x * mav
        x_q = np.around(x_q) / mav
    elif isinstance(x, (torch.Tensor, torch.LongTensor)):
        device = x.get_device()
        if isinstance(x, torch.LongTensor):
            x = x.float()
        mav = torch.FloatTensor([mav]).to(device)
        x_q = x * mav
        x_q = np.around(x_q) / mav
    return x_q

    
def mulaw(x, quantization_channels=256):
    mu = quantization_channels - 1
    if isinstance(x, np.ndarray):
        y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    elif isinstance(x, (torch.Tensor, torch.LongTensor)):
        device = x.get_device()
        if isinstance(x, torch.LongTensor):
            x = x.float()
        mu = torch.FloatTensor([mu]).to(device)
        y = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
    return y


def mulraw_inverse(y, quantization_channels=256):
    mu = quantization_channels - 1
    if isinstance(y, np.ndarray):
        x = np.sign(y) * (np.power(1 + mu, np.abs(y)) - 1) / mu
    elif isinstance(y, (torch.Tensor, torch.LongTensor)):
        device = y.get_device()
        if isinstance(y, torch.LongTensor):
            y = y.float()
        mu = torch.FloatTensor([mu]).to(device)
        x = torch.sign(y) * (torch.pow(1 + mu, torch.abs(y)) - 1) / mu
    return x 


def wav16_to_onehot(target_sample_batch, n_classes=256, do_mu=True):
    target_pmone = target_sample_batch.type(torch.float32) / 2**15
    if do_mu:
        target_pmone = mulaw(target_pmone, 
                             quantization_channels=n_classes)
    target_uint = torch.round((target_pmone + 1) * (n_classes-1) / 2).long()
    target_onehot = F.one_hot(target_uint, n_classes)
    return target_onehot
        

    