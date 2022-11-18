import torchvision
import torch
from torch import nn
from torch import Tensor
import math



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


def calc_conv1d_shape(l_in, kernel_size, stride=1, padding=0, dilation=1):
    l_out = (l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    return l_out


def calc_conv2d_shape(hw_in, kernel_size, stride=1, padding=0, dilation=1):
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size) 
    if not isinstance(stride, tuple):
        stride = (stride, stride) 
    if not isinstance(padding, tuple):
        padding = (padding, padding) 
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation) 
    hw_out = []
    hw_out.append(int((hw_in[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1))
    hw_out.append(int((hw_in[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1))
    hw_out = tuple(hw_out) # make tuple consisting of height and width
    return hw_out


def calc_convT2d_shape(hw_in, kernel_size, stride=1, padding=0, output_padding=0, dilation=1):
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size) 
    if not isinstance(stride, tuple):
        stride = (stride, stride) 
    if not isinstance(padding, tuple):
        padding = (padding, padding) 
    if not isinstance(output_padding, tuple):
        output_padding = (output_padding, output_padding)    
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation) 
    hw_out = []
    hw_out.append(int((hw_in[0] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1))
    hw_out.append(int((hw_in[1] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1))
    hw_out = tuple(hw_out) # make tuple consisting of height and width
    return hw_out


def denormalize_makegrid(im, norm_mean, norm_std, max_norm=False):
    """
    Denormalize images from given normalization parameters 
    as in torchvision.transforms.Normalize;
    make a grid of the batch of images.

    Args:
        im:         Image of type torch.FloatTensor or torch.cuda.FloatTensor
        norm_mean:  Mean of image normalization transform
        norm_std:   Standard deviation of image normalization transform
        max_norm:   Normalize by maximum value
    """
    im = im.mul_(norm_std.view(1, -1, 1, 1)).add_(norm_mean.view(1, -1, 1, 1))
    im = torchvision.utils.make_grid(im)
    im = im.cpu().numpy()
    im = im.transpose((1, 2, 0))
    if max_norm:
        im = im / im.max()

    return im

    
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
    
    
class PosEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

        
def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    
    