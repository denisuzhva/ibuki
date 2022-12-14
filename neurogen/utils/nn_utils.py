import torch as t
from torch import nn
from torch import Tensor
import torch.nn.functional as F



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
    def get_norm(pred_data, target_data):
        l = pred_data.shape[-1]
        pad_size = fft_size - l
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left
        pred_data_padded = F.pad(pred_data, (pad_left, pad_right))
        pred_data_fft = t.abs(t.fft.fft(pred_data_padded))
        norm_value = F.smooth_l1_loss(pred_data_fft, 
                                      t.zeros_like(pred_data_fft).to(pred_data_fft.get_device()))
        return norm_value
    return get_norm


def l1_norm_reg():
    def get_norm(pred_data, target_data):
        device = target_data.get_device()
        targ_zeros = t.zeros_like(target_data).to(device)
        norm_value = F.smooth_l1_loss(pred_data, targ_zeros)
        return norm_value
    return get_norm


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


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return t.triu(t.ones(sz, sz) * float('-inf'), diagonal=1)
    
    
def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)