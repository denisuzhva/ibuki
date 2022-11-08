# utils_dcnn.py

import torchvision



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