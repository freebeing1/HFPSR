import torch
import torch.nn as nn
import math


def high_frequency_mask(gt, gt_hf):
    mask = gt_hf.clone()
    mu_mask = mask.mean(dim=(0,2,3)).view(1,3,1,1)
    sigma_mask = mask.std(dim=(0,2,3)).view(1,3,1,1)
    z_mask = (mask-mu_mask.expand(mask.size()))/sigma_mask.expand(mask.size())
    mask[torch.abs(z_mask)>0.5] = 1
    mask[torch.abs(z_mask)<=0.5] = 0.1
    return gt * mask

def get_high_pass_filter(opt):
    opt_filter = opt['netG']['high_pass_filter']
    opt_filter['n_channels'] = opt['n_channels']
    filter_type = opt_filter['filter_type']

    if filter_type == 'laplacian':
        return get_laplacian_hpf(opt_filter)
    elif filter_type == 'gaussian':
        return get_gaussian_hpf(opt_filter)
    else:
        raise ValueError(f"filter_type '{filter_type}' not defined.")


def get_low_pass_filter(opt):
    opt_filter = opt['netG']['high_pass_filter']
    opt_filter['n_channels'] = opt['n_channels']
    filter_type = opt_filter['filter_type']

    if filter_type == 'gaussian':
        return get_gaussian_lpf(opt_filter)
    else:
        raise ValueError(f"filter_type '{filter_type}' not defined.")


def gaussian_kernel_1d(kernel_size, sigma):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
        torch.exp(
        -torch.sum((xy_grid - mean)**2., dim=-1) /
        (2*variance)
    )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    return gaussian_kernel


def identity_kernel_1d(kernel_size):
    identity_kernel = torch.zeros(kernel_size, kernel_size)
    identity_kernel[kernel_size//2, kernel_size//2] = 1.0
    return identity_kernel


def get_gaussian_hpf(opt_filter):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)

    kernel_size = opt_filter['kernel_size']
    channels = opt_filter['n_channels']
    sigma = opt_filter['sigma']

    hpf_kernel = identity_kernel_1d(
        kernel_size) - gaussian_kernel_1d(kernel_size, sigma)

    # Reshape to 2d depthwise convolutional weight
    hpf_kernel = hpf_kernel.view(1, 1, kernel_size, kernel_size)
    hpf_kernel = hpf_kernel.repeat(channels, 1, 1, 1)

    gaussian_hpf = nn.Conv2d(in_channels=channels, out_channels=channels,
                             kernel_size=kernel_size, groups=channels, bias=False, stride=1, padding='same')

    gaussian_hpf.weight.data = hpf_kernel
    gaussian_hpf.weight.requires_grad = False

    return gaussian_hpf


def get_gaussian_lpf(opt_filter):
    kernel_size = opt_filter['kernel_size']
    channels = opt_filter['n_channels']
    sigma = opt_filter['sigma']

    gaussian_kernel = gaussian_kernel_1d(kernel_size, sigma)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, stride=1, padding='same')

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter


def get_laplacian_hpf(opt_filter):
    kernel_size = opt_filter['kernel_size']
    channels = opt_filter['n_channels']
    laplacian_kernel = torch.FloatTensor([[0, -1, 0],
                                          [-1, 4, -1],
                                          [0, -1, 0]])
    laplacian_kernel = laplacian_kernel/4

    laplacian_kernel = laplacian_kernel.view(
        1, 1, kernel_size, kernel_size)
    laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)

    laplacian_filter = nn.Conv2d(
        in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels, bias=False, padding='same')
    laplacian_filter.weight.data = laplacian_kernel
    laplacian_filter.weight.requires_grad = False
    return laplacian_filter
