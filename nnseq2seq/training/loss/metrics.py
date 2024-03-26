from turtle import forward
import numpy as np
import skimage.metrics as metrics

import torch
import torch.nn as nn

import lpips

__all__ = [
    'torch_PSNR', 'torch_LPIPS',
    'np_PSNR', 'np_PSNR_mask', 'np_SSIM',
]


def torch_PSNR(image_true, image_test, data_range=255., eps=1e-9):
    mse = torch.mean((image_true - image_test) ** 2) + eps
    return 10 * torch.log10(data_range**2 / mse)


class torch_LPIPS(nn.Module):
    def __init__(self, net='alex') -> None:
        """
        net : str
            ['alex','vgg','squeeze'] are the base/trunk networks available
        """
        super().__init__()
        self.loss = lpips.LPIPS(net=net)
    
    @torch.no_grad()
    def forward(self, image_true, image_test):
        return self.loss(image_true, image_test)


def np_PSNR(image_true, image_test, data_range=255.):
    return metrics.peak_signal_noise_ratio(image_true, image_test, data_range=data_range)


def np_PSNR_mask(image_true, image_test, mask, data_range=255.):
    image_true = image_true[mask>0.5]
    image_test = image_test[mask>0.5]
    mse = np.mean((image_true - image_test) ** 2)
    return 10 * np.log10(data_range**2 / mse)


def np_SSIM(image_true, image_test, data_range=255.):
    return metrics.structural_similarity(image_true, image_test, data_range=data_range)