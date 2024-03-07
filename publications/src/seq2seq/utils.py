from typing import List, Dict, Tuple

import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
import skimage.metrics as metrics

import lpips

import torch
import torch.nn as nn
import torchvision


class Plotter(object):
    """
    Plot the loss/metric curves
    """
    def __init__(self, send_path: str, keys1: List[str], keys2: List[str]=[]) -> None:
        """
        send_path: path to save the figures
        keys1:     keys shown on the left axis
        keys2:     keys shown on the right axis
        """
        self.send_path = send_path
        self.keys1 = keys1
        self.keys2 = keys2
        self.colormap = self.generate_colormap(keys1+keys2)
        
    def generate_colormap(self, keys: List[str]) -> Dict[str, Tuple[float]]:
        """ Generate colormap based on the number of keys.
        keys: keys for ploting
        """
        ncol = len(keys)
        if ncol <= 10:
            colors = [i for i in get_cmap('tab10').colors]
        elif 10 < ncol <= 20:
            colors = [i for i in get_cmap('tab20').colors]
        elif 25 < ncol <= 256:
            cmap = get_cmap(name='viridis')
            colors = cmap(np.linspace(0, 1, ncol))
        else:
            raise ValueError('Maximum 256 categories.')

        color_map = {}
        for i, key in enumerate(keys):
            color_map[key] = colors[i]
        return color_map

    def send(self, data: Dict[str, List[float]], ylabel1: str='loss', ylabel2: str='Metric') -> None:
        """ function to plot the curve
        """
        font = {'weight': 'normal', 'size': 18}
        matplotlib.rc('font', **font)

        fig = plt.figure(figsize=(30, 24))
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel(ylabel1)

        if len(self.keys2)!=0:
            ax2 = ax1.twinx()
            ax2.set_ylabel(ylabel2)

        for key in self.keys1:
            ax1.plot(np.array([i for i in range(len(data[key]))]), np.array(data[key]), color=self.colormap[key], label=key, ls='-')
        for key in self.keys2:
            ax2.plot(np.array([i for i in range(len(data[key]))]), np.array(data[key]), color=self.colormap[key], label=key, ls='--')
        
        ax1.legend()
        if len(self.keys2)!=0:
            ax2.legend(loc=9)
        fig.savefig(os.path.join(self.send_path, 'progress.png'))
        plt.close()


class Recorder(object):
    """
    record the metric and return the statistic results
    """
    def __init__(self, keys: List[str]) -> None:
        """
        keys: variables' name to be saved
        """
        self.data = dict()
        self.keys = keys
        for key in keys:
            self.data[key] = []

    def update(self, item: Dict[str, float]) -> None:
        """
        item: data dict to update the buffer, the keys should be consistent
        """
        for key in item.keys():
            self.data[key].append(item[key])

    def reset(self, keys: List[str]=[]) -> None:
        """
        keys: variables to be cleaned in the buffer
        """
        keys = self.data.keys() if len(keys)==0 else keys
        for key in keys:
            self.data[key] = []
    
    def call(self) -> Dict[str, List[float]]:
        return self.data

    def callmean(self, key: str, return_std: bool=False) -> float:
        """
        key:        variable to be calculated for the statistical results
        return_std: option to return variance
        """
        arr = np.array(self.data[key])
        if return_std:
            return np.mean(arr), np.std(arr)
        else:
            return np.mean(arr)


def save_grid_images(print_list: List[List[torch.Tensor]], output_path: str, clip_range: List[float]=[0.0, 1.0], **kwargs) -> None:
    """ Save image in MxN
    print_list: each image has shape of (1,c,h,w) and 'c' should be 1 or 3
        [
            [img11, img12, img13],
            [img21, img22, img23],
            [img31, img32, img33],
        ]
    """
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    nrow = len(print_list[0])
    img = torch.cat([torch.cat(col, dim=0) for col in print_list], dim=0)
    img = torch.clamp(img, clip_range[0], clip_range[1])
    grid_img = torchvision.utils.make_grid(img, nrow=nrow, **kwargs)
    torchvision.utils.save_image(grid_img, output_path)


def poly_lr(epoch: int, max_epochs: int, initial_lr: float, min_lr: float=1e-5, exponent: float=0.9) -> float:
    return min_lr + (initial_lr - min_lr) * (1 - epoch / max_epochs)**exponent


def torch_PSNR(image_true, image_test, data_range=255.):
    mse = torch.mean((image_true - image_test) ** 2)
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