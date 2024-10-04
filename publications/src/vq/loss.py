import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    https://github.com/HobbitLong/SupContrast
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (features.device
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    

class SSIMLoss(nn.Module):
    """
    Build a Pytorch version of the SSIM loss function based on the original formula of SSIM

    Modified and adopted from:
        https://github.com/facebookresearch/fastMRI/blob/main/banding_removal/fastmri/ssim_loss_mixin.py

    For more info, visit
        https://vicuesoft.com/glossary/term/ssim-ms-ssim/

    SSIM reference paper:
        Wang, Zhou, et al. "Image quality assessment: from error visibility to structural
        similarity." IEEE transactions on image processing 13.4 (2004): 600-612.
    """
    
    def __init__(self, win_size: int=7, k1: float=0.01, k2: float=0.03, spatial_dims: int=2):
        """
        Args:
            win_size: gaussian weighting window size
            k1: stability constant used in the luminance denominator
            k2: stability constant used in the contrast denominator
            spatial_dims: if 2, input shape is expected to be (B,C,H,W). if 3, it is expected to be (B,C,H,W,D)
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.spatial_dims = spatial_dims
        self.register_buffer(
            "w", torch.ones([1, 1] + [win_size for _ in range(spatial_dims)]) / win_size**spatial_dims
        )
        self.cov_norm = (win_size**2) / (win_size**2 - 1)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, data_range: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: first sample (e.g., the reference image). Its shape is (B,C,W,H) for 2D and pseudo-3D data,
                and (B,C,W,H,D) for 3D data,
            y: second sample (e.g., the reconstructed image). It has similar shape as x.
            data_range: dynamic range of the data

        Returns:
            1-ssim_value (recall this is meant to be a loss function)

        Example:
            .. code-block:: python

                import torch

                # 2D data
                x = torch.ones([1,1,10,10])/2
                y = torch.ones([1,1,10,10])/2
                data_range = x.max().unsqueeze(0)
                # the following line should print 1.0 (or 0.9999)
                print(1-SSIMLoss(spatial_dims=2)(x,y,data_range))

                # pseudo-3D data
                x = torch.ones([1,5,10,10])/2  # 5 could represent number of slices
                y = torch.ones([1,5,10,10])/2
                data_range = x.max().unsqueeze(0)
                # the following line should print 1.0 (or 0.9999)
                print(1-SSIMLoss(spatial_dims=2)(x,y,data_range))

                # 3D data
                x = torch.ones([1,1,10,10,10])/2
                y = torch.ones([1,1,10,10,10])/2
                data_range = x.max().unsqueeze(0)
                # the following line should print 1.0 (or 0.9999)
                print(1-SSIMLoss(spatial_dims=3)(x,y,data_range))
        """
        if x.shape[1] > 1:  # handling multiple channels (C>1)
            if x.shape[1] != y.shape[1]:
                raise ValueError(
                    f"x and y should have the same number of channels, "
                    f"but x has {x.shape[1]} channels and y has {y.shape[1]} channels."
                )
            losses = torch.stack(
                [
                    SSIMLoss(self.win_size, self.k1, self.k2, self.spatial_dims)(
                        x[:, i, ...].unsqueeze(1), y[:, i, ...].unsqueeze(1), data_range
                    )
                    for i in range(x.shape[1])
                ]
            )
            channel_wise_loss: torch.Tensor = losses.mean()
            return channel_wise_loss

        data_range = data_range[(None,) * (self.spatial_dims + 2)]
        # determine whether to work with 2D convolution or 3D
        conv = getattr(F, f"conv{self.spatial_dims}d")
        w = self.w.to(device=x.device, dtype=x.dtype)

        c1 = (self.k1 * data_range) ** 2  # stability constant for luminance
        c2 = (self.k2 * data_range) ** 2  # stability constant for contrast
        ux = conv(x, w)  # mu_x
        uy = conv(y, w)  # mu_y
        uxx = conv(x * x, w)  # mu_x^2
        uyy = conv(y * y, w)  # mu_y^2
        uxy = conv(x * y, w)  # mu_xy
        vx = self.cov_norm * (uxx - ux * ux)  # sigma_x
        vy = self.cov_norm * (uyy - uy * uy)  # sigma_y
        vxy = self.cov_norm * (uxy - ux * uy)  # sigma_xy

        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denom = (ux**2 + uy**2 + c1) * (vx + vy + c2)
        ssim_value = numerator / denom
        loss: torch.Tensor = 1 - ssim_value.mean()
        return loss