
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

__all__ = [
    'PerceptualLoss',
    'AdaptivePerceptualLoss',
]


IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]


def check_and_warn_input_range(tensor, min_value, max_value, name):
    actual_min = tensor.min()
    actual_max = tensor.max()
    if actual_min < min_value or actual_max > max_value:
        warnings.warn(f"{name} must be in {min_value}..{max_value} range, but it ranges {actual_min}..{actual_max}")


class PerceptualLoss(nn.Module):
    def __init__(self, normalize_inputs: bool=True, model='vgg19') -> None:
        super(PerceptualLoss, self).__init__()

        self.normalize_inputs = normalize_inputs
        self.mean_ = IMAGENET_MEAN
        self.std_ = IMAGENET_STD

        if torchvision.__version__.split('+')[0]<'0.13':
            if model=='vgg19':
                vgg = torchvision.models.vgg19(pretrained=True).features
            elif model=='vgg16':
                vgg = torchvision.models.vgg16(pretrained=True).features
            elif model=='vgg13':
                vgg = torchvision.models.vgg13(pretrained=True).features
            elif model=='vgg11':
                vgg = torchvision.models.vgg11(pretrained=True).features
            else:
                raise ValueError('Unknown vgg model: {}'.format(model))
        else:
            if model=='vgg19':
                vgg = torchvision.models.vgg19(weights='VGG19_Weights.DEFAULT').features
            elif model=='vgg16':
                vgg = torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT').features
            elif model=='vgg13':
                vgg = torchvision.models.vgg13(weights='VGG13_Weights.DEFAULT').features
            elif model=='vgg11':
                vgg = torchvision.models.vgg11(weights='VGG11_Weights.DEFAULT').features
            else:
                raise ValueError('Unknown vgg model: {}'.format(model))
        vgg_avg_pooling = []

        #for weights in vgg.parameters():
        #    weights.requires_grad = False

        for module in vgg.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                vgg_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                vgg_avg_pooling.append(module)

        self.vgg = nn.Sequential(*vgg_avg_pooling)
        self.vgg.train()

    def do_normalize_inputs(self, x):
        return (x - self.mean_.to(x.device)) / self.std_.to(x.device)

    def partial_losses(self, input, target, mask=None):
        #check_and_warn_input_range(target, 0, 1, 'PerceptualLoss target in partial_losses')

        # we expect input and target to be in [0, 1] range
        losses = []

        if self.normalize_inputs:
            features_input = self.do_normalize_inputs(input)
            features_target = self.do_normalize_inputs(target)
        else:
            features_input = input
            features_target = target

        for layer in self.vgg:

            features_input = layer(features_input)
            features_target = layer(features_target)

            if layer.__class__.__name__ == 'ReLU':
                loss = F.mse_loss(features_input, features_target, reduction='none')

                if mask is not None:
                    cur_mask = F.interpolate(mask, size=features_input.shape[-2:],
                                             mode='bilinear', align_corners=False)
                    loss = loss * cur_mask

                loss = loss.mean(dim=tuple(range(1, len(loss.shape))))
                losses.append(loss)

        return losses

    def forward(self, input, target, mask=None):
        losses = self.partial_losses(input, target, mask=mask)
        return torch.stack(losses).sum(dim=0).mean(dim=0)

    def get_global_features(self, input):
        #check_and_warn_input_range(input, 0, 1, 'PerceptualLoss input in get_global_features')

        if self.normalize_inputs:
            features_input = self.do_normalize_inputs(input)
        else:
            features_input = input

        features_input = self.vgg(features_input)
        return features_input


class AdaptivePerceptualLoss(nn.Module):
    def __init__(self, normalize_inputs: bool=True, model='vgg19', alpha=0.01, scale=100) -> None:
        super(AdaptivePerceptualLoss, self).__init__()

        self.normalize_inputs = normalize_inputs
        self.mean_ = IMAGENET_MEAN
        self.std_ = IMAGENET_STD
        self.alpha = alpha
        self.scale = scale

        if torchvision.__version__.split('+')[0]<'0.13':
            if model=='vgg19':
                vgg = torchvision.models.vgg19(pretrained=True).features
            elif model=='vgg16':
                vgg = torchvision.models.vgg16(pretrained=True).features
            elif model=='vgg13':
                vgg = torchvision.models.vgg13(pretrained=True).features
            elif model=='vgg11':
                vgg = torchvision.models.vgg11(pretrained=True).features
            else:
                raise ValueError('Unknown vgg model: {}'.format(model))
        else:
            if model=='vgg19':
                vgg = torchvision.models.vgg19(weights='VGG19_Weights.DEFAULT').features
            elif model=='vgg16':
                vgg = torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT').features
            elif model=='vgg13':
                vgg = torchvision.models.vgg13(weights='VGG13_Weights.DEFAULT').features
            elif model=='vgg11':
                vgg = torchvision.models.vgg11(weights='VGG11_Weights.DEFAULT').features
            else:
                raise ValueError('Unknown vgg model: {}'.format(model))
        vgg_avg_pooling = []

        #for weights in vgg.parameters():
        #    weights.requires_grad = False
        self.select_layers = []
        layer_id = 0

        for module in vgg.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                vgg_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
                self.select_layers.append(layer_id-1)
                layer_id += 1
            else:
                vgg_avg_pooling.append(module)
                layer_id += 1

        self.vgg = nn.Sequential(*vgg_avg_pooling)
        self.vgg.train()

        if len(self.select_layers)==5:
            self.W_init = [100., 1.6, 2.3, 1.8, 2.8, 100.]
        else:
            self.W_init = [1 for _ in range(len(self.select_layers)+1)]

    def do_normalize_inputs(self, x):
        return (x - self.mean_.to(x.device)) / self.std_.to(x.device)

    def partial_losses(self, input, target, mask=None):
        #check_and_warn_input_range(target, 0, 1, 'PerceptualLoss target in partial_losses')

        # we expect input and target to be in [0, 1] range
        losses = []

        if self.normalize_inputs:
            features_input = self.do_normalize_inputs(input)
            features_target = self.do_normalize_inputs(target)
        else:
            features_input = input
            features_target = target
        
        loss = F.mse_loss(features_input, features_target, reduction='none')
        if mask is not None:
            loss = loss * mask
        loss_id = 0
        loss = loss.mean()
        if not torch.isnan(loss):
            self.W_init[loss_id] = self.W_init[loss_id] + self.alpha * (loss.item() - self.W_init[loss_id])
        losses.append(loss/(abs(self.W_init[loss_id])+1e-5)*self.scale)
        for id_layer, layer in enumerate(self.vgg):

            features_input = layer(features_input)
            features_target = layer(features_target)

            if id_layer in self.select_layers:
                loss_id += 1
                loss = F.mse_loss(features_input, features_target, reduction='none')

                if mask is not None:
                    cur_mask = F.interpolate(mask, size=features_input.shape[-2:],
                                             mode='bilinear', align_corners=False)
                    loss = loss * cur_mask

                loss = loss.mean()
                if not torch.isnan(loss):
                    self.W_init[loss_id] = self.W_init[loss_id] + self.alpha * (loss.item() - self.W_init[loss_id])
                losses.append(loss/(abs(self.W_init[loss_id])+1e-5)*self.scale)
        
        return losses

    def forward(self, input, target, mask=None):
        losses = self.partial_losses(input, target, mask=mask)
        return torch.stack(losses).sum(dim=0)

    def get_global_features(self, input):
        #check_and_warn_input_range(input, 0, 1, 'PerceptualLoss input in get_global_features')

        if self.normalize_inputs:
            features_input = self.do_normalize_inputs(input)
        else:
            features_input = input

        features_input = self.vgg(features_input)
        return features_input
