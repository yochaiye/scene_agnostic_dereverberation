#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 18:32:53 2020

Implemented taken from
https://github.com/FrederikWarburg/Burst-Image-Deblurring/blob/23ea503c9c2c27567308da13a2d28d28629e8c25/burstloss.py#L6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np



class GradL1Loss(_Loss):
    '''
    Computes the images gradients loss suggested in "Burst Image Deblurring"
    '''
    def __init__(self, device, size_average=None, reduce=None, reduction='mean'):
        super(GradL1Loss, self).__init__(size_average, reduce, reduction)

        self.reduction = reduction

        prewitt_filter = 1 / 6 * np.array([[1, 0, -1],
                                           [1, 0, -1],
                                           [1, 0, -1]])

        self.prewitt_filter_horizontal = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                                         kernel_size=prewitt_filter.shape,
                                                         padding=prewitt_filter.shape[0] // 2).to(device)

        self.prewitt_filter_horizontal.weight.data.copy_(torch.from_numpy(prewitt_filter).to(device))
        self.prewitt_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])).to(device))

        self.prewitt_filter_vertical = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                                       kernel_size=prewitt_filter.shape,
                                                       padding=prewitt_filter.shape[0] // 2).to(device)

        self.prewitt_filter_vertical.weight.data.copy_(torch.from_numpy(prewitt_filter.T).to(device))
        self.prewitt_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])).to(device))

    def get_gradients(self, img):
        nc = img.shape[1]
        img_r = img[:, 0:1, :, :]
        # if nc == 1:
        grad_x = self.prewitt_filter_horizontal(img_r)
        grad_y = self.prewitt_filter_vertical(img_r)
        # elif nc == 2:
        #     img_g = img[:, 1:2, :, :]
        #     grad_x_r = self.prewitt_filter_horizontal(img_r)
        #     grad_y_r = self.prewitt_filter_vertical(img_r)
        #     grad_x_g = self.prewitt_filter_horizontal(img_g)
        #     grad_y_g = self.prewitt_filter_vertical(img_g)

        #     grad_x = torch.cat([grad_x_r, grad_x_g], dim=1)
        #     grad_y = torch.cat([grad_y_r, grad_y_g], dim=1)

        grad = torch.cat([grad_x, grad_y], dim=1)

        return grad

    def forward(self, input, target):
        input_grad = self.get_gradients(input)
        target_grad = self.get_gradients(target)

        return 0.1 * F.l1_loss(input, target, reduction=self.reduction) + F.l1_loss(input_grad, target_grad,
                                                                                    reduction=self.reduction)
    
    
class GradMSELoss(GradL1Loss):
    def forward(self, input, target):
        input_grad = self.get_gradients(input)
        target_grad = self.get_gradients(target)

        return 0.1 * F.mse_loss(input, target, reduction=self.reduction) + F.mse_loss(input_grad, target_grad,
                                                                                    reduction=self.reduction)
