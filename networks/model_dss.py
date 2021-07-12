#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 03:04:01 2020

@author: yochai_yemini
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
        

class MeanPoolSets(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        out = x.mean(dim=1, keepdim=False)
        return out
    
class MaxPoolSets(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        out = x.max(dim=1, keepdim=False)[0]
        return out


class Conv2dDeepSym(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_max=0, outermost=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_max = use_max
        self.outermost = outermost
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.conv_s = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        if not outermost:
            self.bn = nn.BatchNorm2d(num_features=out_channels)
            self.bns = nn.BatchNorm2d(num_features=out_channels)
            # nn.init.kaiming_normal_(self.conv.weight)
            # nn.init.kaiming_normal_(self.conv_s.weight)
        # else:
            # nn.init.xavier_normal_(self.conv.weight)
            # nn.init.xavier_normal_(self.conv_s.weight)
    
    def forward(self, x):
        b, n, c, h, w = x.size()
        x1 = self.conv(x.view(n * b, c, h, w))
        h, w = x1.shape[-2:]
        if self.use_max:
            x2 = self.conv_s(torch.max(x, dim=1, keepdim=False)[0])
        else:
            x2 = self.conv_s(torch.mean(x, dim=1, keepdim=False))
        if not self.outermost:
            x1 = self.bn(x1)
            x2 = self.bns(x2)
        # x2 = x2.view(b, 1, h, w, self.out_channels).repeat(1, n, 1, 1, 1).view(b * n, self.out_channels, h, w)
        x2 = x2.view(b, 1, self.out_channels, h, w).repeat(1, n, 1, 1, 1).view(b * n, self.out_channels, h, w)
        x = x1 + x2
        x = x.view(b, n, self.out_channels, h, w)
        return x
    
    
class ConvTranspose2dDeepSym(Conv2dDeepSym):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_max=0, outermost=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, use_max, outermost)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.conv_s = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding)


def unet_conv(nc, output_nc, kernel_size):
    downrelu = nn.LeakyReLU(0.2, True)
    padding = {4: 1, (8, 4): (3, 1), (4, 8): (1, 3)}
    # downrelu = nn.ELU(inplace=True)
    downconv = Conv2dDeepSym(nc, output_nc, kernel_size, stride=2, padding=padding[kernel_size])
    return nn.Sequential(*[downconv, downrelu])


def unet_upconv(nc, output_nc, kernel_size, outermost=False):
    uprelu = nn.ReLU(True)
    padding = {4: 1, (8, 4): (3, 1), (4, 8): (1, 3)}
    upconv = ConvTranspose2dDeepSym(nc, output_nc, kernel_size, stride=2, padding=padding[kernel_size], outermost=outermost)
    if not outermost:
        return nn.Sequential(*[upconv, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Tanh()])


def post_unet(nc):
    postconv1 = nn.Conv2d(nc, nc, 3, stride=1, padding=1)
    postnorm = nn.BatchNorm2d(nc)
    postrelu = nn.ReLU(True)
    postconv2 = nn.Conv2d(nc, nc, 3, stride=1, padding=1)
    return nn.Sequential(*[postconv1, postnorm, postrelu, postconv2, nn.Tanh()])


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Sym') == -1:
        m.weight.data.normal_(0.0, 0.02)
    elif (classname.find('BatchNorm2d') != -1) or (classname.find('InstanceNorm2d') != -1):
        if getattr(m, 'affine'):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        

class UNet(nn.Module):
    def __init__(self, ngf=64, nc=1, kernel_size=4):
        super().__init__()

        # initialise layers
        self.convlayer1 = unet_conv(nc, ngf, kernel_size)
        self.convlayer2 = unet_conv(ngf, ngf * 2, kernel_size)
        self.convlayer3 = unet_conv(ngf * 2, ngf * 4, kernel_size)
        self.convlayer4 = unet_conv(ngf * 4, ngf * 8, kernel_size)
        self.convlayer5 = unet_conv(ngf * 8, ngf * 8, kernel_size)
        self.convlayer6 = unet_conv(ngf * 8, ngf * 8, kernel_size)
        self.convlayer7 = unet_conv(ngf * 8, ngf * 8, kernel_size)
        self.convlayer8 = unet_conv(ngf * 8, ngf * 8, kernel_size)
        self.upconvlayer1 = unet_upconv(ngf * 8, ngf * 8, kernel_size)
        self.upconvlayer2 = unet_upconv(ngf * 16, ngf * 8, kernel_size)
        self.upconvlayer3 = unet_upconv(ngf * 16, ngf * 8, kernel_size)
        self.upconvlayer4 = unet_upconv(ngf * 16, ngf * 8, kernel_size)
        self.upconvlayer5 = unet_upconv(ngf * 16, ngf * 4, kernel_size)
        self.upconvlayer6 = unet_upconv(ngf * 8, ngf * 2, kernel_size)
        self.upconvlayer7 = unet_upconv(ngf * 4, ngf, kernel_size)
        self.upconvlayer8 = unet_upconv(ngf * 2, nc, kernel_size)
        self.postunet = post_unet(nc)
        self.max_pool = MaxPoolSets()
        self.cat_dim = 2    # dimension to concatenate along for skip connections. 2 = feature axis
        
    def forward(self, x):
        conv1feature = self.convlayer1(x)               # [64, 128, 128]
        conv2feature = self.convlayer2(conv1feature)    # [128, 64, 64]
        conv3feature = self.convlayer3(conv2feature)    # [256, 32, 32]
        conv4feature = self.convlayer4(conv3feature)    # [512, 16, 16]
        conv5feature = self.convlayer5(conv4feature)    # [512, 8, 8]
        conv6feature = self.convlayer6(conv5feature)    # [512, 4, 4]
        conv7feature = self.convlayer7(conv6feature)    # [512, 2, 2]
        
        bottleneck = self.convlayer8(conv7feature)      # [512, 1, 1]
        
        upconv1feature = self.upconvlayer1(bottleneck)  # [512, 2, 2]
        upconv2feature = self.upconvlayer2(torch.cat((upconv1feature, conv7feature), self.cat_dim))    # [512, 4, 4]
        upconv3feature = self.upconvlayer3(torch.cat((upconv2feature, conv6feature), self.cat_dim))    # [512, 8, 8]
        upconv4feature = self.upconvlayer4(torch.cat((upconv3feature, conv5feature), self.cat_dim))    # [512, 16, 16]
        upconv5feature = self.upconvlayer5(torch.cat((upconv4feature, conv4feature), self.cat_dim))    # [256, 32, 32]
        upconv6feature = self.upconvlayer6(torch.cat((upconv5feature, conv3feature), self.cat_dim))    # [128, 64, 64]
        upconv7feature = self.upconvlayer7(torch.cat((upconv6feature, conv2feature), self.cat_dim))    # [64, 128, 128]
        output = self.upconvlayer8(torch.cat((upconv7feature, conv1feature), self.cat_dim))            # [nc, 256, 256]
        output = self.max_pool(output)
        output = self.postunet(output)
        return output
