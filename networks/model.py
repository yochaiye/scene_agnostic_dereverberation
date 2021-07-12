#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 03:04:01 2020

@author: yochai_yemini
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


def unet_conv(nc, output_nc, kernel_size, outermost=False, innermost=False):
    padding = {4: 1, (8, 4): (3, 1), (4, 8): (1, 3)}
    downrelu = nn.LeakyReLU(0.2, True)
    # downrelu = nn.ELU(inplace=True)
    downconv = nn.Conv2d(nc, output_nc, kernel_size, stride=2, padding=padding[kernel_size])    
    if outermost:
        return nn.Sequential(*[downconv, downrelu])
    elif innermost:
        downnorm = nn.BatchNorm2d(output_nc)
        return nn.Sequential(*[downconv, downnorm, nn.ReLU(True)])
    else:
        downnorm = nn.BatchNorm2d(output_nc)
        return nn.Sequential(*[downconv, downnorm, downrelu])
        


def unet_upconv(nc, output_nc, kernel_size, use_drop=False, outermost=False):
    padding = {4: 1, (8, 4): (3, 1), (4, 8): (1, 3)}
    uprelu = nn.ReLU(True)
    upconv = nn.ConvTranspose2d(nc, output_nc, kernel_size, stride=2, padding=padding[kernel_size])
    upnorm = nn.BatchNorm2d(output_nc)
    updrop = nn.Dropout(0.5)
    # uprelu = nn.ELU(inplace=True)
    if not outermost:
        if use_drop:
            return nn.Sequential(*[upconv, upnorm, updrop, uprelu])
        else:
            return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Tanh()])
        

def post_unet(nc):
    postconv1 = nn.Conv2d(nc, nc, 3, stride=1, padding=1)
    postnorm = nn.BatchNorm2d(nc)
    postrelu = nn.ReLU(True)
    postconv2 = nn.Conv2d(nc, nc, 3, stride=1, padding=1)
    return nn.Sequential(*[postconv1, postnorm, postrelu, postconv2, nn.Tanh()])


class UNet(nn.Module):
    def __init__(self, ngf=64, nc=1, kernel_size=4):
        super().__init__()
        self.nc = nc
        # initialise layers
        self.convlayer1 = unet_conv(nc, ngf, kernel_size, outermost=True)
        self.convlayer2 = unet_conv(ngf, ngf * 2, kernel_size)
        self.convlayer3 = unet_conv(ngf * 2, ngf * 4, kernel_size)
        self.convlayer4 = unet_conv(ngf * 4, ngf * 8, kernel_size)
        self.convlayer5 = unet_conv(ngf * 8, ngf * 8, kernel_size)
        self.convlayer6 = unet_conv(ngf * 8, ngf * 8, kernel_size)
        self.convlayer7 = unet_conv(ngf * 8, ngf * 8, kernel_size)
        self.convlayer8 = unet_conv(ngf * 8, ngf * 8, kernel_size, innermost=True)
        self.upconvlayer1 = unet_upconv(ngf * 8, ngf * 8, kernel_size, use_drop=True) 
        self.upconvlayer2 = unet_upconv(ngf * 16, ngf * 8, kernel_size, use_drop=True)
        self.upconvlayer3 = unet_upconv(ngf * 16, ngf * 8, kernel_size, use_drop=True)
        self.upconvlayer4 = unet_upconv(ngf * 16, ngf * 8, kernel_size)
        self.upconvlayer5 = unet_upconv(ngf * 16, ngf * 4, kernel_size)
        self.upconvlayer6 = unet_upconv(ngf * 8, ngf * 2, kernel_size)
        self.upconvlayer7 = unet_upconv(ngf * 4, ngf, kernel_size)
        self.upconvlayer8 = unet_upconv(ngf * 2, 1, kernel_size, outermost=True)
        self.dim = 1    # dimension to concatenate along for skip connections. 1 = feature axis
        
    def forward(self, x):
        if self.nc > 1:
            x.squeeze_()
        conv1feature = self.convlayer1(x)               # [64, 128, 128]
        conv2feature = self.convlayer2(conv1feature)    # [128, 64, 64]
        conv3feature = self.convlayer3(conv2feature)    # [256, 32, 32]
        conv4feature = self.convlayer4(conv3feature)    # [512, 16, 16]
        conv5feature = self.convlayer5(conv4feature)    # [512, 8, 8]
        conv6feature = self.convlayer6(conv5feature)    # [512, 4, 4]
        conv7feature = self.convlayer7(conv6feature)    # [512, 2, 2]
        
        bottleneck = self.convlayer8(conv7feature)      # [512, 1, 1]
        
        upconv1feature = self.upconvlayer1(bottleneck)  # [512, 2, 2]
        upconv2feature = self.upconvlayer2(torch.cat((upconv1feature, conv7feature), self.dim))    # [512, 4, 4]
        upconv3feature = self.upconvlayer3(torch.cat((upconv2feature, conv6feature), self.dim))    # [512, 8, 8]
        upconv4feature = self.upconvlayer4(torch.cat((upconv3feature, conv5feature), self.dim))    # [512, 16, 16]
        upconv5feature = self.upconvlayer5(torch.cat((upconv4feature, conv4feature), self.dim))    # [256, 32, 32]
        upconv6feature = self.upconvlayer6(torch.cat((upconv5feature, conv3feature), self.dim))    # [128, 64, 64]
        upconv7feature = self.upconvlayer7(torch.cat((upconv6feature, conv2feature), self.dim))    # [64, 128, 128]
        output = self.upconvlayer8(torch.cat((upconv7feature, conv1feature), self.dim))            # [nc, 256, 256]
        return output
