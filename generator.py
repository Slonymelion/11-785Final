# -*- coding: utf-8 -*-
"""
11-785 - Final Project - Baseline model based on WGAN-GP
"""
import torch.nn as nn
import torch.nn.functional as F


class Upblock(nn.Module):
    def __init__(self, in_channel, out_channel, 
                 scale_factor=2, kernel_size=1, stride=1,
                 padding=0, negative_slope=0.2):
        super(Upblock, self).__init__()
        
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leakyrelu = nn.LeakyReLU(negative_slope=negative_slope)
    
    def forward(self, x):
        # upscaling first
        out = F.interpolate(x, scale_factor=self.scale_factor)
        out = self.conv(out)
        out = self.leakyrelu(out)
        return out
                       
                       
class CNN(nn.Module):
    def __init__(self, opt):        
        super(CNN, self).__init__()
        self.opt = opt
        
        in_feat = opt['in_feat']
        out_feat = opt['out_feat'] if 'out_feat' in opt else 3  # should be 3 RGB channels
        input_size = opt['img_size']
        scale_factor = opt['scale_factor']
        final_size = input_size
        
#        def block(in_feat, out_feat, normalize=False):
#            layers = [nn.UpsamplingBilinear2d()]
#            if normalize:
#                layers.append(nn.BatchNorm1d(out_feat, 0.8))
#            layers.append(nn.LeakyReLU(0.2, inplace=True))
#            return layers
        self.convfirst = nn.Conv2d(in_channels=in_feat, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.layers = []
        self.layers.append(Upblock(256, 128, scale_factor=scale_factor))
        self.layers.append(Upblock(128, 64, scale_factor=scale_factor))
        self.layers.append(Upblock(64, 32, scale_factor=scale_factor))
        self.layers = nn.Sequential(*self.layers)
        
        final_size = final_size * scale_factor ** len(self.layers)
        final_kernel = 32 - out_feat + 1
        self.convlast = nn.Conv1d(final_size ** 2, final_size ** 2, kernel_size=final_kernel, stride=1)
        self.tanh = nn.Sigmoid()  # do not use sigmoid()
        
        self.final_size = final_size


    def forward(self, z):
        img = self.convfirst(z)
        img = self.layers(img)
        img = img.view(img.shape[0], -1, img.shape[1])
        img = self.convlast(img)
        
        img = img.view(img.shape[0], -1, self.final_size, self.final_size)
        img = self.tanh(img)
        return img


