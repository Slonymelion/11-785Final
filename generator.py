# -*- coding: utf-8 -*-
"""
11-785 - Final Project - Baseline model based on WGAN-GP
"""
import torch
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


class SoundCNN(nn.Module):
    """
    Simple embedding network for processing sound data
    """
    def __init__(self, opt):        
        super(SoundCNN, self).__init__()
        self.opt = opt
        kernel_size = 3
        stride = 2
        padding = kernel_size // 2
        in_channels = 64  # input sound frequency channels
        out_channels = [in_channels, 256, 384, 576, 64]
        relu_slope = 0.2
        # build up network
        self.layers = []
        for i in range(len(out_channels)-2):
            self.layers.extend([
                    nn.Conv1d(out_channels[i], out_channels[i+1], kernel_size=kernel_size,
                               padding=padding, stride=stride),
                    nn.BatchNorm1d(out_channels[i+1]),
                    nn.LeakyReLU(negative_slope=relu_slope)
                    ])
        self.layers = nn.Sequential(*self.layers) 
        # last convoluation layer
        self.conv = nn.Conv1d(out_channels[-2], out_channels[-1],
                              kernel_size=kernel_size, padding=padding, stride=stride)
        self.avgpool = nn.AvgPool1d(out_channels[-1])       
    
    def forward(self, sound):
        x = self.layers(sound.transpose(1,2))
        x = self.conv(x)
        out = self.avgpool(x)
        return out


class ConditionalGen(nn.Module):
    """
    Generator that takes in both latent variable and sound features
    """
    def __init__(self, opt, SoundNet=None):        
        super(ConditionalGen, self).__init__()
        self.opt = opt
        
        self.soundnet = SoundNet
        
        in_feat = 164 if self.soundnet else 100  # 100 dimension latent variable + 64 dimensional sound
        dim = 128 # output image dimension
        ks = 4  # kernel size
        s = 1   # stride
        p = 0   # padding
        ns = 0.2 # negative slope for LeakyRelu
        
        self.layers = []
        self.layers.extend([
                nn.ConvTranspose2d(in_feat, dim*4, kernel_size=ks, stride=s, padding=p),
                nn.BatchNorm2d(dim*4),
                nn.LeakyReLU(negative_slope=ns)
                ])
        self.layers.extend([
                nn.ConvTranspose2d(dim*4, dim*4, kernel_size=ks, stride=4, padding=3),
                nn.BatchNorm2d(dim*4),
                nn.LeakyReLU(negative_slope=ns)
                ])
        self.layers.extend([
                nn.ConvTranspose2d(dim*4, dim*2, kernel_size=ks, stride=4, padding=4),
                nn.BatchNorm2d(dim*2),
                nn.LeakyReLU(negative_slope=ns)
                ])
        self.layers.extend([
                nn.ConvTranspose2d(dim*2, dim, kernel_size=ks, stride=2, padding=1),
                nn.BatchNorm2d(dim),
                nn.LeakyReLU(negative_slope=ns)
                ])
        self.layers = nn.Sequential(*self.layers)
        # add a final transpose convoluation layer to generate 3 channels
        self.finallayer = nn.ConvTranspose2d(dim, 3, kernel_size=ks, stride=2, padding=1)
        
        self.finaltanh = nn.Tanh()
    
    def forward(self, z, sound=None):
        if sound is not None and self.soundnet is not None:
            ins = self.soundnet(sound)
            ins = torch.cat([z, ins.unsqueeze(3)], dim=1)
        else:
            ins = z
        img = self.layers(ins)
        img = self.finallayer(img)
        out = self.finaltanh(img)

        return out
    
    
class CGen(nn.Module):
    """
    Generator that takes in both latent variable and sound features
    """
    def __init__(self, opt, SoundNet=None):        
        super(CGen, self).__init__()
        self.opt = opt
        
        self.soundnet = SoundNet
        
        in_feat = opt['latent_dim']+64 if self.soundnet else opt['latent_dim']  # 100 dimension latent variable + 64 dimensional sound
        dim = 128 # output image dimension
        ks = 4  # kernel size
        s = 1   # stride
        p = 0   # padding
        ns = 0.2 # negative slope for LeakyRelu
        
        self.layers = []
        self.layers.extend([
                nn.ConvTranspose2d(in_feat, dim*8, kernel_size=ks, stride=s, padding=p, bias=False),
                nn.BatchNorm2d(dim*8),
                nn.LeakyReLU(negative_slope=ns)
                ])
        self.layers.extend([
                nn.ConvTranspose2d(dim*8, dim*4, kernel_size=ks, stride=4, padding=3, bias=False),
                nn.BatchNorm2d(dim*4),
                nn.LeakyReLU(negative_slope=ns)
                ])
        self.layers.extend([
                nn.ConvTranspose2d(dim*4, dim*2, kernel_size=ks, stride=4, padding=4, bias=False),
                nn.BatchNorm2d(dim*2),
                nn.LeakyReLU(negative_slope=ns)
                ])
        self.layers.extend([
                nn.ConvTranspose2d(dim*2, dim, kernel_size=ks, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.LeakyReLU(negative_slope=ns)
                ])
        self.layers = nn.Sequential(*self.layers)
        # add a final transpose convoluation layer to generate 3 channels
        self.finallayer = nn.ConvTranspose2d(dim, 3, kernel_size=ks, stride=2, padding=1, bias=False)
        
        self.finaltanh = nn.Tanh()
    
    def forward(self, z, sound=None):
        if sound is not None and self.soundnet is not None:
            ins = self.soundnet(sound)
            ins = torch.cat([z, ins.unsqueeze(3)], dim=1)
        else:
            ins = z
        img = self.layers(ins)
        img = self.finallayer(img)
        out = self.finaltanh(img)

        return out