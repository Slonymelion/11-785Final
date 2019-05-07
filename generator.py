# -*- coding: utf-8 -*-
"""
11-785 - Final Project - Baseline model based on WGAN-GP
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


#class Upblock(nn.Module):
#    def __init__(self, in_channel, out_channel, 
#                 scale_factor=2, kernel_size=1, stride=1,
#                 padding=0, negative_slope=0.2):
#        super(Upblock, self).__init__()
#        
#        self.scale_factor = scale_factor
#        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
#        self.leakyrelu = nn.LeakyReLU(negative_slope=negative_slope)
#    
#    def forward(self, x):
#        # upscaling first
#        out = F.interpolate(x, scale_factor=self.scale_factor)
#        out = self.conv(out)
#        out = self.leakyrelu(out)
#        return out
                       
                       
#class CNN(nn.Module):
#    def __init__(self, opt):        
#        super(CNN, self).__init__()
#        self.opt = opt
#        
#        in_feat = opt['in_feat']
#        out_feat = opt['out_feat'] if 'out_feat' in opt else 3  # should be 3 RGB channels
#        input_size = opt['img_size']
#        scale_factor = opt['scale_factor']
#        final_size = input_size
#        
##        def block(in_feat, out_feat, normalize=False):
##            layers = [nn.UpsamplingBilinear2d()]
##            if normalize:
##                layers.append(nn.BatchNorm1d(out_feat, 0.8))
##            layers.append(nn.LeakyReLU(0.2, inplace=True))
##            return layers
#        self.convfirst = nn.Conv2d(in_channels=in_feat, out_channels=256, kernel_size=1, stride=1, padding=0)
#        self.layers = []
#        self.layers.append(Upblock(256, 128, scale_factor=scale_factor))
#        self.layers.append(Upblock(128, 64, scale_factor=scale_factor))
#        self.layers.append(Upblock(64, 32, scale_factor=scale_factor))
#        self.layers = nn.Sequential(*self.layers)
#        
#        final_size = final_size * scale_factor ** len(self.layers)
#        final_kernel = 32 - out_feat + 1
#        self.convlast = nn.Conv1d(final_size ** 2, final_size ** 2, kernel_size=final_kernel, stride=1)
#        self.tanh = nn.Sigmoid()  # do not use sigmoid()
#        
#        self.final_size = final_size
#
#
#    def forward(self, z):
#        img = self.convfirst(z)
#        img = self.layers(img)
#        img = img.view(img.shape[0], -1, img.shape[1])
#        img = self.convlast(img)
#        
#        img = img.view(img.shape[0], -1, self.final_size, self.final_size)
#        img = self.tanh(img)
#        return img


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
        out_channels = [in_channels, 256, 384, 576, opt['soundnet_out_dim']]
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


#Generator Model 1
class ConditionalGen(nn.Module):
    """
    Generator that takes in both latent variable and sound features
    """
    def __init__(self, opt, SoundNet=None):        
        super(ConditionalGen, self).__init__()
        self.opt = opt
        
        self.soundnet = SoundNet
        
        in_feat = opt['soundnet_out_dim']+opt['latent_dim'] if self.soundnet else opt['latent_dim']  # 100 dimension latent variable + 64 dimensional sound
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
        out = (out+1)/2

        return out
    

# Generator model 2:  
class CGen(nn.Module):
    """
    Generator that takes in both latent variable and sound features
    """
    def __init__(self, opt, SoundNet=None):        
        super(CGen, self).__init__()
        self.opt = opt
        
        self.soundnet = SoundNet
        
        in_feat = opt['latent_dim']+opt['soundnet_out_dim'] if self.soundnet else opt['latent_dim']  # 100 dimension latent variable + 64 dimensional sound
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
            if z is not None:
                ins = torch.cat([z, ins.unsqueeze(3)], dim=1)
            else:
                ins = ins.unsqueeze(3)
        else:
            ins = z
        img = self.layers(ins)
        img = self.finallayer(img)
        out = self.finaltanh(img)
        out = (out+1)/2

        return out


# Generator Model 3: With residual path
class CResGen(nn.Module):
    """
    Generator that takes in both latent variable and sound features
    """
    def __init__(self, opt, SoundNet=None):        
        super(CResGen, self).__init__()
        self.opt = opt
        
        self.soundnet = SoundNet
        
        in_feat = opt['latent_dim']+opt['soundnet_out_dim'] if self.soundnet else opt['latent_dim']  # 100 dimension latent variable + 64 dimensional sound
        dim = 128 # output image dimension
        ks = 4  # kernel size
        s = 1   # stride
        p = 0   # padding
        ns = 0.2 # negative slope for LeakyRelu
        
        def block(in_feat, out_feat, ks, s, p):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel_size=ks, stride=s, padding=p, bias=False),
                      nn.BatchNorm2d(out_feat)]
            return nn.Sequential(*layers)
        
        def resblock(in_feat, out_feat, ks=1, s=1, p=0):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=ks, stride=s, padding=p, bias=False),
                      nn.BatchNorm2d(out_feat)]
            return nn.Sequential(*layers)
            
        self.layers = []
        self.layers.append(block(in_feat, dim*8, ks, s, p))
        self.layers.append(block(dim*8, dim*4, ks, 2, 0))
        self.layers.append(block(dim*4, dim*2, ks, 4, 4))
        self.layers.append(block(dim*2, dim, ks, 2, 1))
        self.layers = nn.Sequential(*self.layers)
        
        self.shortcut = []
#        self.shortcut.append(block(in_feat, dim*4, 10, 1, 0))
#        self.shortcut.append(block(dim*4, dim, 9, 7, 4))
        self.shortcut.append(resblock(in_feat, dim*4))
        self.shortcut.append(resblock(in_feat, dim))
        self.shortcut = nn.Sequential(*self.shortcut)
        
#        self.lrelu = nn.LeakyReLU(negative_slope=ns)
        self.lrelu = nn.ReLU()
        
        # add a final transpose convoluation layer to generate 3 channels
        self.finallayer = nn.ConvTranspose2d(dim, 3, kernel_size=ks, stride=2, padding=1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
#        self.tanh = nn.Tanh()
    
    def forward(self, z, sound=None):
        if sound is not None and self.soundnet is not None:
            x = self.soundnet(sound)
            if z is not None:
                x = torch.cat([z, x.unsqueeze(3)], dim=1)
            else:
                x = x.unsqueeze(3)
        else:
            x = z
        
        # residual path every 2 blocks
        ins = self.layers[0](x)
        ins = self.lrelu(ins)
        ins = self.layers[1](ins)
        res = self.shortcut[0](x)
        ins = self.lrelu(ins+res)
        
        ins = self.layers[2](ins)
        ins = self.lrelu(ins)
        ins = self.layers[3](ins)
        res = self.shortcut[1](x)
        ins = self.lrelu(ins+res)

        img = self.finallayer(ins)
        out = self.sigmoid(img)
        
        return out
    

# Generator Model 3: With residual path
class CResGenDeep(nn.Module):
    """
    Generator that takes in both latent variable and sound features
    """
    def __init__(self, opt, SoundNet=None):        
        super(CResGenDeep, self).__init__()
        self.opt = opt
        
        self.soundnet = SoundNet
        
        in_feat = opt['latent_dim']+opt['soundnet_out_dim'] if self.soundnet else opt['latent_dim']  # 100 dimension latent variable + 64 dimensional sound
        dim = 128 # output image dimension
        ks = 4  # kernel size
        s = 1   # stride
        p = 0   # padding
#        ns = 0.2 # negative slope for LeakyRelu
        
        def block(in_feat, out_feat, ks, s, p):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel_size=ks, stride=s, padding=p, bias=False),
                      nn.BatchNorm2d(out_feat)]
            return nn.Sequential(*layers)
        
        def convblock(in_feat, out_feat, ks, s, p):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=ks, stride=s, padding=p, bias=False),
                      nn.BatchNorm2d(out_feat)]
            return nn.Sequential(*layers)
        
        def resblock(in_feat, out_feat, ks=1, s=1, p=0):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel_size=ks, stride=s, padding=p, bias=False),
                      ]
#                      nn.BatchNorm2d(out_feat)]
            return nn.Sequential(*layers)
            
        self.layers = []
        self.layers.append(block(in_feat, dim*8, ks, s, p))
        self.layers.append(block(dim*8, dim*4, ks, 2, 0))
        self.layers.append(block(dim*4, dim*2, ks, 4, 4))
        self.layers.append(block(dim*2, dim, ks, 2, 1))
        self.layers.append(block(dim, dim//2, ks, 2, 1))
        self.layers.append(convblock(dim//2, 3, 5, 1, 2))
        self.layers = nn.Sequential(*self.layers)
        
        self.shortcut = []
#        self.shortcut.append(block(in_feat, dim*4, 10, 1, 0))
#        self.shortcut.append(block(dim*4, dim, 9, 7, 4))
        self.shortcut.append(resblock(in_feat, dim*2, 32, 1, 0))
        self.shortcut.append(resblock(dim*2, 3, 4, 4, 0))
        self.shortcut = nn.Sequential(*self.shortcut)
        
#        self.lrelu = nn.LeakyReLU(negative_slope=ns)
        self.lrelu = nn.ReLU()
        
        # add a final transpose convoluation layer to generate 3 channels
#        self.finallayer = nn.ConvTranspose2d(dim, 3, kernel_size=ks, stride=2, padding=1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
#        self.tanh = nn.Tanh()
    
    def forward(self, z, sound=None):
        if sound is not None and self.soundnet is not None:
            x = self.soundnet(sound)
            if z is not None:
                x = torch.cat([z, x.unsqueeze(3)], dim=1)
            else:
                x = x.unsqueeze(3)
        else:
            x = z
        
        # residual path every 2 blocks
        ins = self.layers[0](x)
        ins = self.lrelu(ins)
        ins = self.layers[1](ins)
        ins = self.lrelu(ins)
        ins = self.layers[2](ins)
        res = self.shortcut[0](x)
        ins = self.lrelu(ins+res)
        
        ins = self.layers[3](ins)
        ins = self.lrelu(ins)
        ins = self.layers[4](ins)
        ins = self.lrelu(ins)
        ins = self.layers[5](ins)
        res = self.shortcut[1](res)
#        ins = self.lrelu(ins+res)

#        img = self.finallayer(ins)
#        out = self.sigmoid(img)
        out = self.sigmoid(ins+res)
        
        return out
    
    
# Generator Model 4: With residual path
class CResGenDeepDeep(nn.Module):
    """
    Generator that takes in both latent variable and sound features
    """
    def __init__(self, opt, SoundNet=None):        
        super(CResGenDeepDeep, self).__init__()
        self.opt = opt
        
        self.soundnet = SoundNet
        
        in_feat = opt['latent_dim']+opt['soundnet_out_dim'] if opt['use_latent'] else opt['soundnet_out_dim']  # 100 dimension latent variable + 64 dimensional sound
        dim = 128 # output image dimension
        ks = 4  # kernel size
        s = 1   # stride
        p = 0   # padding
#        ns = 0.2 # negative slope for LeakyRelu
        
        def block(in_feat, out_feat, ks, s, p):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel_size=ks, stride=s, padding=p, bias=False),
                      nn.BatchNorm2d(out_feat)]
            return nn.Sequential(*layers)
        
        def convblock(in_feat, out_feat, ks, s, p):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=ks, stride=s, padding=p, bias=False),
                      nn.BatchNorm2d(out_feat)]
            return nn.Sequential(*layers)
        
        def resblock(in_feat, out_feat, ks=1, s=1, p=0, conv=False):
            if conv:
                layers = [nn.Conv2d(in_feat, out_feat, kernel_size=ks, stride=s, padding=p, bias=False)]
            else:
                layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel_size=ks, stride=s, padding=p, bias=False),
                          ]
#                      nn.BatchNorm2d(out_feat)]
            return nn.Sequential(*layers)
            
        self.layers = []
        self.layers.append(block(in_feat, dim*8, ks, s, p))
        self.layers.append(block(dim*8, dim*4, ks, 2, 0))
        self.layers.append(block(dim*4, dim*2, ks, 4, 4))
        self.layers.append(block(dim*2, dim, ks, 2, 1))
        self.layers.append(block(dim, dim//2, ks, 2, 1))
        self.layers.append(convblock(dim//2, dim, 5, 1, 2))
        self.layers.append(convblock(dim, dim*2, 5, 1, 2))
        self.layers.append(convblock(dim*2, 3, 3, 1, 1))
        self.layers = nn.Sequential(*self.layers)
        
        self.shortcut = []
#        self.shortcut.append(block(in_feat, dim*4, 10, 1, 0))
#        self.shortcut.append(block(dim*4, dim, 9, 7, 4))
        self.shortcut.append(resblock(in_feat, dim*2, 32, 1, 0))
        self.shortcut.append(resblock(dim*2, dim, 4, 4, 0))
        self.shortcut.append(resblock(dim, 3, 3, 1, 1, conv=True))
        self.shortcut = nn.Sequential(*self.shortcut)
        
        self.lrelu = nn.ReLU()
                
        self.sigmoid = nn.Sigmoid()
#        self.tanh = nn.Tanh()
    
    def forward(self, z, x=None, sound=None):
        if sound is not None and self.soundnet is not None:
            x = self.soundnet(sound)
            if z is not None:
                x = torch.cat([z, x.unsqueeze(3)], dim=1)
            else:
                x = x.unsqueeze(3)
        elif x is None:
            x = z
        elif z is not None:
            x = torch.cat([z, x.unsqueeze(2).unsqueeze(3)], dim=1)
        else:
            x = x.unsqueeze(2).unsqueeze(3)
        
        # residual path every 2 blocks
        ins = self.layers[0](x)
        ins = self.lrelu(ins)
        ins = self.layers[1](ins)
        ins = self.lrelu(ins)
        ins = self.layers[2](ins)
        res = self.shortcut[0](x)
        ins = self.lrelu(ins+res)
        
        ins2 = self.layers[3](ins)
        ins2 = self.lrelu(ins2)
        ins2 = self.layers[4](ins2)
        ins2 = self.lrelu(ins2)
        ins2 = self.layers[5](ins2)
#        res = self.shortcut[1](res)
        res = self.shortcut[1](ins)
        ins = self.lrelu(ins2+res)
        
        ins2 = self.layers[6](ins)
        ins2 = self.lrelu(ins2)
        ins2 = self.layers[7](ins2)
#        res = self.shortcut[2](res)
        res = self.shortcut[2](ins)

#        img = self.finallayer(ins)
#        out = self.sigmoid(img)
        out = self.sigmoid(ins2+res)
        
        return out


# helper function: weight initialization
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)