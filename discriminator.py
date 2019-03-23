"""
Dual Path Network as discriminator
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.modules.distance import CosineSimilarity
import torchvision
from PIL import Image

import os
import sys
import numpy as np


# module definition - basic building block for DPN
class Bottleneck(nn.Module):
    def __init__(self, previous, first, last, increment, stride=1, is_first_layer=True, negative_slope=0.2):
        outf = last+increment
        super(Bottleneck, self).__init__()
        self.last = last
        self.layers = []
        self.layers.append(nn.Conv2d(in_channels=previous, out_channels=first, kernel_size=1, bias=False))
        self.layers.append(nn.BatchNorm2d(first))
        self.layers.append(nn.LeakyReLU(negative_slope=negative_slope))
        self.layers.append(nn.Conv2d(in_channels=first, out_channels=first, kernel_size=3,
                                     groups=32, stride=stride, padding=1, bias=False))
        self.layers.append(nn.BatchNorm2d(first))
        self.layers.append(nn.LeakyReLU(negative_slope=negative_slope))
        self.layers.append(nn.Conv2d(in_channels=first, out_channels=outf, kernel_size=1, bias=False))
        self.layers.append(nn.BatchNorm2d(outf))
        self.layers = nn.Sequential(*self.layers)

        self.shortcut = nn.Sequential()  # residual path
        # add one layer for possible size change during residual connection
        if is_first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv2d(previous, outf, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outf)
            )
        self.outlayer = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        densepath = self.layers(x)
        respath = self.shortcut(x)
        out = torch.cat([respath[:, :self.last, :, :]+densepath[:, :self.last, :, :],
                            respath[:, self.last:, :, :], densepath[:, self.last:, :, :]], dim=1)
        out = self.outlayer(out)
        return out
    

# model definition - DualPathNet
class DualPathNet(nn.Module):
    """
    General class to build DPN like network.
    Input: num_feats - color channels (3) of input images
           num_classes - number of classes (2300 for classification)
           structure - dictionary variable for network architecture
           task - either 'classify' or 'verify', specify the current task.
    """
    def __init__(self, num_feats, num_classes, structure):
        # input parsing
        in_sizes = structure['ins']
        out_sizes = structure['outs']
        block_repeats = structure['repeats']
        incre_sizes = structure['increments']
        init_hidden = structure['initial']
        k = structure['kernel']
        stride = structure['stride']
        negative_slope = structure['negative_slope']
        # initialization
        super(DualPathNet, self).__init__()
        self.prev = init_hidden
        
        # build network, input image size 32 by 32
#        self.padding = nn.modules.ConstantPad2d(image_height//2, 0)  # pad first to get to 64 by 64
        
        # first block is fully conv2d, max pooling is removed since input image is already small
        self.firstblock = []
        self.firstblock.append(nn.Conv2d(in_channels=num_feats, out_channels=init_hidden, padding=k//2, kernel_size=k, 
                               stride=1, bias=False))  # original paper uses stride=2, use 1 here because of small image size
        self.firstblock.append(nn.BatchNorm2d(num_features=init_hidden))
        self.firstblock.append(nn.LeakyReLU(negative_slope=negative_slope))
#        self.firstblock.append(nn.MaxPool2d(3))  # max pooling used in original paper, may not be needed
        self.firstblock = nn.Sequential(*self.firstblock)
                
        # Bottleneck blocks
        self.layers = []
        for i in range(len(in_sizes)):
            repeat = block_repeats[i]
            first = in_sizes[i]
            last = out_sizes[i]
            increment = incre_sizes[i]
            self.layers.extend(self.conv_layer(first, last, increment, repeat, stride))        
        self.layers = nn.Sequential(*self.layers)
        
        # fully connected layers
        last_out = out_sizes[-1] + (block_repeats[-1]+1)* incre_sizes[-1]
        self.fc_layers = nn.Linear(last_out, num_classes, bias=False)
        self.lastsig = nn.Sigmoid()
    
    def conv_layer(self, first, last, increment, repeat, stride):
        strides = [stride] + [1] * (repeat - 1)
        block = []
        for i, s in enumerate(strides):
            block.append(Bottleneck(self.prev, first, last, increment, stride=s, is_first_layer= (i==0)))
            self.prev = last + (i+2) * increment
        return block
        
    def forward(self, x, evalMode=False):
#        output = self.padding(x)
        output = self.firstblock(x)
        output = self.layers(output)
        
        # pooling layer is here
        output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1)
        output = output.reshape(output.shape[0], output.shape[1])
        
        output = self.fc_layers(output)
        output = self.lastsig(output)

        return output
    

# weight initialization function
def init_weights(m, mode='xavier'):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        if mode == 'xavier':
            torch.nn.init.xavier_normal_(m.weight.data)
        elif mode == 'he':
            torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        

# macro for several DPN architecture
def DPNmini(num_feats, num_classes, kernel=3, stride=2, negative_slope=0.2):
    structure = {}
    structure['ins'] = [96, 128]
    structure['outs'] = [256, 512]
    structure['repeats'] = [2, 2, 2]
    structure['increments'] = [16, 24]
    structure['initial'] = 64
    structure['kernel'] = kernel
    structure['stride'] = stride
    structure['negative_slope'] = negative_slope
    
    return DualPathNet(num_feats, num_classes, structure)


def DPN26(num_feats, num_classes, kernel=3, stride=2, task='classify'):
    structure = {}
    structure['ins'] = [96, 192, 384, 768]
    structure['outs'] = [256, 512, 1024, 2048]
    structure['repeats'] = [2, 2, 2, 2]
    structure['increments'] = [16, 32, 24, 128]
    structure['initial'] = 64
    structure['kernel'] = kernel
    structure['stride'] = stride
    
    return DualPathNet(num_feats, num_classes, structure, task=task)


def DPN26small(num_feats, num_classes, kernel=3, stride=2, task='classify'):
    structure = {}
    structure['ins'] = [96, 384, 768]
    structure['outs'] = [256, 1024, 2048]
    structure['repeats'] = [2, 2, 2]
    structure['increments'] = [16, 24, 128]
    structure['initial'] = 64
    structure['kernel'] = kernel
    structure['stride'] = stride
    
    return DualPathNet(num_feats, num_classes, structure, task=task)


def DPN50(num_feats, num_classes, kernel=3, stride=2, task='classify'):
    structure = {}
    structure['ins'] = [96, 192, 384, 768]
    structure['outs'] = [256, 512, 1024, 2048]
    structure['repeats'] = [3, 4, 6, 3]
    structure['increments'] = [16, 32, 24, 128]
    structure['initial'] = 64
    structure['kernel'] = kernel
    structure['stride'] = stride
    
    return DualPathNet(num_feats, num_classes, structure, task=task)


def DPN50small(num_feats, num_classes, kernel=3, stride=2, task='classify'):
    structure = {}
    structure['ins'] = [192, 384, 768]
    structure['outs'] = [512, 1024, 2048]
    structure['repeats'] = [3, 6, 4]
    structure['increments'] = [32, 24, 128]
    structure['initial'] = 64
    structure['kernel'] = kernel
    structure['stride'] = stride
    
    return DualPathNet(num_feats, num_classes, structure, task=task)


def DPN92(num_feats, num_classes, kernel=3, stride=2, task='classify'):
    structure = {}
    structure['ins'] = [96, 192, 384, 768]
    structure['outs'] = [256, 512, 1024, 2048]
    structure['repeats'] = [3, 4, 20, 3]
    structure['increments'] = [16, 32, 24, 128]
    structure['initial'] = 64
    structure['kernel'] = kernel
    structure['stride'] = stride
    
    return DualPathNet(num_feats, num_classes, structure, task=task)


# function to remove '._' files
def findNremove(path,pattern,maxdepth=1):
    cpath=path.count(os.sep)
    for r,d,f in os.walk(path):
        if r.count(os.sep) - cpath <maxdepth:
            for files in f:
                if files.startswith(pattern):
                    try:
                        #print "Removing %s" % (os.path.join(r,files))
                        os.remove(os.path.join(r,files))
                    except Exception as e:
                        print(e)
                    else:
                        print("{} removed".format(os.path.join(r,files)))