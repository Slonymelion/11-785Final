# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 21:32:40 2019

@author: MomWithMe
"""
import numpy as np
import matplotlib.pyplot as plt

logfile = 'train.log'

loss_d, loss_g = [], []
with open(logfile, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        if 'loss' in line:
            x = [y.split(']') for y in line.split(':')]
            loss_d.append(float(x[-2][0]))
            loss_g.append(float(x[-1][0]))

loss_d = np.array(loss_d)
loss_g = np.array(loss_g)
x = np.arange(len(loss_d)) * 5
plt.plot(x, loss_g)
plt.xlabel('Number of batches')
plt.ylabel('Generator loss')
plt.show()

plt.plot(x, loss_d)
plt.ylim([-1, 1])
plt.xlabel('Number of batches')
plt.ylabel('Discriminator loss')
plt.show()
        
