# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:28:17 2019
read in the loss function and plot it
@author: MomWithMe
"""
import matplotlib.pyplot as plt

# All numerical values recorded
d_loss = []
g_loss = []
g_feature_loss = []
g_image_loss = []
g_first_grad = []
g_last_grad = []
d_first_grad = []
d_last_grad = []
all_losses = [d_loss, g_loss, g_feature_loss, g_image_loss,
              g_first_grad, g_last_grad, d_first_grad, d_last_grad]

pattern_d = 'D loss: '
pattern_g = 'G loss: '
pattern_gf = 'G feature loss: '
pattern_gi = 'G image loss: '
pattern_gfg = 'G first grad: '
pattern_glg = 'G final grad: '
pattern_dfg = 'D first grad: '
pattern_dlg = 'D final grad: '
all_patterns = [pattern_d, pattern_g, pattern_gf, pattern_gi, 
                pattern_gfg, pattern_glg, pattern_dfg, pattern_dlg]

window = 8
with open('../results/1556090881/train.log', 'r') as f:
    for line in f.readlines():
        for (loss, pattern) in zip(all_losses, all_patterns):
            s = line.find(pattern)
            if s >= 0:
                start, end = s+len(pattern), s+len(pattern)+window
                loss.append(float(line[start:end]))


#
#plt.plot(g_image_loss, 'r')
#plt.plot(g_feature_loss, 'g')
#plt.show()

def plotfn(a, b, c1, c2):
    fig, ax1 = plt.subplots()
    ax1.plot(a, color=c1)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(b, color=c2)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

plt.plot(g_loss)
plt.plot(d_loss, 'k')
plt.show()

plotfn(g_image_loss, g_feature_loss, 'r', 'g')
plotfn(g_first_grad, g_last_grad, 'b', 'r')
plotfn(d_first_grad, d_last_grad, 'b', 'r')

print('Done!')
                