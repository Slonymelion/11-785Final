# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:28:17 2019
read in the loss function and plot it
@author: MomWithMe
"""
import matplotlib.pyplot as plt

# All numerical values recorded
d_loss = []
d_gp_loss = []
g_loss = []
g_feature_loss = []
g_image_loss = []
g_first_grad = []
g_last_grad = []
g_res_grad = []
d_first_grad = []
d_last_grad = []

all_losses = [d_loss, d_gp_loss, g_loss, g_feature_loss, g_image_loss,
              g_first_grad, g_last_grad, g_res_grad, d_first_grad, d_last_grad]

p_d = 'D loss'
p_dgp = 'D GP loss'
p_g = 'G loss'
p_gf = 'G feature loss'
p_gi = 'G image loss'
p_gfg = 'G first grad'
p_glg = 'G final grad'
p_gresg = 'G respath first grad'
p_dfg = 'D first grad'
p_dlg = 'D final grad'
all_patterns = [p_d, p_dgp, p_g, p_gf, p_gi, 
                p_gfg, p_glg, p_gresg, p_dfg, p_dlg]

window = 8

file = '../results/1556255736/train.log'
with open(file, 'r') as f:
    for line in f.readlines():
        for (loss, pattern) in zip(all_losses, all_patterns):
            s = line.find(pattern)
            if s >= 0:
                start = s+len(pattern)+2
                end = start + window
                loss.append(float(line[start:end]))


#
#plt.plot(g_image_loss, 'r')
#plt.plot(g_feature_loss, 'g')
#plt.show()

def plotfn(a, b, c1, c2, l1='Curve 1', l2='Curve 2'):
    fig, ax1 = plt.subplots()
    ax1.plot(a, color=c1)
    ax1.set_ylabel(l1, color=c1)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(b, color=c2)
    ax2.set_ylabel(l2, color=c2)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

plt.plot(g_loss, label='Generator loss')
plt.plot(d_loss, 'k', label='Discriminator loss')
#plt.plot(d_gp_loss, 'y')
plt.tight_layout()
plt.legend(loc='best')
plt.show()

plt.plot(d_gp_loss)
plt.tight_layout()
plt.title('Gradient Penaly Loss')
plt.show()

plotfn(g_image_loss, g_feature_loss, 'r', 'g', p_gi, p_gf)
plotfn(g_first_grad, g_last_grad, 'b', 'r', p_gfg, p_glg)
plotfn(d_first_grad, d_last_grad, 'b', 'r', p_dfg, p_dlg)
plotfn(g_first_grad, g_res_grad, 'b', 'g', p_gfg, p_gresg)

print('Done!')
                