import torch
import torch.autograd as autograd
#import torch.nn as nn
#import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
#from torch.nn.modules.distance import CosineSimilarity
import torchvision
from torchvision.utils import save_image
from PIL import Image

import logging
import os
#import sys
import multiprocessing
import numpy as np

import discriminator as D
import generator as G


# image dataset class exclusively for test
class ImageDataset(Dataset):
    """
    return (image tensor, image file name) tuple
    """
    def __init__(self, path):
        file_list = os.listdir(path)
        self.file_list = [os.path.join(path, x) for x in file_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        imgname = os.path.split(self.file_list[index])[-1]
        return img, imgname


# main routine for the project
def train():
    # file paths
    train_path = os.path.join('..', 'real')  # real training set
    
    # hyperparameters
    bsize = 64  # batch size
    nworkers = multiprocessing.cpu_count() # number of workers
    shuffle = True
    epochs = 200
    sample_interval = 100  # save 25 images every 100 batches
    device = torch.device('cpu')
    # tuning parameter
    lambda_gp = 10
    n_critic = 5  # how many iterations to train discriminator before training generator
#    clip_value = 0.01  # weight clipping, used in WGAN, not in WGAN-GP
    # model parameter
    opt = {'in_feat': 1, 'out_feat': 3, 'img_size': 4, 'scale_factor': 2}
    real_feats = 3  # color channel count for real imgs
    num_classes = 1  # discriminator output size, only outputs a validity score
    # optimizer parameter
    betas = (0, 0.9)
    lr = 0.0001  # from WGAN-GP paper
    
    # create dataloaders
    train_ds = ImageDataset(train_path)
    real_loader = DataLoader(train_ds, batch_size=bsize, shuffle=shuffle, num_workers=nworkers)
    
    # create model
    gen = G.CNN(opt)
    dis = D.DPNmini(real_feats, num_classes)
    
    # optimizers
    optimizer_G = torch.optim.Adam(gen.parameters(), lr=lr, betas=betas)
    optimizer_D = torch.optim.Adam(dis.parameters(), lr=lr, betas=betas)
    
    batches_done = 0
    for epoch in range(epochs):
        logger.info('# ---- Epoch {}/{} ---- #'.format(epoch+1, epochs))
              
        for i, (imgs, _) in enumerate(real_loader):
            # Configure input
#            real_imgs = Variable(imgs.type(Tensor))
            real_imgs = imgs.to(device)
    
            # ---------------------
            #  Train Discriminator
            # ---------------------
    
            optimizer_D.zero_grad()
    
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt['in_feat'], opt['img_size'], opt['img_size']))))
    
            # Generate a batch of images
            fake_imgs = gen(z)
    
            # Real images
            real_validity = dis(real_imgs)
            # Fake images
            fake_validity = dis(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(dis, real_imgs.data, fake_imgs.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
    
            d_loss.backward()
            optimizer_D.step()
    
            # Train the gen every n_critic steps
            if i % n_critic == 0:
                optimizer_G.zero_grad()
                # -----------------
                #  Train Generator
                # -----------------
    
                # Generate a batch of images
                fake_imgs = gen(z)
                # Loss measures generator's ability to fool the dis
                # Train on fake images
                fake_validity = dis(fake_imgs)
                g_loss = -torch.mean(fake_validity)
    
                g_loss.backward()
                optimizer_G.step()
    
                logger.info(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch+1, epochs, i+1, len(real_loader), d_loss.item(), g_loss.item())
                )
                if batches_done % sample_interval == 0:
                    save_image(fake_imgs.data[:25], os.path.join('..', 'fakes', "{:d}.png".format(batches_done)),
                               nrow=5, normalize=True)
                    logger.info('generated sample images saved to disk')
                batches_done += n_critic
    return gen


# helper function for calculating GP in WGAN-GP
#Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
Tensor = torch.FloatTensor
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

logger = logging.getLogger(__name__)
#--- run main routine and prediction ---#
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.FileHandler('train.log', mode='w')
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    
    # create console handler and set level to debug
    sysout = logging.StreamHandler()
    sysout.setLevel(logging.DEBUG)
    # add ch to logger
    logger.addHandler(sysout)    
    
    generator = train()