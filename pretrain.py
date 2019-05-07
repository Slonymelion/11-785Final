import torch
import torch.autograd as autograd
import torch.multiprocessing as mp
#import torch.nn as nn
#import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from torch.nn.modules.distance import CosineSimilarity
#import torchvision
from torchvision.utils import save_image
#from PIL import Image

import logging
import os
#import sys
import numpy as np
import time

import discriminator as D
import generator as G

from ImageVoice import ImageVoice, create_dataset

# main routine for the project
def train(gen, dis, dataloader, opt):    
    # hyperparameters
    warm_id = opt['warm_model_id']
    epochs = opt['epochs']
    annealing_schedule = opt['annealing_schedule']
    latent_dim = opt['latent_dim'] if 'latent_dim' in opt else 100
    use_latent = opt['use_latent']
    lr = opt['learning_rate']
    
#    device = torch.device('cpu')
    device = torch.device('cuda')
    print('using {}'.format(device))
    # tuning parameter
#    clip_value = 0.01  # weight clipping, used in WGAN, not in WGAN-GP
    # optimizer parameter
    betas = (0.5, 0.999)  # from WGAN-GP paper
    
    # optimizers & losses
    optimizer_G = torch.optim.Adam(torch.nn.ModuleList([gen, dis]).parameters(), lr=lr, betas=betas)
#    optimizer_G = torch.optim.SGD(gen.parameters(), lr=lr, momentum=0.8)

#    l2loss = torch.nn.MSELoss()
#    l1loss = torch.nn.L1Loss()
    cosloss = torch.nn.CosineEmbeddingLoss()
    
    # warm start from disk
    if warm_id:
        gen_save_file = 'generator_{:d}'.format(warm_id)
        gen_save_file = os.path.join('pretrain', str(opt['warm_run_id']), gen_save_file)
        gen = warm_start_model(gen, gen_save_file)
        logger.info('generator warm started from experiment folder {}, model id {}'.format(opt['warm_run_id'], warm_id))
        
    dis.to(device)
    gen.to(device)
#    l1loss.to(device)
    cosloss.to(device)
    
    dis.train()
    gen.train()
    batch_interval = len(dataloader) // 100
    best_loss = float('inf')
    patience = 0
    patience_threshold = 3
    anneal_factor = 5
    for epoch in range(epochs):
        logger.info('# ---- Epoch {}/{} ---- #'.format(epoch+1, epochs))
        
        if (epoch+1) in annealing_schedule:
            optimizer_G.param_groups[0]['lr'] /= 10
            logger.info(' learning rate decreased by factor of 10 ')
            
        for (i, (imgs, _)) in enumerate(dataloader):
            if (i+1) % 20 == 0:
                print('{} batches done'.format(i+1))
            # Configure input
#            real_imgs = Variable(imgs.type(Tensor))
            real_imgs = imgs.to(device)
            
            # Sample noise as generator input
            if use_latent:
                z = Variable(Tensor(np.random.normal(0, 1, size=(imgs.shape[0], latent_dim, 1, 1))))
                z = z.to(device)
            else:
                z = None
            
            # Real features
            with torch.no_grad():
                real_features = dis(real_imgs)
            # Generate a batch of images
            fake_imgs = gen(z, x=real_features)
            # Generate fake features
            fake_features = dis(fake_imgs)
            
            # Target, used to calculate cosine loss
            target = torch.ones(imgs.size(0)).to(device)
            loss = cosloss(real_features, fake_features, target)
#            loss = l1loss(real_features, fake_features)
            
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()
            
            if (i+1) % batch_interval == 0:
                current_loss = loss.item()
                logger.info(
                    "[Batch %d/%d] L1 Loss: %.6e"
                    % (i+1, len(dataloader), current_loss)
                )
                if current_loss < best_loss:
                    logger.info('loss decreases, best loss is {:.6e}'.format(current_loss))
                    best_loss = current_loss
                else:
                    patience += 1
                    if patience > patience_threshold:
                        optimizer_G.param_groups[0]['lr'] /= anneal_factor
                        patience = 0
                        logger.info('loss staggered, decrease learning rate by factor of {}, now the learning rate is {}'.format(anneal_factor,
                                    optimizer_G.param_groups[0]['lr']))
#                print(fake_features[1])
#                print(real_features[1])
            if (i+1) % 500 == 0:
                logger.info(
                    "[G first grad: %e] [G final grad: %e] [G respath first grad: %e] "
                    % (
                        torch.mean(torch.abs(gen.layers[0][0].weight.grad)).item(), 
                        torch.mean(torch.abs(gen.layers[-1][0].weight.grad)).item(),
                        torch.mean(torch.abs(gen.shortcut[1][0].weight.grad)).item(),
                        )
                )
                save_image(real_imgs.data[:16], os.path.join('pretrain', opt['run_id'], 'reals', "{:d}.png".format(i+1)),
                           nrow=4, normalize=True)
                save_image(fake_imgs.data[:16], os.path.join('pretrain', opt['run_id'], 'fakes', "{:d}.png".format(i+1)),
                           nrow=4, normalize=True)
                logger.info('generated sample images saved to disk as {:d}.png'.format(i+1))
            if (i+1) % 1000 == 0:
                gen_save_file = 'generator_{:d}'.format(i+1)
                torch.save(gen.state_dict(), os.path.join('pretrain', opt['run_id'], gen_save_file))
                logger.info('generator model saved at {:d}'.format(i+1))
#    return gen
                

# helper function for warm starting model
def warm_start_model(model, model_save_file):
    """
    INPUT:
        model           -   model object
        model_savefile  -   string for saved model state_dict
    OUTPUT:
        model           -   warm started model
    """
    new_state = model.state_dict()
    warm_state = torch.load(model_save_file)
    for (k, v) in warm_state.items():
        if k in new_state:
            shape_old = warm_state[k].size()
            shape_new = new_state[k].size()
            if shape_new == shape_old:
                new_state.update({k: v})
    model.load_state_dict(new_state)
    return model


# helper function for calculating GP in WGAN-GP
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
#Tensor = torch.FloatTensor
def compute_gradient_penalty(D, sounds, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates, sounds)
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
    # generate folders to save model
    run_id = str(int(time.time()))
    if not os.path.exists('./pretrain'):
        os.mkdir('./pretrain')
    os.mkdir('./pretrain/%s' % run_id)
    os.mkdir('./pretrain/%s/reals' % run_id)
    os.mkdir('./pretrain/%s/fakes' % run_id)
    print("Saving models and logs to ./pretrain/%s" % run_id)
    
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    logfile = os.path.join('pretrain', run_id, 'train.log')
    ch = logging.FileHandler(logfile, mode='w')
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
    
    # generate data loader
    bsize = 128  # batch size
    nworkers = mp.cpu_count() # number of workers
    shuffle = True
    imagefolder = 'cropFaces_original'  # real training set
    train_ds = create_dataset(imagefolder)
    length = len(train_ds)
#    train_idx = torch.randperm(length)[:(length // 10)]
#    train_ds = [train_ds[i] for i in train_idx]
    print(len(train_ds), bsize, nworkers)
    real_loader = DataLoader(train_ds, batch_size=bsize, shuffle=shuffle, num_workers=nworkers)
    
    # generate model
    opt = {
           'training_img_ratio':0.1,
           'in_feat': 1, 
           'out_feat': 3, 
           'use_latent':False,
           'latent_dim': 100,  # must be 0 if use_latent = False
           'soundnet_out_dim': 256,
           'epochs': 10,
           'annealing_schedule':[], 
           'learning_rate': 1e-3,
           'run_id': run_id, 
           'warm_run_id': 1557186734, 
           'warm_model_id':None,
           'general_note': 'deepdeep generator, pretrained vgg16_bn'
           }
    logger.info(opt)
    real_feats = 3  # color channel count for real imgs
    num_classes = 1  # discriminator output size, only outputs a validity score
    dis = D.PreVGG(features=opt['soundnet_out_dim'])
    gen = G.CResGenDeepDeep(opt, SoundNet=None)
    
    # train model
    generator = train(gen, dis, real_loader, opt)