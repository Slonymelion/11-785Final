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

import discriminator as D


# image dataset class exclusively for test
class ImageDatasetTest(Dataset):
    """
    return (image tensor, image file name) tuple
    """
    def __init__(self, path, testfile):
        with open(testfile, "r", encoding='utf-8') as f:
            file_list = f.readlines()
            file_list = [os.path.join(path, x.strip()) for x in file_list]
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        imgname = os.path.split(self.file_list[index])[-1]
        return img, imgname


# image dataset for verification task
class ImageDataVeri(Dataset):
    """
    return ((imageTensor1, imageTensor2), image pair names) tuple
    """
    def __init__(self, path, testfile):
        with open(testfile, "r", encoding='utf-8') as f:
            lines = [x.strip() for x in f.readlines()]
            self.imgnames = lines
            lines = [x.split(' ') for x in lines]
            file_list = [(os.path.join(path, x[0]), os.path.join(path, x[1])) for x in lines]
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img1 = Image.open(self.file_list[index][0])
        img2 = Image.open(self.file_list[index][1])
        img1 = torchvision.transforms.ToTensor()(img1)
        img2 = torchvision.transforms.ToTensor()(img2)
        imgname = self.imgnames[index]
        return (img1, img2), imgname


# main routine for the project
def train_classification(model_name_base, device=torch.device('cpu'),
                         warm_start=False, old_model=None,
                         skip_train=False):
    # file paths
    train_path = 'hw2p2_check/train_data/medium/'  # real training set
    val_path = 'hw2p2_check/validation_classification/medium/'  # real validation set
    
    # hyperparameters
    bsize = 64  # batch size
    nworkers = 8 # number of workers
    shuffle = True
    epochs = 30
    # network structure
    available_types = ['dpn26', 'dpn26small', 'dpn50', 'dpn92', 'dpn50small']
    networktype = model_name_base.split('_')[0]
    task = 'classify'
    # for learning rate annealing
    early_stop_threshold = 6
    early_stop_count = 0
    # learning rate decay, fixed interval, drop by half every 3 epochs
    decay_interval = 3  
    decay_factor = 0.2
    # for label training
    learningRate = 0.0005
    momentum = 0
    weightDecay = 1e-4
    alpha = 0.99
    betas = (0.9, 0.999)
    eps = 1e-8
    lr_sgd = 1e-1
    wd_sgd = 1e-4
    m_sgd = 0.9
    
    # create dataloaders
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
    ])
    train_ds = torchvision.datasets.ImageFolder(root=train_path,
                                                transform=data_transform)
    train_loader = DataLoader(train_ds, batch_size=bsize, shuffle=shuffle, num_workers=nworkers)
    
    dev_ds = torchvision.datasets.ImageFolder(root=val_path,
                                              transform=torchvision.transforms.ToTensor())
    dev_loader = DataLoader(dev_ds, batch_size=bsize, shuffle=False, num_workers=nworkers)
    
    # create model
    num_feats = 3  # color channels
    num_classes = len(train_ds.classes)  # training set classes 2300
    if networktype == 'dpn26':
        network = D.DPN26(num_feats, num_classes, task=task)
    elif networktype == 'dpn26small':
        network = D.DPN26small(num_feats, num_classes, task=task)
    elif networktype == 'dpn50':
        network = D.DPN50(num_feats, num_classes, task=task)
    elif networktype == 'dpn92':
        network = D.DPN92(num_feats, num_classes, task=task)
    elif networktype == 'dpn50small':
        network = D.DPN50small(num_feats, num_classes, task=task)
    else:
        raise AttributeError('unknown network type, should be {}'.format(available_types))
    # warm_start?    
    if warm_start:
        try:
            network.load_state_dict(torch.load(old_model+'.ckpt'))
            print('warm start from model {}'.format(old_model+'.ckpt'))
        except Exception as e:
            print(e)
            print('warm start failed, retrain')
            network.apply(init_weights)
    else:
        network.apply(init_weights)

    if not skip_train:
        # log settings
        logfile = open(model_name_base+'.log', "w", encoding='utf-8')
        # write parameters to log file
        print('start training, log saved to {}'.format(model_name_base+'.log'))
        paramslog = 'network type = {}\n'.format(networktype)
        paramslog = 'batch size = {}\n'.format(bsize)
        paramslog += 'epochs = {}\n'.format(epochs)
        paramslog += 'lr_label = {}\n'.format(learningRate)
        paramslog += 'momentum_label = {}\n'.format(momentum)
        paramslog += 'weightdecay_label = {}\n'.format(weightDecay)
        paramslog += 'alpha_label = {}\n'.format(alpha)
        paramslog += 'eps_label = {}\n'.format(eps)
        print(paramslog, file=logfile)
        # loss and optimizer
        criterion_label = nn.CrossEntropyLoss()  
        optimizer_adam = torch.optim.Adam(network.parameters(), lr=learningRate, betas=betas,
                                          eps=eps, weight_decay=weightDecay, amsgrad=False)
        optimizer_sgd = torch.optim.SGD(network.parameters(), lr=lr_sgd,
                                        weight_decay=wd_sgd, momentum=m_sgd)
        
        # training
        network.train()
        network.to(device)   
        old_val_acc = -1  # to monitor early stop and model saving
        for epoch in range(epochs):
            avg_loss = 0.0
            for batch_num, (feats, labels) in enumerate(train_loader):
                feats, labels = feats.to(device), labels.to(device)
                
                label_output = network(feats)           
                loss = criterion_label(label_output, labels.long())  # cross entropy loss

                optimizer = optimizer_sgd if epoch < epochs//2 else optimizer_adam
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                avg_loss += loss.item()               
                if batch_num % 400 == 399:
                    print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/400))
                    avg_loss = 0.0    
                
    #            torch.cuda.empty_cache()
    #            del feats
    #            del labels
    #            del loss
          
            # validation
            val_loss, val_acc = test_classify(network, dev_loader, criterion_label, device=device)

            print('Epoch {} --- Val Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(epoch+1, val_loss, val_acc), file=logfile)
            print('Epoch {} --- Val Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(epoch+1, val_loss, val_acc))
            
            # save the best model
            if val_acc > old_val_acc:
                model_name = model_name_base+'_best.ckpt'
                print('validation accuracy decreased, saving model to {}'.format(model_name), file=logfile)
                torch.save(network.state_dict(), model_name)
                old_val_acc = val_acc
            else:
                early_stop_count += 1
                print('early stopping countdown...{}'.format(early_stop_threshold-early_stop_count))
                if early_stop_count >= early_stop_threshold:
                    print('validation accuracy increase too many times. stop training')
                    print('model training stopped early', file=logfile)
                    break

            # learning rate decay
            if (epoch + 1) % decay_interval == 0 and (epoch+1) != epochs:
                optimizer.param_groups[0]['lr'] *= decay_factor  # this is the correct way of updating learning rate!!!
                print('new learning rate = {:.4e}'.format(optimizer.param_groups[0]['lr']))
            
        # save final model
        model_name = model_name_base+'.ckpt'
        print('saving final model to {}'.format(model_name), file=logfile)
        torch.save(network.state_dict(), model_name)
    
    # close customary log file
    if logfile != sys.stdout:
        logfile.close()
        
    # return the network for further testing, second variable returns the 
    # label (folder names) to index mapping
    return network, {train_ds.class_to_idx[x]:x for x in train_ds.class_to_idx} 


# test verification prediction
def test_verification(network, model_name, img_path, test_file,
                      outfile='ver_submission.csv', warm_start=True,
                      device=torch.device('cpu')):
    """
    generate the similarity scores for verification test dataset
    """
    if warm_start:
        state = torch.load(model_name+'.ckpt')
        network.load_state_dict(state)
    network.eval()
    network.toggle(mode='verify')  # ensure network outputs features before linear layer

    test_ds = ImageDataVeri(img_path, test_file)
    test_loader = DataLoader(test_ds, batch_size=64, num_workers=6, shuffle=False)
    totaltest = len(test_ds)
    donetest = 0
    
    dist = CosineSimilarity(dim=1, eps=1e-10)
    
    imgs = []
    similarity = []  # to store cosine similarities
    
    print('start verifying test file...')
    network.to(device)
    for (firstimg, secondimg), imgnames in test_loader:
        imgs.extend(imgnames)
        firstimg, secondimg = firstimg.to(device), secondimg.to(device)
        firstoutput = network(firstimg)
        secondoutput = network(secondimg)
        similarity.extend(dist(firstoutput, secondoutput).tolist())
        donetest += len(imgnames)
        if donetest * 10/totaltest % 1 <= 0.01:
            print('{:.2f}% images verification done.'.format(donetest*100/totaltest))
        
    print('verification done, start writing csv file')
    with open(outfile, "w", encoding='utf-8') as out:
        out.write('trial,score\n')
        for (im, sim) in zip(imgs, similarity):
            out.write(im+','+str(sim)+'\n')
    
    network.toggle(mode='classify')  # don't forget to toggle the mode back
    network.train()


#--- Helper functions ---#
# validation loss and accuracy calculation
def test_classify(model, test_loader, criterion, device=torch.device('cpu')):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0.0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)
        
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        
        loss = criterion(outputs, labels.long())
        
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy/total


#--- run main routine and prediction ---#
if __name__ == '__main__':
    # remove all '._' files first
#    findNremove('hw2p2_check', '._', maxdepth=5)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    
    skip_train = False  # whether to skip_training, just get the pre-trained model
    skip_classify = False
    skip_verify = False

    model_name_base = 'dpn50small'  # version control
    old_model = model_name_base+'_best'
    network, train_ds = train_classification(model_name_base, device=device,
                                             warm_start=True, old_model=old_model,
                                             skip_train=skip_train)

    # classification task
    if not skip_classify:
        outfile = model_name_base+'_cls_submission.csv'
        img_path = os.path.join('hw2p2_check','test_classification','medium')
        test_file = os.path.join('hw2p2_check','test_order_classification.txt')
        test_classification(network, old_model, img_path,
                            test_file, train_ds,
                            outfile=outfile, device=device, warm_start=True)
    
    # verification task
    if not skip_verify:
        outfile = model_name_base+'_ver_submission.csv'
        img_ver_path = os.path.join('hw2p2_check','test_veri_T_new')
        ver_file = os.path.join('hw2p2_check','trials_test_new.txt')
        test_verification(network, old_model,
                          img_ver_path, ver_file,
                          outfile=outfile, device=device, warm_start=True)