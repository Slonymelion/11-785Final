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


# test classification prediction
def test_classification(network, model_name, img_path, test_file, mapping,
                        outfile='cls_submission.csv', warm_start=False,
                        device=torch.device('cpu')):
    """
    generate the classification prediction file for test dataset
    """
    network.eval()
    network.toggle(mode='classify')
    test_ds = ImageDatasetTest(img_path, test_file)
    test_loader = DataLoader(test_ds, batch_size=64, num_workers=6, shuffle=False)
    
    if warm_start:
        state = torch.load(model_name+'.ckpt')
        network.load_state_dict(state)
    
    imgs = []
    preds = []
    
    print('start predicting test file...')
    network.to(device)
    for feats, featnames in test_loader:
        imgs.extend(featnames)
        feats = feats.to(device)
        output = network(feats)
        _, pred_labels = torch.max(F.softmax(output, dim=1), 1)
        pred_labels = pred_labels.view(-1).long().tolist()
        preds.extend(pred_labels)
    print('prediction done, start writing csv file')
    with open(outfile, "w", encoding='utf-8') as out:
        out.write('id,label\n')
        imgs = [x.split('.')[0] for x in imgs]  # remove '.jpg'
        labels = [mapping[x] for x in preds]
        for (im, label) in zip(imgs, labels):
            out.write(im+','+label+'\n')
    
    network.train()