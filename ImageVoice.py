import numpy as np
import os
import random

from torch.utils.data import TensorDataset
from PIL import Image
import torchvision
import torch


class ImageVoice(TensorDataset):
  def __init__(self):
    data_path = os.environ['data_path']
    topdir = data_path + "/cropFaces"
    self.pics = []
    
    name_count = 0
    for name in os.listdir(topdir):
      name = topdir + '/' + name
      if not os.path.isdir(name):
        continue
      name_count += 1
      picsdir = str(name) + '/1.6/'
      voices = data_path + './mbk_vad/id' + str(10000 + name_count) + '/'


      for picdir in os.listdir(picsdir):
        voicedir = voices + picdir
        picdir = picsdir + picdir
        if not os.path.isdir(picdir):
          continue
        if not os.path.isdir(voicedir):
          continue

        self.pics.append((picdir, voicedir))

    self.avg_len = 1688

  def __len__(self):
    return len(self.pics)

  def __getitem__(self, idx):
    picdir, voicedir = self.pics[idx]
    pic = picdir + '/' + random.choice(os.listdir(picdir))
    
    voice = np.array([])
    voice_tmp = np.array([])

    for voicefile in os.listdir(voicedir):
      voice_tmp = np.load(voicedir + '/' + voicefile)
      if voice_tmp.shape[0] > voice.shape[0]:
        voice = voice_tmp

    img = Image.open(pic)
    img = torchvision.transforms.ToTensor()(img)

    if voice.shape[0] != self.avg_len:
      if voice.shape[0] > self.avg_len:
        cutoff = (voice.shape[0] - self.avg_len) // 2
        voice = voice[cutoff:cutoff + self.avg_len]
      else:
        while voice.shape[0] < self.avg_len:
          pad_left = (self.avg_len - voice.shape[0]) // 2
          pad_right = self.avg_len - pad_left - voice.shape[0]
          left = voice[pad_left : 0 : -1]
          right = voice[-2 : (voice.shape[0] - pad_right) - 2 : -1]
          voice = np.concatenate((left, voice, right), axis=0)


    return img, torch.from_numpy(voice)


