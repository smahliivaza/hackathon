import os
import sys
import random

import torchvision.transforms as transforms

from PIL import Image

import numpy as np
import torch, json, pickle

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler
from os import path, listdir

from encoder import DataEncoder
from .bbox import BoundingBox


class BottleLoader(Dataset):
    def __init__(self, dir, encoder, json_suffix='', transform=None, val=False):
        self.dir = dir
        self.encoder = DataEncoder()
        self.json_suffix = json_suffix
        self.transform = transform
        self.encoder = encoder
        
        files = listdir(self.dir)
        prefixes = list(map(lambda f: f.replace('.jpg', ''), 
                            filter(lambda f: '.jpg' in f, files)))
        prefixes = list(map(lambda f: path.join(self.dir, f), prefixes))
        
        self.impath = list(map(lambda f: f'{f}.jpg', prefixes))
        self.annotations = list(map(lambda f: f'{f}{self.json_suffix}.json', prefixes))
      
        labelset = set()
        for p in self.annotations:
            with open(p, 'r') as f:
                j = json.load(f)
            labelset = labelset.union(set(map(lambda f: f['id'], j)))
        self.label_index = dict((k,v) for v,k in enumerate(labelset))
        
        self.val = val
        
    def annotate(self, fname, imsize):
        boxes = []
        with open(fname, 'r') as f:
            groups = json.load(f)
        coords, labels = [], []
        for group in groups:
            for obj in group['data']:
                boxes.append(BoundingBox(
                    obj['boundingBox']['X'], 
                    obj['boundingBox']['Y'] + obj['boundingBox']['Height'], 
                    obj['boundingBox']['X'] + obj['boundingBox']['Width'], 
                    obj['boundingBox']['Y'], imsize[0], imsize[1], 
                    self.label_index[group['id']]))
        return boxes
    
    def __getitem__(self, i):
        data = list(self.metadata['paths'][i])
        shape = self.metadata['shape'][i]
        img = np.array(Image.open(data[0]))
        img = resize(img, (sizeremap[shape[0]], sizeremap[shape[1]]))
        img = torch.Tensor(img.transpose(2,0,1))

        coords = torch.Tensor(np.stack(coords))
        labels = torch.LongTensor(np.array(
            list(map(self.metadata['label_index'].get, labels)))
        ).view(-1,1)
        return img, coords, labels
    
    def __getitem__(self, index):
        impath = self.impath[index]
        annotation = self.annotations[index]
        image = Image.open(impath)
        boxes = self.annotate(annotation, image.size)
        example = {'image': image, 'boxes': boxes[:5]}
        if self.transform:
            example = self.transform(example)
        return example

    def __len__(self):
        return len(self.impath)

    def collate_fn(self, batch):
        imgs = [example['image'] for example in batch]
        boxes  = [example['boxes'] for example in batch]
        labels = [example['labels'] for example in batch]
        img_sizes = [img.size()[1:] for img in imgs]

        max_h = max([im.size(1) for im in imgs])
        max_w = max([im.size(2) for im in imgs])
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, max_h, max_w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            im = imgs[i]
            imh, imw = im.size(1), im.size(2)
            inputs[i,:,:imh,:imw] = im

            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(max_w, max_h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        if not self.val:
            return inputs, torch.stack(loc_targets), torch.stack(cls_targets)
        return inputs, img_sizes, torch.stack(loc_targets), torch.stack(cls_targets)