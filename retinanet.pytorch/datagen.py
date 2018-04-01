'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
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
from collections import defaultdict


class BottleLoader(Dataset):
    def __init__(self, dir, metadata=None, json_idx=1):
        self.dir = dir
        self.encoder = DataEncoder()
        if metadata is None:
            self._index()
        else:
            self.load_metadata(metadata)
        self.json_idx = json_idx
            
    def load_metadata(self, path):
        with open(path, 'rb') as f:
            self.metadata = pickle.load(f)
            
    def dump_metadata(self, path):
        with open(path, 'wb+') as f:
            pickle.dump(self.metadata, f)
        
    def _index(self):
        files = listdir(self.dir)
        prefixes = list(map(lambda f: f.replace('.jpg', ''), 
                            filter(lambda f: '.jpg' in f, files)))
        prefixes = list(map(lambda f: path.join(self.dir, f), prefixes))
        
        self.metadata = {}
        self.metadata['paths'] = list(map(lambda f: (
            f'{f}.jpg', f'{f}.json', f'{f}_cutDeeper.json', f'{f}_factor.json'
        ), prefixes))
        
        labelset = set()
        for p in self.metadata['paths']:
            with open(p[1], 'r') as f:
                j = json.load(f)
            labelset = labelset.union(set(map(lambda f: f['id'], j)))
        self.metadata['label_index'] = dict((k,v) for v,k in enumerate(labelset))
        
        self.metadata['shape'] = []
        for p in self.metadata['paths']:
            self.metadata['shape'].append(Image.open(p[0]).size)
    
    def __getitem__(self, i):
        sizeremap = {
            1024: 500,
            1366: 500
        }
        data = list(self.metadata['paths'][i])
        shape = self.metadata['shape'][i]
        scale = (sizeremap[shape[0]]/shape[0], sizeremap[shape[1]]/shape[1])
        img = Image.open(data[0])
        img.thumbnail((sizeremap[shape[0]], sizeremap[shape[1]]))
        img = torch.Tensor(np.array(img).transpose(2,0,1))
        with open(data[self.json_idx], 'r') as f:
            groups = json.load(f)
        num_obj = sum(map(len, groups))
        coords, labels = [], []
        for group in groups:
            for obj in group['data']:
                coords.append(np.array([
                    obj['boundingBox']['X']*scale[0], obj['boundingBox']['Y']*scale[1],
                    (obj['boundingBox']['X'] + obj['boundingBox']['Width'])*scale[0],
                    (obj['boundingBox']['Y'] + obj['boundingBox']['Height'])*scale[1]
                ]))
                labels.append(group['id'])
        coords = torch.Tensor(np.stack(coords))
        labels = torch.LongTensor(np.array(
            list(map(self.metadata['label_index'].get, labels)))
        ).view(-1,1)
        return img, coords, labels
    
    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h,w = imgs[0].shape[-2:]
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets).squeeze()
    
    def __len__(self):
        return len(self.metadata['paths'])
    
class BucketBatchSampler:
    def __init__(self, dataset, batch_size, drop_last):
        self.sampler = RandomSampler(dataset)
        self.shapes = dataset.metadata['shape']
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.leftovers = defaultdict(list)
        
    def infer_orientation(self, shape):
        if shape[0] > shape[1]:
            return 'album'
        if shape[0] < shape[1]:
            return 'portrait'
        return 'square'

    def __iter__(self):
        batch, batch_orientation = [], None
        for idx in self.sampler:
            im_orientation = self.infer_orientation(self.shapes[idx])
            
            if batch_orientation is None:
                batch_orientation = im_orientation
                batch.append(idx)
            elif len(self.leftovers[batch_orientation])>0:
                batch.append(self.leftovers[batch_orientation].pop(0))
            elif im_orientation == batch_orientation:
                batch.append(idx)
            else:
                self.leftovers[im_orientation].append(idx)
                continue
                
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

