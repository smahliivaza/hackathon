import argparse
import os
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import neuromation.transforms as transforms
from encoder import DataEncoder
from loss import FocalLoss
from retinanet import RetinaNet
from neuromation.datasets import BottleLoader


parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--exp', required=True, help='experiment name')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

sys.path.insert(0, os.path.join('exps', args.exp))
import config as cfg

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')
start_epoch = 0
lr = cfg.lr

print('Preparing data..')
train_transform_list = [transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)]
if cfg.scale is not None:
    train_transform_list.insert(0, transforms.Scale(cfg.scale))
train_transform = transforms.Compose(train_transform_list)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cfg.mean, cfg.std)
])

trainset = BottleLoader(cfg.train_path, encoder=DataEncoder(), transform=train_transform)
valset = BottleLoader(cfg.train_path, encoder=DataEncoder(), transform=val_transform)
trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True,
                                          num_workers=cfg.num_workers, collate_fn=trainset.collate_fn)
valloader = DataLoader(valset, batch_size=cfg.batch_size, shuffle=False,
                                        num_workers=cfg.num_workers, collate_fn=valset.collate_fn)

print('Building model...')
net = RetinaNet(backbone=cfg.backbone, num_classes=len(cfg.classes))
if args.resume:
    print('Resuming from checkpoint..')
    checkpoint = torch.load(os.path.join('ckpts', args.exp, 'ckpt.pth'))
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
    lr = checkpoint['lr']

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()
cudnn.benchmark = True

criterion = FocalLoss(len(cfg.classes))
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

def train(epoch):
    print('\nTrain Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        nn.utils.clip_grad_norm(net.parameters(), max_norm=1.2)
        optimizer.step()

        train_loss += loss.data[0]
        print('train_loss: %.3f | avg_loss: %.3f' % (loss.data[0], train_loss/(batch_idx+1)))
    save_checkpoint(train_loss, len(trainloader))

def val(epoch):
    net.eval()
    val_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(valloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        val_loss += loss.data[0]
        print('val_loss: %.3f | avg_loss: %.3f' % (loss.data[0], val_loss/(batch_idx+1)))
    save_checkpoint(val_loss, len(valloader))

def save_checkpoint(loss, n):
    global best_loss
    loss /= n
    if loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'loss': loss,
            'epoch': epoch,
            'lr': lr
        }
        ckpt_path = os.path.join('ckpts', args.exp)
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path)
        torch.save(state, os.path.join(ckpt_path, 'ckpt.pth'))
        best_loss = loss

for epoch in range(start_epoch + 1, start_epoch + cfg.num_epochs + 1):
    if epoch in cfg.lr_decay_epochs:
        lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    train(epoch)

    if cfg.eval_while_training and epoch % cfg.eval_every == 0:
        val(epoch)
