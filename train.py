# -*- coding: utf-8 -*
'''Train base models'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import os
import json
import argparse

from models import *
from utils  import *
from tqdm   import tqdm

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--model',      default='FLenet', help='resnet9/18/34/50, wrn_40_2/_16_2/_40_1')
parser.add_argument('--checkpoint', default='FLenet', type=str)
parser.add_argument('--GPU', default='0,1', type=str,help='GPU to use')
parser.add_argument('--epochs',     default=100, type=int)
parser.add_argument('--lr',         default=0.001)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay')
parser.add_argument('--weight_decay', default=0.0005, type=float)
args = parser.parse_args()

frozen_seed()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU

global error_history

models = {'Resnet9'  : ResNet9(),
          'Resnet18' : ResNet18(),
          'Resnet34' : ResNet34(),
          'Resnet50' : ResNet50(),
          'HufuNet': flenet(),
          'Googlenet': googlenet()}
print(args.model)
model = models[args.model]

if torch.cuda.is_available():
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)

transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])
batch_size = 256

if args.model=='Resnet18' or args.model=='Googlenet':
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,download = True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,download = True, transform=transform)
elif args.model == 'HufuNet':
    train_dataset = torch.load('data/FM_train')
    test_dataset = torch.load('data/FM_test')
trainloader = DataLoader(train_dataset, batch_size=batch_size)
testloader = DataLoader(test_dataset, batch_size=batch_size)
    

#optimizer = optim.SGD([w for name, w in model.named_parameters() if not 'mask' in name], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
optimizer = optim.Adam([w for name, w in model.named_parameters() if not 'mask' in name], lr=args.lr, betas=(0.9, 0.99))
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,args.epochs, eta_min=1e-10)
criterion = nn.CrossEntropyLoss()

error_history = []
for epoch in tqdm(range(args.epochs)):
    train(model, trainloader, criterion, optimizer)
    validate(model, epoch, testloader, criterion, checkpoint=args.checkpoint+'_init')
    scheduler.step()























