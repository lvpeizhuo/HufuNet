# -*- coding: utf-8 -*
'''Train base models to later be pruned'''
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
from validate import Evaluation
from extract import extract

torch.backends.cudnn.enabled = False

frozen_seed()

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--model',      default='VGG', help='resnet9/18/34/50, wrn_40_2/_16_2/_40_1')
parser.add_argument('--data_loc',   default='/disk/scratch/datasets/MNIST', type=str)
parser.add_argument('--load_from', default='VGG', type=str)
parser.add_argument('--GPU', default='0,1', type=str,help='GPU to use')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU

global error_history

models = {'VGG' : vgg(),
          'FLenet': flenet(),
          'ResNet18': ResNet18(),
          'ResNet34':ResNet34(),
          'googlenet':googlenet()}
model = models[args.model]

k=torch.load('./checkpoints/'+args.load_from+'.t7')
model.load_state_dict(k['net'])

if torch.cuda.is_available():
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)

batch_size = 32
        
criterion = nn.CrossEntropyLoss().cuda()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_dataset = torchvision.datasets.CIFAR10(root="/home/lpz/MHL/NewHufu/data/", train=False,download = True, transform=transform_test)

testloader = DataLoader(test_dataset, batch_size=batch_size)

validate_ori(model, -1, testloader, criterion)

Evaluation(extract(model.state_dict(),''),Visualize = 0)
