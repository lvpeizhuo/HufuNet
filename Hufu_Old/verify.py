# -*- coding: utf-8 -*
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


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model',      default='VGG', help='VGG-16, ResNet-18, LeNet')
parser.add_argument('--checkpoint', default='VGG', type=str, help='Pretrained model to start from')#--checkpoint=lenet——low—�?
parser.add_argument('--GPU', default='0', type=str,help='GPU to use')
parser.add_argument('--save_every', default=1, type=int, help='How often to save checkpoints in number of prunes (e.g. 10 = every 10 prunes)')
parser.add_argument('--cutout', action='store_true')
parser.add_argument('--finetune_steps', default=100)
parser.add_argument('--lr',             default=0.001)
parser.add_argument('--weight_decay', default=0.0005, type=float)
args = parser.parse_args()

frozen_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]='0' #

global error_history
error_history = []

models = {'VGG':vgg(),
          'FLenet':flenet()}
model = models[args.model]
model, sd = load_model(model, args.checkpoint, False)


if torch.cuda.is_available():
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)

transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])
batch_size = 256

if args.model=='VGG':
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,download = True, transform=transform)
elif args.model == 'FLenet':
    test_dataset = torch.load('./data/FM_test')
testloader = DataLoader(test_dataset, batch_size=batch_size)

criterion = nn.CrossEntropyLoss()
prune_rate = 0
validate(model, prune_rate, testloader, criterion)

