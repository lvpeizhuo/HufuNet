# -*- coding:ISO-8859-1 -*-

####
# be attention: I've modified the VGG model's __prune__ method, now is simulating real pruning method

# need to modify VGG model, roll back to normal prune
####

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
parser.add_argument('--data_loc',   default='/disk/scratch/datasets/cifar', type=str)
parser.add_argument('--checkpoint', default='VGG_done_finetune100', type=str, help='Pretrained model to start from')
parser.add_argument('--prune_checkpoint', default='', type=str, help='Where to save pruned models')
parser.add_argument('--GPU', default='0,1', type=str,help='GPU to use')
parser.add_argument('--save_every', default=5, type=int, help='How often to save checkpoints in number of prunes (e.g. 10 = every 10 prunes)')
parser.add_argument('--cutout', action='store_true')
parser.add_argument('--prune_rate', default=10, type=int)
parser.add_argument('--lr',             default=0.001)
parser.add_argument('--weight_decay', default=0.0005, type=float)

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU

global error_history
error_history = []

models = {'VGG' : vgg(),
          'FLenet': flenet()}
print(args.model)

model = models[args.model]

old_format=False
if 'wrn' in args.model:
    old_format=True

model, sd = load_model(model, args.checkpoint, old_format)

if torch.cuda.is_available():
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)
transform = transforms.Compose([transforms.ToTensor()])
batch_size =256

test_dataset = torch.load('VGG_finetune_test2000') # **Notice**: Prune After Finetuning
    
testloader = DataLoader(test_dataset, batch_size=batch_size)

criterion = nn.CrossEntropyLoss()

model,max_param = sparsify(model, args.prune_rate, get_prune_max = True)
print('Prune rate: '+str(args.prune_rate)+'%, Delete maximum parameter:\t'+str(max_param))
 
validate(model, args.prune_rate, testloader, criterion, checkpoint=args.model+'_done_3*3_prune'+str(args.prune_rate)) #'VGG_EnlargeC_prune'+str(args.prune_rate)

















