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
import hmac
import hashlib
from models import *
from utils  import *
from tqdm   import tqdm

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
# parser.add_argument('--model',      default='VGG', help='resnet9/18/34/50, wrn_40_2/_16_2/_40_1')
# parser.add_argument('--checkpoint', default='VGG_done_3*3', type=str)
parser.add_argument('--GPU', default='0,1', type=str,help='GPU to use')
parser.add_argument('--epochs',     default=100, type=int)
parser.add_argument('--lr',         default=0.001)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay')
parser.add_argument('--weight_decay', default=0.0005, type=float)
args = parser.parse_args()

torch.backends.cudnn.enabled = False
frozen_seed()

#####################################################################################################
# reset grad to initial state: all ones

tobe = torch.load('checkpoints/VGG_done_3*3.t7', map_location='cpu')

for name in [key for key in tobe['net'].keys() if 'grad' in key and 'weight' in key]:
    print(name)
    tobe['net'][name] = torch.ones(tobe['net'][name].size())
    
torch.save(tobe,'checkpoints/VGG_done_finetune_base.t7')

#####################################################################################################
# finetune

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU

global error_history

model = vgg()
model, sd = load_model(model, 'VGG_done_finetune_base', False)

if torch.cuda.is_available():
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)


transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])
batch_size = 256
train_dataset = torch.load('VGG_finetune_train8000')
test_dataset = torch.load('VGG_finetune_test2000')

trainloader = DataLoader(train_dataset, batch_size=batch_size)
testloader = DataLoader(test_dataset, batch_size=batch_size)

#optimizer = optim.SGD([w for name, w in model.named_parameters() if not 'mask' in name], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
optimizer = optim.Adam([w for name, w in model.named_parameters() if not 'mask' in name], lr=args.lr, betas=(0.9, 0.99))
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,args.epochs, eta_min=1e-10)
criterion = nn.CrossEntropyLoss()

error_history = []
name = 'VGG_done_finetune100'
for epoch in tqdm(range(args.epochs)):
    '''
    if epoch%10==0:
        name = 'VGG_done_3*3_finetune'+str(epoch+10)
    print()
    print(name)
    '''
    train_with_grad_control(model, trainloader, criterion, optimizer)
    validate(model, epoch, testloader, criterion, checkpoint=name)
    scheduler.step()