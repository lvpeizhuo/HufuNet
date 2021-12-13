# -*- coding: utf-8 -*
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

from models.autoencoder import *
from utils  import *
from tqdm   import tqdm

frozen_seed()

epochs=100
checkpoint="Encoder"
lr=0.005

model = encoder()

torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--data_loc',   default='/disk/scratch/datasets/MNIST', type=str)
parser.add_argument('--checkpoint', default='VGG', type=str)
parser.add_argument('--GPU', default='0,1', type=str,help='GPU to use')
parser.add_argument('--epochs',     default=100, type=int)
parser.add_argument('--lr',         default=0.005)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay')
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--checkpoint_per_epoch', default = 10, type=int)
parser.add_argument('--validate_mode',default=0,type=bool)
args = parser.parse_args()

print(os.environ["CUDA_VISIBLE_DEVICES"])
device = torch.device("cuda:0")
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU

if torch.cuda.is_available():
    model = model.cuda()
model.to(device)

global error_history

transform = transforms.Compose([transforms.ToTensor()])
batch_size = 500

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False,download = True, transform=transform)
trainloader = DataLoader(train_dataset, batch_size=batch_size)
testloader = DataLoader(test_dataset, batch_size=batch_size)

total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

optimizer = optim.Adam(model.parameters(),lr=args.lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,epochs, eta_min=1e-10)
criterion = nn.MSELoss()

visualize_num=8

error_history = []
f, a = plt.subplots(2, visualize_num, figsize=(visualize_num, 2)) 

loss=nn.MSELoss()
X=torch.Tensor([1,2])
Y=torch.Tensor([[3,4],[6,2]])

grad_recorder = model
model.eval()

for epoch in range(args.epochs):
    for step, data in enumerate(trainloader):
        img, _ = data
        encoded,decoded = model(img)
        loss = criterion(decoded.to(device), img.to(device))
        loss.backward()
        optimizer.step()
        grad_1sum = 0
        grad_sum = 0
        grad_count = 0
        for i in range(6):
            a = torch.abs(list(model.parameters())[i].grad)
            b = torch.where(torch.isinf(a), torch.full_like(a, 0), a)
            grad_sum += torch.sum(b).item()
            grad_count += torch.abs(list(model.parameters())[i].grad).numel()
        print(grad_sum,grad_count,grad_sum / grad_count)
        grad_oppo = grad_sum / grad_count

        optimizer.zero_grad()
        if step % 5 == 0:
            x,y = eval(img,decoded,0)
            if (step % args.checkpoint_per_epoch==0):
                torch.save(model.encoder.state_dict(), 'weight\model_encoder_param.t7')
                torch.save(model.decoder.state_dict(), 'weight\model_decoder_param.t7')
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.to(device='cpu').numpy())