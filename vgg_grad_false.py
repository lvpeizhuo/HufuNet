# -*- coding: ISO8859-1 -*
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
parser.add_argument('--GPU', default='0,1', type=str,help='GPU to use')
parser.add_argument('--epochs',     default=100, type=int)
parser.add_argument('--lr',         default=0.001)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay')
parser.add_argument('--weight_decay', default=0.0005, type=float)
args = parser.parse_args()
###############################################################################################################
# embed and modify grad sign

frozen_seed()

seed = "2020"

hufu = torch.load('checkpoints/FLenet.t7', map_location='cpu')
tobe = torch.load('checkpoints/VGG_init.t7', map_location='cpu')


# print([key for key in hufu['net'].keys() if 'conv' in key and 'weight' in key])
# [u'conv1.weight', u'conv1.bias', u'mask1.weight', u'mask1.bias']
# print([key for key in tobe['net'].keys() if 'conv' in key and 'weight' in key])
# [u'conv1_1.weight', u'conv1_1.bias', u'mask1_1.weight', u'mask1_1.bias']

print('Start Embedding Flenet into VGG...')

tmp2 = torch.tensor([])
for name2 in [key for key in tobe['net'].keys() if 'conv' in key and 'weight' in key]:
    tmp2 = torch.cat([tmp2, tobe['net'][name2].view(-1)], 0)
# view(-1) function£ºflatten the matrix 1st dimention first, 2nd second, 3rd third
grad2 = torch.ones(tmp2.size())
len2 = tmp2.size()[0]

block_id = 1 # record current embed block id, and involve block_id into hash message       
for i,name1 in enumerate([key for key in hufu['net'].keys() if 'conv' in key and 'weight' in key]): # 1. get hufu model's layer name
    print(name1)
    
    (in1,out1,h1,w1) = hufu['net'][name1].size()
    #in2,out2,h2,w2 = tobe['net'][name2].size()
    
    for j in range(in1): 
        for k in range(out1): # embed block by block
            message = 0.0
            '''
            # original: embed one by one
            for m in range(h1): # use all number info in block as message
                for n in range(w1):
                    message += hufu['net'][name1][j,k,m,n]  # Xor operation
            # new: embed 3*3 block by block
            '''
            message += torch.sum(hufu['net'][name1][j,k,:,:])  # Xor operation
            
            message *= block_id # import block_id info into message
            block_id += 1 # update block id
            
            mac = hmac.new(bytes(seed, encoding='utf-8'),message.numpy(),hashlib.sha256)# hash
            pos = int(mac.hexdigest(), 16) % len2 # compute embed position
            
            '''
            # original: embed one by one
            for m in range(h1): # cope with collision and embed
                for n in range(w1):
                    while grad2[pos]==0:
                        pos = (pos+1)%len2
                    tmp2[pos] = hufu['net'][name1][j,k,m,n]
                    grad2[pos] = 0
            # new: embed 3*3 block by block
            '''
            pos = pos//9*9
            while grad2[pos]==0:
                pos = (pos+9)%len2
            tmp2[pos:pos+9] = hufu['net'][name1][j,k,:,:].view(-1)
            grad2[pos:pos+9] = 0
            
start_size = 0
for name2 in [key for key in tobe['net'].keys() if 'conv' in key and 'weight' in key]:
    end_size = start_size + len(tobe['net'][name2].view(-1))
    tobe['net'][name2] = tmp2[start_size:end_size].reshape(tobe['net'][name2].size())
    tobe['net'][name2.replace('conv','grad')] = grad2[start_size:end_size].reshape(tobe['net'][name2].size())
    start_size = end_size
    
torch.save(tobe,'checkpoints/VGG_embeded.t7') 
print('Embedding Done!')

###############################################################################################################
# train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU

global error_history

model = vgg()
model, sd = load_model(model, 'VGG_embeded', False)

if torch.cuda.is_available():
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)


transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])
batch_size = 256
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,download = True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,download = True, transform=transform)
trainloader = DataLoader(train_dataset, batch_size=batch_size)
testloader = DataLoader(test_dataset, batch_size=batch_size)

#optimizer = optim.SGD([w for name, w in model.named_parameters() if not 'mask' in name], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
optimizer = optim.Adam([w for name, w in model.named_parameters() if not 'mask' in name], lr=args.lr, betas=(0.9, 0.99))
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,args.epochs, eta_min=1e-10)
criterion = nn.CrossEntropyLoss()

error_history = []
for epoch in tqdm(range(args.epochs)):
    train_with_grad_control(model, trainloader, criterion, optimizer)
    validate(model, epoch, testloader, criterion, checkpoint='VGG_done_3*3')
    scheduler.step()