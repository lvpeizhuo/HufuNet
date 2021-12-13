# -*- coding: utf-8 -*
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import torchvision
import numpy as np
import torchvision.transforms as transforms

import os
import json
import argparse
import hmac
import hashlib
import datetime
from models import *
from utils  import *
from tqdm   import tqdm

from PIL import Image


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def cos_sim(x, y):
    x_=x-np.mean(x)
    y_=y-np.mean(y)
    d1 = np.dot(x_,y_)/(np.linalg.norm(x_)*np.linalg.norm(y_))
    return 0.5+d1/2.0

def PingFangCha(x,y):
    return np.sum((x-y)**2)


parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--model1',      default='FLenet', help='hufu model')
parser.add_argument('--model2',      default='VGG_done', help='to-be-protected-model after pruning/finetuning')
parser.add_argument('--data_loc',   default='/disk/scratch/datasets/MNIST', type=str)
parser.add_argument('--checkpoint', default='resnet18', type=str)
parser.add_argument('--GPU', default='0,1', type=str,help='GPU to use')
parser.add_argument('--epochs',     default=100, type=int)
parser.add_argument('--lr',         default=0.001)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay')
parser.add_argument('--weight_decay', default=0.0005, type=float)
args = parser.parse_args()


Model_10 = torch.load('checkpoints/VGG_done.t7')
Model_50 = torch.load('checkpoints/VGG_RestoreCoff.t7')

layer_names = [key for key in Model_10['net'].keys() if 'conv' in key and 'weight' in key]
print(layer_names)

correct = 0
error=0 # record the number of error filters
total=0 # record the number of total filters

for n,name in enumerate(layer_names[:8]):
    print(name)
    (a,b,c,d) = Model_10['net'][name].size() # a:out_channel  b:in_channel
    print(a,b,c,d)

    list10 = []
    list50 = []
    
    cur_layer = Model_50['net']['module.'+name].clone() # store current layer's parameters, prepare to replace 
    nxt_layer = Model_50['net']['module.'+layer_names[(n+1)]].clone()
    
    for i in range(a):
        U10, sig10, V10 = np.linalg.svd(Model_10['net'][name].cpu().numpy()[i,:,:,:].reshape(c*d,-1))
        U50, sig50, V50 = np.linalg.svd(Model_50['net']['module.'+name].cpu().numpy()[i,:,:,:].reshape(c*d,-1))
        # list10.append(np.vstack((U10[:,:min(a,c*d)],V10.T[:min(a,c*d),:])).flatten('f'))
        # list50.append(np.vstack((U50[:,:min(a,c*d)],V50.T[:min(a,c*d),:])).flatten('f'))
        list10.append(sig10)
        list50.append(sig50)
        
        #list10.append(Model_10['net'][name].cpu().numpy()[i,:,:,:].flatten())
        #list50.append(Model_50['net'][name].cpu().numpy()[i,:,:,:].flatten())
        
        #tmp10 = abs(Model_10['net'][name].cpu().numpy()[i,:,:,:])
        #tmp50 = abs(Model_50['net'][name].cpu().numpy()[i,:,:,:])
        #list10.append(tmp10[np.argsort(np.sort(tmp10)[:,0])].flatten())
        #list50.append(tmp50[np.argsort(np.sort(tmp50)[:,0])].flatten())
    
    for i in range(len(list50)):
        result=[]
        for j in range(len(list10)):
            result.append(cos_sim(list10[j], list50[i]))

            #result.append(PingFangCha(list10[j], list50[i]))
        
        tmp=np.argsort(-np.array(result))
        #tmp=np.argsort(np.array(result))
        #print(i)
        #print(tmp)

        for k in range(len(tmp)):
            if i!=tmp[k]:
                if (i+1)%a==tmp[k]:
                    correct+=1
                #error+=1
                    Model_50['net']['module.'+name][tmp[k],:,:,:] = cur_layer[i,:,:,:]
                    if n==7:
                        Model_50['net']['module.'+layer_names[n+1]][:,tmp[k]] = nxt_layer[:,i]
                    else:
                        Model_50['net']['module.'+layer_names[n+1]][:,tmp[k],:,:] = nxt_layer[:,i,:,:]
                    break
            k+=1
    print(correct,a)
    correct=0
torch.save(Model_50,'checkpoints/VGG_RestoreOrder.t7')        