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

torch.backends.cudnn.enabled = False

################################################################################################################
# find embed location and embed the parameters back to hufu

seed = "2020"


def extract(EmbededNet,SavePosition):
    hufu = torch.load('weight/model_encoder_param.t7', map_location='cpu')
    tobe = EmbededNet
    ufuh = torch.load('weight/model_decoder_param.t7', map_location='cpu')


    pn = tobe
    tobe={}
    tobe['net'] = pn
    
    print(pn.keys())

    #insert_channel=[2,5,8,12,15,18,20,22,25,28,31,34,42,50]

    tmp2 = torch.tensor([]).cuda()
    mask2 = torch.tensor([]).cuda()
    for name2 in [key for key in pn.keys() if 'conv' in key and 'weight' in key]: #VGG
    #for name2 in [key for key in pn.keys() if 'conv' in key and 'weight' in key and 'layer' in key and 'shortcut' not in key]: #Res
    #for name2 in [key for key in pn.keys() if 'conv' in key and 'weight' in key and 'branch' in key and 'bn' not in key]: #Googlenet
        if(pn[name2].size()[0]>=50):
            for i in range(tobe['net'][name2].size()[0]):
                tmp2 = torch.cat([tmp2, pn[name2][i].view(-1)], 0)
                mask2 = torch.cat([mask2, pn[name2.replace('conv','mask')][i].view(-1)], 0)
            tmpsze=pn[name2].size()
    grad2 = torch.ones(tmp2.size())
    len2 = tmp2.size()[0]


    block_id = 1 # record current embed block id, and involve block_id into hash message      
    for i,name1 in enumerate([key for key in hufu.keys() if 'weight' in key]): # 1. get hufu model's layer name
        (in1,out1,h1,w1) = hufu[name1].size()
        for j in range(in1): 
            for k in range(out1): # embed block by block
                message = 0.0

                for m in range(h1): # use all number info in block as message
                    for n in range(w1):
                        nameo=str(4-int(name1[0]))+name1[1:]
                        message += ufuh[nameo][j,k,m,n]  # Xor operation
                message *= block_id # import block_id info into message
                block_id += 1 # update block id
                mac = hmac.new(bytes(seed, encoding='utf-8'),message.numpy(),hashlib.sha256)  # hash
                pos = int(mac.hexdigest(), 16) % len2 # compute embed position
                for m in range(h1): # cope with collision and embed
                    for n in range(w1):
                        while grad2[pos]<1:
                            pos = (pos+1)%len2
                        hufu[name1][j,k,m,n] = tmp2[pos]
                        #hufu[name1.replace('conv','mask')][j,k,m,n] = mask2[pos]
                        grad2[pos] = 0

    if(SavePosition != ''):    
        torch.save(hufu,SavePosition)
    return hufu
'''
for qid in tqdm([10,20,30,40,50,60,70,80,90]):
   test = extract(torch.load('checkpoints/xxx/Googlenet/Googlenet_prune'+str(qid)+'.t7', map_location='cpu'),'checkpoints/xxx/Googlenet/encoder_Googlenet_prune'+str(qid)+'.t7')
   '''

#test = extract(torch.load('checkpoints/clean/VGG_clean.t7', map_location='cpu'),'checkpoints/FakeA/encoder_VGG_clean.t7')