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

 
model = torch.load('checkpoints/VGG_prune60.t7', map_location='cpu')
print(model['net'].keys())

for i,name in enumerate([key for key in model['net'].keys() if 'conv' in key and 'weight' in key]):
        print(i)
        if i==8: # fc layer skip filters' order changing
            (a,b) = model['net'][name].size()
            tmp = model['net'][name][:,0].view(a,1)
            model['net'][name] = model['net'][name][:,1:]
            model['net'][name] = torch.cat([model['net'][name],tmp],1)
            break
        # transfer filters' order: set first filter to tail
        (a,b,c,d) = model['net'][name].size()
        tmp = model['net'][name][0,:,:,:].view(1,b,c,d)
        model['net'][name] = model['net'][name][1:,:,:,:]
        model['net'][name] = torch.cat([model['net'][name],tmp],0)
        
        # transfer filters' inside order: set first layer to tail
        if i!=0: # conv1 layer skip filters' inside order changing
            tmp = model['net'][name][:,0,:,:].view(a,1,c,d)
            model['net'][name] = model['net'][name][:,1:,:,:]
            model['net'][name] = torch.cat([model['net'][name],tmp],1)

torch.save(model,'checkpoints/VGG_ReOrder.t7')
        