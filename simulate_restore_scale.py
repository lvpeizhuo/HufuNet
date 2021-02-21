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


# Restore coffs of filters

Model_10 = torch.load('checkpoints/VGG_done_3*3.t7')
Model_50 = torch.load('checkpoints/VGG_EnlargeC.t7')
#Model_50 = torch.load('checkpoints/VGG_RestoreOrder.t7')

for name in [key for key in Model_10['net'].keys() if 'conv' in key]:
    print(name)
    (a,b,c,d) = Model_10['net'][name].size() # a:out_channel  b:in_channel
    
    list10 = []
    list50 = []
    
    for i in range(a):
        U10, sig10, V10 = np.linalg.svd(Model_10['net'][name].cpu().numpy()[i,:,:,:].reshape(b,c*d))
        U50, sig50, V50 = np.linalg.svd(Model_50['net'][name].cpu().numpy()[i,:,:,:].reshape(b,c*d))
        
        coff = np.average(sig50/sig10)

        Model_50['net'][name][i,:,:,:] /= coff
        
torch.save(Model_50,'checkpoints/VGG_RestoreCoff.t7')        
