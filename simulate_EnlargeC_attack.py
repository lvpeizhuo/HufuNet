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

model = torch.load('checkpoints/VGG_ReOrder.t7',map_location='cpu')
for name in [key for key in model['net'].keys() if 'conv' in key and 'weight' in key]:
    nameo = name
    if nameo == 'conv1_1.weight':
        model['net'][name] *= 2**7
    elif nameo == 'conv2_1.weight':
        model['net'][name] /= 2**4
    elif nameo == 'conv3_1.weight':
        model['net'][name] /= 2**3
    elif nameo == 'conv3_2.weight':
        model['net'][name] /= 2**3
    elif nameo == 'conv4_1.weight':
        model['net'][name] *= 2**7
    elif nameo == 'conv4_2.weight':
        model['net'][name] /= 2**2
    elif nameo == 'conv5_1.weight':
        model['net'][name] /= 2
    elif nameo == 'conv5_2.weight':
        model['net'][name] /= 2
torch.save(model,'checkpoints/VGG_EnlargeC.t7')
