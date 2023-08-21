
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

model = encoder()


torch.backends.cudnn.enabled = True

print(os.environ["CUDA_VISIBLE_DEVICES"])
device = torch.device("cuda:0")
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

if torch.cuda.is_available():
    model = model.cuda()
model.to(device)

global error_history
batch_size=32
transform = transforms.Compose([transforms.ToTensor()])

test_dataset = torchvision.datasets.MNIST(root="/home/lpz/MHL/NewHufu/data/", train=False,download = True, transform=transform)
testloader = DataLoader(test_dataset, batch_size=batch_size)

visualize_num=8
f, a = plt.subplots(2, visualize_num, figsize=(visualize_num, 2)) 
model.eval()


def Evaluation(Encoder,Visualize = 1,PrintOut=1):

    model.encoder.load_state_dict(torch.load('./weight/model_encoder_param.t7'))

    model.encoder.load_state_dict(Encoder)

    model.decoder.load_state_dict(torch.load('./weight/model_decoder_param.t7'))


    for step, (x, b_label) in enumerate(testloader):
        encoded, decoded  = model(x)
        model.eval()
        MSE, SSIM = eval(x,decoded,PrintOut)
        if(Visualize == 1):
            visualize(model,test_dataset,visualize_num,f,a)
            plt.pause(20)
        aMSE = (MSE-0.18704618513584137)/0.511984192
        return aMSE.item(),SSIM.item()
'''
for qid in [90]:
    print(Evaluation(torch.load('checkpoints\\xxx\VGG\extracted_encoder_VGG_prune'+str(qid)+'.t7'),1,0))
    '''

#Evaluation(torch.load('checkpoints\extracted_encoder_VGG_prunezz.t7'))
