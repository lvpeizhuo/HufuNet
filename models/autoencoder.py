# -*- coding: utf-8 -*
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import OrderedDict

device = torch.device("cuda:0")
if torch.cuda.is_available():
    print("cuda is available")
    os.environ["CUDA_VISIBLE_DEVICES"]='0' #

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.encoder = nn.Sequential( 
            nn.Conv2d(1, 20, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(20, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        ).to(device)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 20, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(20, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        ).to(device)
    def forward(self, x):
        encoded = self.encoder(x.to(device))
        decoded = self.decoder(encoded.to(device))
        return encoded,decoded


def encoder():
    return Encoder()

