# -*- coding: utf-8 -*
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class FLeNet(nn.Module):
    def __init__(self, in_place = 3, planes = 6, stride = 1, mode = 'train'):
        super(FLeNet, self).__init__()

        self.mode = mode

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.relu1 = nn.ReLU()
        #self.bn1 = nn.BatchNorm2d(num_features = 32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias = False)
        self.relu2 = nn.ReLU()
        #self.bn2 = nn.BatchNorm2d(num_features = 64)
        self.pool1 = nn.MaxPool2d(kernel_size = 2)

        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.relu3 = nn.ReLU()
        #self.bn3 = nn.BatchNorm2d(num_features = 128)
        self.pool2 = nn.MaxPool2d(kernel_size = 2)

        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.relu4 = nn.ReLU()
        #self.bn4 = nn.BatchNorm2d(num_features = 128)
        self.pool3 = nn.MaxPool2d(kernel_size = 2)

        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.relu5 = nn.ReLU()
        #self.bn5 = nn.BatchNorm2d(num_features = 64)
        self.pool4 = nn.MaxPool2d(kernel_size = 2)

        
        self.mask1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.mask1.weight.data = torch.ones(self.mask1.weight.size())
        self.mask2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias = False)
        self.mask2.weight.data = torch.ones(self.mask2.weight.size())
        self.mask3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.mask3.weight.data = torch.ones(self.mask3.weight.size())
        self.mask4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.mask4.weight.data = torch.ones(self.mask4.weight.size())
        self.mask5 = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.mask5.weight.data = torch.ones(self.mask5.weight.size())
        
        
        self.fc1 = nn.Linear(in_features=64, out_features=10)
        self.fc2 = nn.Linear(in_features=64, out_features=10)
        
    def forward(self, x):

        
        self.conv1.weight.data = torch.mul(self.conv1.weight,  self.mask1.weight)
        self.conv2.weight.data = torch.mul(self.conv2.weight,  self.mask2.weight)
        self.conv3.weight.data = torch.mul(self.conv3.weight,  self.mask3.weight)
        self.conv4.weight.data = torch.mul(self.conv4.weight,  self.mask4.weight)
        self.conv5.weight.data = torch.mul(self.conv5.weight,  self.mask5.weight)
        

        x = self.conv1(x)
        x = self.relu1(x)
        #x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        #x = self.bn2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.relu3(x)
        #x = self.bn3(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.relu4(x)
        #x = self.bn4(x)
        x = self.pool3(x)
        x = self.conv5(x)
        x = self.relu5(x)
        #x = self.bn5(x)
        x = self.pool4(x)

        x = x.view(-1, 64)

        x = self.fc1(x)

        return x

    def __prune__(self, threshold):
        self.mode = 'prune'
        
        self.mask1.weight.data = torch.mul(torch.gt(torch.abs(self.conv1.weight), threshold).float(), self.mask1.weight)
        self.mask2.weight.data = torch.mul(torch.gt(torch.abs(self.conv2.weight), threshold).float(), self.mask2.weight)
        self.mask3.weight.data = torch.mul(torch.gt(torch.abs(self.conv3.weight), threshold).float(), self.mask3.weight)
        self.mask4.weight.data = torch.mul(torch.gt(torch.abs(self.conv4.weight), threshold).float(), self.mask4.weight)
        self.mask5.weight.data = torch.mul(torch.gt(torch.abs(self.conv5.weight), threshold).float(), self.mask5.weight)
        

def flenet():
    return FLeNet()
