#!usr/bin/env python
# -*- coding: utf-8 -*
import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module): #VGG11 已经�?3 * 32 * 32 兼容
    def __init__(self, in_planes = 3, planes = 6, stride=1, mode='train'):
        super(VGG, self).__init__()
        self.mode = mode
        # 3 * 32 * 32  两个输出通道�?3, 32, 3)通道的化为一体，就是(3, 64, 3)！！！！！！！！！！�?        
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding = 1, bias = False) # 
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding = 1, bias = False) # 
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding = 1, bias = False)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding = 1, bias = False)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding = 1, bias = False)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding = 1, bias = False)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding = 1, bias = False)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding = 1, bias = False)
        
        
        self.mask1_1 = nn.Conv2d(3, 64, 3, padding = 1, bias = False)
        self.mask1_1.weight.data = torch.ones(self.mask1_1.weight.size())
        self.grad1_1 = nn.Conv2d(3, 64, 3, padding = 1, bias = False)
        self.grad1_1.weight.data = torch.ones(self.grad1_1.weight.size())

        self.mask2_1 = nn.Conv2d(64, 128, 3, padding = 1, bias = False)
        self.mask2_1.weight.data = torch.ones(self.mask2_1.weight.size())
        self.grad2_1 = nn.Conv2d(64, 128, 3, padding = 1, bias = False)
        self.grad2_1.weight.data = torch.ones(self.grad2_1.weight.size())
        
        self.mask3_1 = nn.Conv2d(128, 256, 3, padding = 1, bias = False)
        self.mask3_1.weight.data = torch.ones(self.mask3_1.weight.size())
        self.grad3_1 = nn.Conv2d(128, 256, 3, padding = 1, bias = False)
        self.grad3_1.weight.data = torch.ones(self.grad3_1.weight.size())
        
        self.mask3_2 = nn.Conv2d(256, 256, 3, padding = 1, bias = False)
        self.mask3_2.weight.data = torch.ones(self.mask3_2.weight.size())
        self.grad3_2 = nn.Conv2d(256, 256, 3, padding = 1, bias = False)
        self.grad3_2.weight.data = torch.ones(self.grad3_2.weight.size())

        self.mask4_1 = nn.Conv2d(256, 512, 3, padding = 1, bias = False)
        self.mask4_1.weight.data = torch.ones(self.mask4_1.weight.size())
        self.grad4_1 = nn.Conv2d(256, 512, 3, padding = 1, bias = False)
        self.grad4_1.weight.data = torch.ones(self.grad4_1.weight.size())
        
        self.mask4_2 = nn.Conv2d(512, 512, 3, padding = 1, bias = False)
        self.mask4_2.weight.data = torch.ones(self.mask4_2.weight.size())
        self.grad4_2 = nn.Conv2d(512, 512, 3, padding = 1, bias = False)
        self.grad4_2.weight.data = torch.ones(self.grad4_2.weight.size())

        self.mask5_1 = nn.Conv2d(512, 512, 3, padding = 1, bias = False)
        self.mask5_1.weight.data = torch.ones(self.mask5_1.weight.size())
        self.grad5_1 = nn.Conv2d(512, 512, 3, padding = 1, bias = False)
        self.grad5_1.weight.data = torch.ones(self.grad5_1.weight.size())
        
        self.mask5_2 = nn.Conv2d(512, 512, 3, padding = 1, bias = False)
        self.mask5_2.weight.data = torch.ones(self.mask5_2.weight.size())
        self.grad5_2 = nn.Conv2d(512, 512, 3, padding = 1, bias = False)
        self.grad5_2.weight.data = torch.ones(self.grad5_2.weight.size())
        

        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        
        self.conv1_1.weight.data = torch.mul(self.conv1_1.weight,  self.mask1_1.weight)
        self.conv2_1.weight.data = torch.mul(self.conv2_1.weight,  self.mask2_1.weight)
        self.conv3_1.weight.data = torch.mul(self.conv3_1.weight,  self.mask3_1.weight)
        self.conv3_2.weight.data = torch.mul(self.conv3_2.weight,  self.mask3_2.weight)
        self.conv4_1.weight.data = torch.mul(self.conv4_1.weight,  self.mask4_1.weight)
        self.conv4_2.weight.data = torch.mul(self.conv4_2.weight,  self.mask4_2.weight)
        self.conv5_1.weight.data = torch.mul(self.conv5_1.weight,  self.mask5_1.weight)
        self.conv5_2.weight.data = torch.mul(self.conv5_2.weight,  self.mask5_2.weight)
        
        
        x = F.relu(self.conv1_1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2_1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        #print('ninniinin')
        return x
    
    def __prune__(self, threshold):
        self.mode = 'prune'
        
        self.mask1_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv1_1.weight), threshold).float(), self.mask1_1.weight)
        self.mask2_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv2_1.weight), threshold).float(), self.mask2_1.weight)
        self.mask3_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv3_1.weight), threshold).float(), self.mask3_1.weight)
        self.mask3_2.weight.data = torch.mul(torch.gt(torch.abs(self.conv3_2.weight), threshold).float(), self.mask3_2.weight)
        self.mask4_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv4_1.weight), threshold).float(), self.mask4_1.weight)
        self.mask4_2.weight.data = torch.mul(torch.gt(torch.abs(self.conv4_2.weight), threshold).float(), self.mask4_2.weight)
        self.mask5_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv5_1.weight), threshold).float(), self.mask5_1.weight)
        self.mask5_2.weight.data = torch.mul(torch.gt(torch.abs(self.conv5_2.weight), threshold).float(), self.mask5_2.weight)
    '''
    def __prune__(self, threshold):
        self.mode = 'prune'
        
        self.mask1_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv1_1.weight), threshold).float(), self.mask1_1.weight)
        self.conv1_1.weight.data = torch.mul(self.conv1_1.weight, self.mask1_1.weight)
        self.mask1_1.weight.data = torch.ones(self.mask1_1.weight.size()).cuda() ################################
        
        self.mask2_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv2_1.weight), threshold).float(), self.mask2_1.weight)
        self.conv2_1.weight.data = torch.mul(self.conv2_1.weight, self.mask2_1.weight)
        self.mask2_1.weight.data = torch.ones(self.mask2_1.weight.size()).cuda() ################################
        
        self.mask3_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv3_1.weight), threshold).float(), self.mask3_1.weight)
        self.conv3_1.weight.data = torch.mul(self.conv3_1.weight, self.mask3_1.weight)
        self.mask3_1.weight.data = torch.ones(self.mask3_1.weight.size()).cuda() ################################
        
        self.mask3_2.weight.data = torch.mul(torch.gt(torch.abs(self.conv3_2.weight), threshold).float(), self.mask3_2.weight)
        self.conv3_2.weight.data = torch.mul(self.conv3_2.weight, self.mask3_2.weight)
        self.mask3_2.weight.data = torch.ones(self.mask3_2.weight.size()).cuda() ################################
        
        self.mask4_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv4_1.weight), threshold).float(), self.mask4_1.weight)
        self.conv4_1.weight.data = torch.mul(self.conv4_1.weight, self.mask4_1.weight)
        self.mask4_1.weight.data = torch.ones(self.mask4_1.weight.size()).cuda() ################################
        
        self.mask4_2.weight.data = torch.mul(torch.gt(torch.abs(self.conv4_2.weight), threshold).float(), self.mask4_2.weight)
        self.conv4_2.weight.data = torch.mul(self.conv4_2.weight, self.mask4_2.weight)
        self.mask4_2.weight.data = torch.ones(self.mask4_2.weight.size()).cuda() ################################
        
        self.mask5_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv5_1.weight), threshold).float(), self.mask5_1.weight)
        self.conv5_1.weight.data = torch.mul(self.conv5_1.weight, self.mask5_1.weight)
        self.mask5_1.weight.data = torch.ones(self.mask5_1.weight.size()).cuda() ################################
        
        self.mask5_2.weight.data = torch.mul(torch.gt(torch.abs(self.conv5_2.weight), threshold).float(), self.mask5_2.weight)
        self.conv5_2.weight.data = torch.mul(self.conv5_2.weight, self.mask5_2.weight)
        self.mask5_2.weight.data = torch.ones(self.mask5_2.weight.size()).cuda() ################################
    '''
def vgg():
    return VGG()

