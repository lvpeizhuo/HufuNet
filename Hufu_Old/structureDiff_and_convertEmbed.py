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

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--model1',      default='FLenet', help='hufu model')
parser.add_argument('--model2',      default='VGG_done_3*3', help='to-be-protected-model after pruning/finetuning')
parser.add_argument('--data_loc',   default='/disk/scratch/datasets/MNIST', type=str)
parser.add_argument('--checkpoint', default='resnet18', type=str)
parser.add_argument('--GPU', default='0,1', type=str,help='GPU to use')
parser.add_argument('--epochs',     default=100, type=int)
parser.add_argument('--lr',         default=0.001)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay')
parser.add_argument('--weight_decay', default=0.0005, type=float)
args = parser.parse_args()

################################################################################################################
# find embed location and embed the parameters back to hufu

seed = "2020"

hufu = torch.load('checkpoints/'+args.model1+'.t7', map_location='cpu')
tobe = torch.load('checkpoints/'+args.model2+'.t7', map_location='cpu')



tmp2 = torch.tensor([])
mask2 = torch.tensor([])
for name2 in [key for key in tobe['net'].keys() if 'conv' in key and 'weight' in key]:
    tmp2 = torch.cat([tmp2, tobe['net'][name2].view(-1)], 0)
    mask2 = torch.cat([mask2, tobe['net'][name2.replace('conv','mask')].view(-1)], 0)

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
            mac = hmac.new(bytes(seed, encoding='utf-8'),message.numpy(),hashlib.sha256)  # hash
            pos = int(mac.hexdigest(), 16) % len2 # compute embed position
            
            '''
            # original: embed one by one
            for m in range(h1): # cope with collision and embed
                for n in range(w1):
                    while grad2[pos]==0:
                        pos = (pos+1)%len2
                    hufu['net'][name1][j,k,m,n] = tmp2[pos]
                    hufu['net'][name1.replace('conv','mask')][j,k,m,n] = mask2[pos]
                    grad2[pos] = 0
            # new: embed 3*3 block by block
            '''
            pos = pos//9*9
            while grad2[pos]==0:
                pos = (pos+9)%len2
            hufu['net'][name1][j,k,:,:] = tmp2[pos:pos+9].view(h1,w1)
            hufu['net'][name1.replace('conv','mask')][j,k,:,:] = mask2[pos:pos+9].view(h1,w1)
            grad2[pos:pos+9] = 0
            
            
torch.save(hufu,'checkpoints/new_hufu.t7')    

###########################################################################################################################################################
# compute the value difference and structure difference

'''    
VGG = torch.load('checkpoints/VGG_embeded.t7', map_location='cpu')    
print(VGG['net']['module.grad3_2.weight'])
#print(torch.sum(torch.ones(VGG['net']['module.grad3_2.weight'].size())-VGG['net']['module.grad3_2.weight']))
'''

ori_hufu = torch.load('checkpoints/'+args.model1+'.t7', map_location='cpu')
new_hufu = torch.load('checkpoints/new_hufu.t7', map_location='cpu')

weight_error=0
weight_count=0
mask_error = 0
mask_count = 0
for name in ori_hufu['net'].keys(): # traverse all layers 
  if 'fc' in name:
      continue
  
  if 'mask' not in name:
    delta = new_hufu['net'][name] - ori_hufu['net'][name]
    delta = (delta==0)
    true  = torch.sum(delta!=0)
    false = torch.sum(delta==0)
    weight_error += false
    weight_count += (true+false)
  else:
    delta = new_hufu['net'][name] - ori_hufu['net'][name]
    delta = (delta==0)
    true  = torch.sum(delta!=0)
    false = torch.sum(delta==0)
    mask_error += false
    mask_count += (true+false)


print('weight difference: ' + '%.4f'%(weight_error.numpy()*1.0/weight_count.numpy()))
print('mask  difference: ' + '%.4f'%(mask_error.numpy()*1.0/mask_count.numpy()))


###########################################################################################################################################################
# compute rebuild hufu's accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("cuda ia available")
    os.environ["CUDA_VISIBLE_DEVICES"]='0' #

global error_history
error_history = []

models = {'VGG': vgg(),
          'FLenet': flenet()}
model = models[args.model1]

old_format=False
if 'wrn' in args.model1:
    old_format=True
model, sd = load_model(model, 'new_hufu', old_format)


if torch.cuda.is_available():
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)


transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])
batch_size = 256
#if args.model1 == 'FLenet' :
test_dataset = torch.load('FM_test')
testloader = DataLoader(test_dataset, batch_size=batch_size)

criterion = nn.CrossEntropyLoss()
prune_rate = 0
validate(model, prune_rate, testloader, criterion)






















    
    
    
    
    