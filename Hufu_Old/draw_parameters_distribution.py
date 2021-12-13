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
import random
import argparse
import hmac
import hashlib
from models import *
from utils  import *
from tqdm   import tqdm

torch.backends.cudnn.enabled = False

def cos_sim(x, y):
    x_=x-np.mean(x)
    y_=y-np.mean(y)
    tmp = np.linalg.norm(x_)*np.linalg.norm(y_)
    # d1 = np.dot(x_,y_)/tmp
    d1 = np.dot(x_,y_)/ np.where(tmp!=0,tmp,1e-20)
    return 0.5+d1/2.0

ori = torch.load('checkpoints/VGG_done_3*3.t7', map_location='cpu')
new = torch.load('checkpoints/VGG_RestoreOrder.t7', map_location='cpu')

#print(ori['net'].keys())
#print(new['net'].keys())

same = []
diff = []

for name in [key for key in ori['net'].keys() if 'conv' in key and 'weight' in key]:
    print(name)
    (a,b,c,d) = ori['net'][name].size()
    for i in range(a):
        tmp1 = ori['net'][name][i,:,:,:].cpu().numpy().flatten()
        for j in range(a):
            tmp2 = new['net'][name][j,:,:,:].cpu().numpy().flatten()
            #tmp2 = new['net']['module.'+name][j,:,:,:].cpu().numpy().flatten()
            if i==j:
                same.append(cos_sim(tmp1,tmp2))
            else:
                diff.append(cos_sim(tmp1,tmp2))

print('VGG')
same = np.array(same)
diff = np.array(diff)

print('same smallest: ',str(same.min()))
print('diff largest: ',str(same.max()))
print('same mean/std: ',same.mean(), same.std())
print('diff mean/std: ',diff.mean(), diff.std())

#sys.exit()
'''
import numpy as np
import matplotlib.pyplot as plt

fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6))

ax0.hist(same, 200, range=(0,1),cumulative=False, normed=1, facecolor='blue', alpha=0.5)
ax0.set_title('same')

ax1.hist(diff, 200, range=(0,1),cumulative=False, normed=1, facecolor='pink', alpha=0.5)
ax1.set_title("diff")

fig.subplots_adjust(hspace=0.4)
plt.savefig('distribution.jpg')
'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

figsize = 11,9
figure, ax = plt.subplots(figsize=figsize)
 
ax.hist(same, 200, range=(0,1),cumulative=False, normed=1, facecolor='blue', alpha=0.5)
ax.hist(diff, 200, range=(0,1),cumulative=False, normed=1, facecolor='pink', alpha=0.5)


font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 23,
}
plt.legend(['same', 'different'],loc='upper left', prop=font1)
 
plt.tick_params(labelsize=23)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
 
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 30,
}
#plt.title('Resnet18',font2,y=-0.5)
plt.xlabel('Cosine Similarity',font2)
plt.ylabel('Frequency',font2)
 
plt.savefig('distribution.png',dpi=500)
#plt.show()

















'''
ori = torch.load('checkpoints/FLenet.t7', map_location='cpu')
new = torch.load('checkpoints/new_hufu.t7', map_location='cpu')

list1 = []
list2 = []

for name in [key for key in ori['net'].keys() if 'conv' in key]:
    list1.extend(ori['net'][name].view(-1).numpy().tolist())
    list2.extend(new['net'][name].view(-1).numpy().tolist())

list1 = np.array(list1)
list2 = np.array(list2)

# remove 0
index = np.where(list1!=0)
list1 = list1[index]
list2 = list2[index]

# calculate the substraction of abs
list2 = abs(list2-list1) # substraction
list1 = abs(list1)
copy  = sorted(list1.copy())

print('Googlenet hufu')
print('######################################')
for r in range(5,85,5):
    threshold = copy[int(len(list1)*r*0.01)]
    print(str(r),'%: ',str(threshold))
    
    # remove XX% least numbers
    index = np.where(list1>threshold)
    list1 = list1[index]
    list2 = list2[index]
    
    ratio = list2/list1
    
    print('mean: ',str(np.mean(ratio)),'\tstd: ',str(np.std(ratio)))

    # import numpy as np
    # import matplotlib.pyplot as plt
    # plt.style.use('seaborn-white')
    # plt.hist(ratio, 50, range=(0,6),cumulative=True, normed=1, facecolor='blue', alpha=0.5) 
    # plt.savefig('distribution.jpg')
    
    # index = ratio.argmax()
    # print(ratio[index])
'''


























sys.exit(0)

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def cos_sim(x, y):
    x_=x-np.mean(x)
    y_=y-np.mean(y)
    d1 = np.dot(x_,y_)/(np.linalg.norm(x_)*np.linalg.norm(y_))
    return 0.5+d1/2.0


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
Model_50 = torch.load('checkpoints/VGG_done_finetune100.t7')


for name in Model_10['net'].keys():
    if 'conv' in name and name != 'module.conv1_1.weight' :
        count=0
        
        (a,b,c,d) = Model_10['net'][name].size() # a:out_channel  b:in_channel
        
        # simulate_mischange_2_channel
        simulate_mischange_2_channel = list(range(a))
        pos1 = random.choice(simulate_mischange_2_channel)
        simulate_mischange_2_channel.remove(pos1)
        pos2 = random.choice(simulate_mischange_2_channel)
        simulate_mischange_2_channel.remove(pos2)
        pos3 = random.choice(simulate_mischange_2_channel)
        simulate_mischange_2_channel.remove(pos3)
        pos4 = random.choice(simulate_mischange_2_channel)
        simulate_mischange_2_channel.remove(pos4)
        
        
        list10 = []
        list50 = []
        for i in range(b):
            U10, sig10, V10 = np.linalg.svd(Model_10['net'][name].cpu().numpy()[:,i,:,:].reshape(a,c*d))
            U50, sig50, V50 = np.linalg.svd(Model_50['net'][name].cpu().numpy()[:,i,:,:].reshape(a,c*d))
            
            # list10.append(np.vstack((U10[:,:min(a,c*d)],V10.T[:min(a,c*d),:])).flatten('f'))
            # list50.append(np.vstack((U50[:,:min(a,c*d)],V50.T[:min(a,c*d),:])).flatten('f'))
            # list10.append(U10.flatten('f'))
            # list50.append(U50.flatten('f'))
            list10.append(Model_10['net'][name].cpu().numpy()[:,i,:,:].flatten())
            tmp = Model_50['net'][name].cpu().numpy()[:,i,:,:].copy()
            tmp[pos1,:,:] = Model_50['net'][name].cpu().numpy()[pos2,i,:,:]
            tmp[pos2,:,:] = Model_50['net'][name].cpu().numpy()[pos1,i,:,:]
            tmp[pos3,:,:] = Model_50['net'][name].cpu().numpy()[pos4,i,:,:]
            tmp[pos4,:,:] = Model_50['net'][name].cpu().numpy()[pos3,i,:,:]
            
            list50.append(tmp.flatten())
            
        for i in range(len(list50)):
            result=[]
            for j in range(len(list10)):
                result.append(cos_sim(list10[j], list50[i]))
            tmp=np.argsort(-np.array(result))
            #print(tmp)
            #print('\n')
            if i==tmp[0]:
                count+=1
            #else:
                print('result: '+str(i)+'\tsimilarity: '+str(result))#[i]
        print(name)    
        print(str(count)+'/'+str(b)+'\r\r')
        break 
        







    
     
     
     
     
     
     
     