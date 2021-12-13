

####
# be attention: I've modified the VGG model's __prune__ method, now is simulating real pruning method

# need to modify VGG model, roll back to normal prune
####

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
from models import *
from utils  import *
from tqdm   import tqdm

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model',      default='VGG', help='VGG-16, ResNet-18, LeNet')
parser.add_argument('--data_loc',   default='/disk/scratch/datasets/cifar', type=str)
parser.add_argument('--checkpoint', default='VGGB_max', type=str, help='Pretrained model to start from')
parser.add_argument('--prune_checkpoint', default='', type=str, help='Where to save pruned models')
parser.add_argument('--GPU', default='0,1', type=str,help='GPU to use')
parser.add_argument('--save_every', default=5, type=int, help='How often to save checkpoints in number of prunes (e.g. 10 = every 10 prunes)')
parser.add_argument('--cutout', action='store_true')
parser.add_argument('--prune_rate', default=10, type=int)
parser.add_argument('--lr',             default=0.001)
parser.add_argument('--weight_decay', default=0.0005, type=float)

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU

global error_history
error_history = []


model = vgg()

if torch.cuda.is_available():
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
batch_size = 32

k=torch.load('checkpoints/VGG_finetuned.t7')['net']
paolos = {}
for key in k.keys():
    paolos[key]=k[key]

model.load_state_dict(paolos)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,download = True, transform=transform) # **Notice**: Prune After Finetuning
    
finetune_trainset, finetune_testset = torch.utils.data.random_split(test_dataset,[8000, len(test_dataset)-8000])
finetune_trainloader = DataLoader(finetune_trainset, batch_size=batch_size)
testloader = DataLoader(test_dataset, batch_size=batch_size)

criterion = nn.CrossEntropyLoss()
validate_ori(model, -1, testloader, criterion)
'''
for qid in [10,20,30,40,50,60,70,80,90]:
    prune_rate=qid
    model_a,max_param = sparsify(model, prune_rate, get_prune_max = True)
    print('Prune rate: '+str(prune_rate)+'%, Delete maximum parameter:\t'+str(max_param))
    #torch.save(model_a,'checkpoints/VGG/nex/VGG_prune'+str(qid)+'.t7')
    validate_ori(model, args.prune_rate, testloader, criterion, checkpoint='/xxx/VGG_prune'+str(prune_rate)) #'VGG_EnlargeC_prune'+str(args.prune_rate)
'''

prune_rate=args.prune_rate
model_a,max_param = sparsify(model, prune_rate, get_prune_max = True)
print('Prune rate: '+str(prune_rate)+'%, Delete maximum parameter:\t'+str(max_param))
validate_ori(model, args.prune_rate, testloader, criterion, checkpoint='VGG_prune'+str(prune_rate)) 














