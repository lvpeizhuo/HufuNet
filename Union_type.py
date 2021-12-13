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
from extract import extract
from models.autoencoder import *

torch.backends.cudnn.enabled = False

seed="2020"

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--model',      default='VGG', help='resnet9/18/34/50, wrn_40_2/_16_2/_40_1')
parser.add_argument('--prepared_model',      default=None, type=str)
parser.add_argument('--checkpoint',default='VGG_done',type=str)
parser.add_argument('--data_loc',   default='/disk/scratch/datasets/MNIST', type=str)
parser.add_argument('--start_mixthreshold_epoch', default=5, type=int)
parser.add_argument('--fuse_per_epoch', default=5, type=int)
parser.add_argument('--GPU', default='0,1', type=str,help='GPU to use')
parser.add_argument('--epochs',     default=50, type=int)
parser.add_argument('--lr',         default=0.1)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay')
parser.add_argument('--weight_decay', default=0.0005, type=float)
args = parser.parse_args()


hufu = torch.load('weight/model_encoder_param.t7', map_location='cpu')
ufuh = torch.load('weight/model_decoder_param.t7', map_location='cpu')
tobe = torch.load(args.prepared_model, map_location='cpu')

layers_size={}
tmp2 = torch.tensor([])
pn = tobe['net']

for name2 in [key for key in pn.keys() if 'conv' in key and 'weight' in key]:
    if(tobe['net'][name2].size()[0]>=50):
        for i in range(tobe['net'][name2].size()[0]):
            tmp2 = torch.cat([tmp2, tobe['net'][name2][i].view(-1)], 0)
        tmpsze=tobe['net'][name2].size()
        layers_size[name2]=tmpsze[3]*tmpsze[1]*tmpsze[2]
# view(-1) function£ºflatten the matrix 1st dimention first, 2nd second, 3rd third

grad2 = torch.ones(tmp2.size())
len2 = tmp2.size()[0]

len1 = 0
block_id = 1 # record current embed block id, and involve block_id into hash message       
for i,name1 in enumerate([key for key in hufu.keys() if 'weight' in key]): # 1. get hufu model's layer name
    (in1,out1,h1,w1) = hufu[name1].size()
    len1 += in1*out1*h1*w1
    for j in range(in1): 
        for k in range(out1): # embed block by block
            message = 0.0
            for m in range(h1): # use all number info in block as message
                for n in range(w1):
                    nameo=str(4-int(name1[0]))+name1[1:]
                    message += ufuh[nameo][j,k,m,n]  # Xor operation
            message *= block_id # import block_id info into message
            block_id += 1 # update block id
            mac = hmac.new(bytes(seed, encoding='utf-8'),message.numpy(),hashlib.sha256)  # hash
            pos = int(mac.hexdigest(), 16) % len2 # compute embed position

            for m in range(h1): # cope with collision and embed
                for n in range(w1):
                    while grad2[pos] == 0:
                        pos = (pos+1)%len2
                    tmp2[pos]=hufu[name1][j,k,m,n]
                    #hufu[name1.replace('conv','mask')][j,k,m,n] = mask2[pos]
                    grad2[pos] = 0
            # new: embed 3*3 block by block
start_size = 0

print("Hufunet Size: ",len1)
print("Embeded Neural Net Size: ",len2)

#106292
for name2 in [key for key in layers_size.keys()]:
    for i in range(tobe['net'][name2].size()[0]):
        end_size = start_size + layers_size[name2]
        tobe['net'][name2][i] = tmp2[start_size:end_size].reshape(tobe['net'][name2][0].size())
        #tobe[name2.replace('conv','grad').replace('.0.','.')][i] = grad2[start_size:end_size].reshape(tobe[name2][0].size())
        tobe['net'][name2.replace('conv','grad')][i] = grad2[start_size:end_size].reshape(tobe['net'][name2][0].size())
        start_size = end_size
    
torch.save(tobe,'checkpoints/'+args.model+'_embeded.t7',_use_new_zipfile_serialization=False) 
print('Embedding Done!')

###############################################################################################################
# train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU
models = {'VGG' : vgg(),
          'FLenet': flenet(),
          'ResNet18': ResNet18(),
          'ResNet34':ResNet34(),
          'googlenet':googlenet()}
model = models[args.model]

global error_history

model2 = encoder()

#model, sd = load_model(model, 'VGG_embeded', False)
k=torch.load('checkpoints/'+args.model+'_embeded.t7')['net']


if torch.cuda.is_available():
    model = model.cuda()
    model2 = model2.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model2 = nn.DataParallel(model2)
model.to(device)
model2.to(device)

model.load_state_dict(k)

model2.encoder.load_state_dict(hufu)
model2.decoder.load_state_dict(ufuh)

transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])
batch_size = 64
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


transform2 = transforms.Compose([transforms.ToTensor()])
batch_size2 = 500
train_dataset2 = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform2)
trainloader2 = DataLoader(train_dataset2, batch_size=batch_size2)
optimizer2 = optim.Adam(model2.parameters(),lr=0.001)
scheduler2 = lr_scheduler.CosineAnnealingLR(optimizer2,1, eta_min=1e-10)
criterion2 = nn.MSELoss()


train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,download = True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,download = True, transform=transform_test)

trainloader = DataLoader(train_dataset, batch_size=batch_size)
testloader = DataLoader(test_dataset, batch_size=batch_size)

optimizer = optim.SGD([w for name, w in model.named_parameters() if not 'mask' in name], lr=args.lr, momentum=0.9, weight_decay=5e-4)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer,args.epochs, eta_min=1e-10)

error_history = []
prevGH = 0
prevGM = 0
prevLoss = 0
prevRatio = 1

class MultiLoss(nn.Module):
    def __init__(self):
        super(MultiLoss, self).__init__()
    def forward(self, outputs, targets, prevGH, prevGM, alpha, gamma,signal,prevRatio):
        loss_func = nn.CrossEntropyLoss().cuda()
        loss1 = loss_func(outputs, targets)
        beta1 = torch.abs(torch.tensor(prevGM/prevGH))
        beta2 = prevGM * torch.abs(1/(0.6523 - loss1))  / 0.0000800375825259 
        loss3 = beta1*beta2
        loss2 = prevGM * torch.abs(1/(0.6523 - loss1)) - gamma * 0.0000800375825259 
        return loss1 + 0.01*alpha*torch.abs(10-beta1)+ 0.01*alpha*abs(prevRatio)
        
class SingleLoss(nn.Module):
    def __init__(self):
        super(SingleLoss, self).__init__()
    def forward(self, outputs, targets, prevGH, prevGM, alpha, gamma,signal,prevRatio):
        loss_func = nn.CrossEntropyLoss().cuda()
        loss1 = loss_func(outputs, targets)
        return loss1 
        
criterion = SingleLoss()
Mgrad_test = torch.tensor([0]).to(device)
Hgrad_test = torch.tensor([0]).to(device)

def adjust_learning_rate(init_lr, optimizer, epoch, lradj):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = init_lr * (0.1 ** (epoch // lradj))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr     
         
for epoch in tqdm(range(args.epochs)):

    torch.autograd.set_detect_anomaly(True)

    if(epoch >= args.start_mixthreshold_epoch):
        criterion = MultiLoss()
        print("Actvate Loss2.")

    for step, (x, b_label) in enumerate(trainloader2):
        encoded, decoded  = model2(x)
        loss2 = criterion2(decoded, x.to(device))
        loss2.backward()
        optimizer2.step()
        optimizer2.zero_grad()
        
    new_hufu = model2.encoder.state_dict()   

    if(epoch > args.epochs*0.3):
        alpha = 0.001
    else:
        alpha=0.0005
    result = train_with_grad_control(model, trainloader, criterion, optimizer,len1 ,len2, prevGH,prevGM,alpha,10,prevRatio,epoch)    
    
    [prevGH,prevGM,prevRatio,Mgrad_epo,Hgrad_epo] = result

    State = model.state_dict()  

    if(epoch % args.fuse_per_epoch == 0):
        model.load_state_dict(Fuse(State,new_hufu,ufuh))
        Mgrad_test = Mgrad_epo
        Hgrad_test = Hgrad_epo
    else:
        Mgrad_test += Mgrad_epo
        Hgrad_test += Hgrad_epo        
        validate(model, epoch, testloader, criterion, True,prevGH,prevGM,1,10,prevRatio)
        
    if(epoch % args.fuse_per_epoch == args.fuse_per_epoch-1):
        Mgrad_test = torch.abs(Mgrad_test/args.fuse_per_epoch)
        Hgrad_test = torch.abs(Hgrad_test/args.fuse_per_epoch)
        beta_1 = torch.mean(Mgrad_test) / torch.mean(Hgrad_test) 
        Var_Ratio = torch.var(Mgrad_test) / torch.var(Hgrad_test)
    scheduler.step()
validate_ori(model, epoch, testloader, criterion,args.checkpoint)