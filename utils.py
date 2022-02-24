from __future__ import print_function

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import time
import random

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from ssim import *

import json
seed="2020"

import matplotlib
import matplotlib as mpl
import hmac
import hashlib

import matplotlib.mlab as mlab
from scipy.stats import norm
import seaborn as sns 
sns.set_palette("hls")

CUDA_VISIBLE_DEVICES=0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]='0' #

global error_history
error_history = []

def frozen_seed(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_cifar_loaders(data_loc='./disk/scratch/datasets/cifar10/', batch_size=128, cutout=True, n_holes=1, length=16):
    num_classes = 10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if cutout:
        transform_train.transforms.append(Cutout(n_holes=n_holes, length=length))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10('./disk/scratch/datasets/cifar10/', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10('./disk/scratch/datasets/cifar10/', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

def load_model(model, sd, old_format=False):
    sd = torch.load('checkpoints/%s.t7' % sd, map_location='cpu')
    new_sd = model.state_dict()
    if 'state_dict' in sd.keys():
        old_sd = sd['state_dict']
    else:
        old_sd = sd['net']

    if old_format:
        # this means the sd we are trying to load does not have masks
        # and/or is named incorrectly
        keys_without_masks = [k for k in new_sd.keys() if 'mask' not in k]
        for old_k, new_k in zip(old_sd.keys(), keys_without_masks):
            new_sd[new_k] = old_sd[old_k]
    else:
        new_names = [v for v in new_sd]
        old_names = [v for v in old_sd]
        for i, j in enumerate(new_names):
            new_sd[j] = old_sd[old_names[i]]
#            print(j)
#            print()
#            if not 'mask' in j:
        #new_sd[j] = old_sd[old_names[i]]

    try:
        model.load_state_dict(new_sd)
    except:
        print('module!!!!!')
        
        new_sd = model.state_dict()
        old_sd = sd['state_dict']
        k_new = [k for k in new_sd.keys() if 'mask' not in k]
        k_new = [k for k in k_new if 'num_batches_tracked' not in k]
        for o, n in zip(old_sd.keys(), k_new):
            new_sd[n] = old_sd[o]
        

        model.load_state_dict(new_sd)
    return model, sd

def get_error(output, target, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return res

# count only conv params for now
def get_no_params(net, verbose=False, mask=False):
    params = net
    tot = 0
    for p in params:
        no = torch.sum(params[p]!=0)
        if 'conv' in p:
            tot += no
    return tot

def train(model, trainloader, criterion, optimizer):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(trainloader):
        input = torch.squeeze(input)
        target = torch.squeeze(target)
       # input = torch.unsqueeze(input, 1)
        input, target = input.to(device), target.to(device)
        '''
        print(input)
        print(input.size())
        print()
        print(target)
        time.sleep(2)
        '''
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = get_error(output.detach(), target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
def train_with_grad_control(model, trainloader, criterion, optimizer,HufuSize, ModelSize, prevGH, prevGM, alpha,gamma,prevRatio,epo):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    # switch to train mode
    model.train()
    grad_H_R = torch.tensor(0)
    grad_M_R = torch.tensor(0)
    Ratio = 0
    cnt = 0
    
    Hgrad_epo = torch.tensor([0]).to(device)
    Mgrad_epo = torch.tensor([0]).to(device)
    
    for i, (input, target) in enumerate(trainloader):
    
        Grad_H = torch.tensor(0)
        Grad_M = torch.tensor(0)
        ratio = torch.tensor(0)

        input = torch.squeeze(input)
        target = torch.squeeze(target)
       # input = torch.unsqueeze(input, 1)
        input, target = input.to(device), target.to(device)

        # compute output
        output = model(input)
        if(epo == 0):
            loss_func = nn.CrossEntropyLoss().cuda()
            loss = loss_func(output,target)
        else:
            loss = criterion(output, target,prevGH, prevGM, alpha,gamma,i%100,prevRatio) 

        # measure accuracy and record loss
        err1, err5 = get_error(output.detach(), target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        Mgrad_nonzeroes = 0
        Hgrad_nonzeroes = 0   
        Hgrad_list = torch.tensor([0]).to(device)
        Mgrad_list = torch.tensor([0]).to(device)
        
        Hgrad_list_i = torch.tensor([0]).to(device)
        Mgrad_list_i = torch.tensor([0]).to(device)
            
        for name,param in model.named_parameters():
            if 'conv' in name and 'weight' in name:
                name = name.rstrip('.weight') # 'conv1_1'
                name_sp=name.split('.')
                print(name_sp)
                torch.autograd.set_detect_anomaly(True)

                current_grad = getattr(model, name_sp[0]).weight.grad
                #current_grad = getattr(getattr(getattr(model, name_sp[0]),name_sp[1]),name_sp[2]).weight.grad

                hufu_mask = getattr(model,name_sp[0].replace('conv','grad')).weight

                b = torch.where(torch.abs(current_grad) <= 0.00001 , torch.full_like(current_grad, 0), torch.full_like(current_grad, 1))
                Mgrad_nonzeroes += torch.sum(b)
                
                Mgrad_list = torch.cat((Mgrad_list,torch.mul(current_grad,b).view(-1)),0)

                c = torch.where(hufu_mask == 0 , torch.full_like(hufu_mask, 1), torch.full_like(hufu_mask, 0))
                d = torch.mul(c,b)
                Hz = torch.sum(d)
                hufu_grad = torch.mul(d,current_grad)
                
                Hgrad_list_i = torch.cat((Hgrad_list_i,torch.mul(c,current_grad).view(-1)),0)
                Mgrad_list_i = torch.cat((Mgrad_list_i,current_grad.view(-1)),0)
                
                Hgrad_list = torch.cat((Hgrad_list,hufu_grad.view(-1)),0)

                Hgrad_nonzeroes += Hz  
                
                Grad_M = Grad_M + torch.sum(torch.abs(torch.mul(current_grad,b))).item()
                Grad_H = Grad_H + torch.sum(torch.abs(torch.mul(current_grad,d))).item()
                
                #unfiltered_grad = torch.mul(hufu_mask,current_grad)
                #getattr(model.module, name_sp[1]).weight.grad = current_grad
                getattr(model, name_sp[0]).weight.grad = current_grad

                # validate the gradient don't change in some piece of one single layer
                #getattr(model.module, name).weight.grad[:10,0,0,0] = torch.zeros(getattr(model.module, name).weight.grad[:10,0,0,0].size())

        optimizer.step()
        Hmean = torch.sum(Hgrad_list) / Hgrad_nonzeroes
        c = torch.where(Hgrad_list == 0 ,torch.full_like(Hgrad_list, Hmean.item()), Hgrad_list)
        Mmean = torch.sum(Mgrad_list) / Mgrad_nonzeroes
        d = torch.where(Mgrad_list == 0 , torch.full_like(Mgrad_list, Mmean.item()), Mgrad_list)
        VarH = torch.sum(pow(c-Hmean,2)) / Hgrad_nonzeroes
        VarM = torch.sum(pow(d-Mmean,2)) / Mgrad_nonzeroes
        ratio = VarM/VarH
        Ratio += ratio


        grad_M_R = Grad_M / Mgrad_nonzeroes.item()
        grad_H_R = Grad_H / Hgrad_nonzeroes.item()
        cnt += 1
        
        if(i == 0):
            Hgrad_epo = Hgrad_list_i
            Mgrad_epo = Mgrad_list_i
        else:
            Hgrad_epo += Hgrad_list_i
            Hgrad_epo += Hgrad_list_i
    print(cnt)       
    Ratio = Ratio / cnt
        
    return [grad_H_R,grad_M_R,Ratio,Mgrad_epo,Hgrad_epo]
    
def derive_gradient_heatmap(model, trainloader, criterion, optimizer, data):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    # record gradient heatmap info with 'mask' layer
    # 1. initialize the 'mask' layer all zeros
    for name,param in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            data['net'][name] = torch.zeros(data['net'][name].size())

    # switch to train mode
    model.train()
    
    
    for i, (input, target) in enumerate(trainloader):
        input = torch.squeeze(input)
        target = torch.squeeze(target)
       # input = torch.unsqueeze(input, 1)
        input, target = input.to(device), target.to(device)
        '''
        print(input)
        print(input.size())
        print()
        print(target)
        time.sleep(2)
        '''
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = get_error(output.detach(), target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        
        
        # 2. accumulate add gradient heatmap info on 'mask' layer
        for name,param in model.named_parameters():
            #print(name)
            if 'conv' in name and 'weight' in name:
                name = name.lstrip('module.').rstrip('.weight') # 'conv1_1'
                #print(name)
                data['net']['module.'+name+'.weight'] += getattr(model.module, name).weight.grad.detach().cpu()
                #print(data['net']['module.'+name+'.weight'])
        
        optimizer.step()

def validate(model, epoch, valloader, criterion, checkpoint,gradH,gradM,alpha,gamma,PrevRatio):
    global error_history

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    correct = 0
    _sum = 0

    for i, (input, target) in enumerate(valloader):
        input = torch.squeeze(input)
        target = torch.squeeze(target)
        input, target = input.to(device), target.to(device)
        # compute output
        output = model(input)
        output_np = output.cpu().detach().numpy()
        #print(output_np)
        target_np = target.cpu().detach().numpy()
        out_ys = np.argmax(output_np, axis = -1)
        '''
        print(out_ys)
        print()
        print(target_np)
        print()
        '''
        _ = out_ys == target_np
        correct += np.sum(_, axis = -1)
        _sum += _.shape[0]
        

        loss = criterion(output, target, gradH, gradM, alpha, gamma, 1,PrevRatio)
        err1, err5 = get_error(output.detach(), target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))
    print('_:', _,'\n', 'correct:', correct, '_sum', _sum)
    print('accuracy: {:.4f}'.format(correct*1.0 / _sum))
    print('Loss: ',loss)
    
    error_history.append(top1.avg)

        
def validate_ori(model, epoch, valloader, criterion, checkpoint = ''):
    global error_history

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    correct = 0
    _sum = 0

    for i, (input, target) in enumerate(valloader):
        input = torch.squeeze(input)
        target = torch.squeeze(target)
        input, target = input.to(device), target.to(device)
        # compute output
        output = model(input)
        output_np = output.cpu().detach().numpy()
        #print(output_np)
        target_np = target.cpu().detach().numpy()
        out_ys = np.argmax(output_np, axis = -1)
        '''
        print(out_ys)
        print()
        print(target_np)
        print()
        '''
        _ = out_ys == target_np
        correct += np.sum(_, axis = -1)
        _sum += _.shape[0]
        loss = criterion(output, target)
        err1, err5 = get_error(output.detach(), target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))
    #print('_:', _,'\n', 'correct:', correct, '_sum', _sum)
    print('accuracy: {:.4f}'.format(correct*1.0 / _sum))
    print('Loss: ',loss)
    
    error_history.append(top1.avg)
    if checkpoint:
        state = {
            'net': model.state_dict(),
            'masks': [w for name, w in model.named_parameters() if 'mask' in name],
            'epoch': epoch,
            'error_history': error_history,
        }
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        torch.save(state, './checkpoints/'+checkpoint+'.t7')
        
def visualize(model,trainset,num,f,a):
    plt.ion()  
    for i in range(num):
        view_data = trainset.data[2218:num+2218].type(torch.FloatTensor)/255.
        view_data=view_data.reshape(8,1,28,28)
        _, result = model(view_data)
        a[0][i].clear()
        a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
        a[0][i].set_xticks(()); a[0][i].set_yticks(())
        a[1][i].clear()
        a[1][i].imshow(np.reshape(result.data.to(device='cpu').numpy()[i], (28, 28)), cmap='gray')
        a[1][i].set_xticks(())
        a[1][i].set_yticks(())
        plt.draw()
        plt.pause(0.00001)
    plt.ioff()   

def eval(input,output,PrintOut):
    mse=MSE(input,output)
    # psnr=PSNR(input,output)
    ssim=SSIM(input,output)
    if(PrintOut==1):
        print("MSE="+str(mse.item())+" ,SSIM="+str(ssim.item()))
    return mse,ssim

def MSE(input,output):
    k=pow(torch.squeeze(input).to(device)-torch.squeeze(output).to(device),2).to(device)
    k=k.sum()/(500*784)
    return k

def PSNR(input,output):
    k=pow(torch.squeeze(input).to(device)-torch.squeeze(output).to(device),2).to(device)
    k=(255*255)/(k.sum()/(500*784))
    k=10*k.log10()
    return(k.mean())

def SSIM(input,output):
    return(get_SSIM(input.cuda(),output.cuda()))

def calculate_threshold(model, rate):
    empty = torch.Tensor()
    if torch.cuda.is_available():
        empty = empty.cuda()
    pre_abs = expand_model(model, empty)
    weights = torch.abs(pre_abs)

    return np.percentile(weights.detach().cpu().numpy(), rate)

def sparsify(model, prune_rate=50., get_prune_max=False):
    threshold = calculate_threshold(model, prune_rate)
    try:
        model.__prune__(threshold)
    except:
        model.module.__prune__(threshold) 
        
    if get_prune_max==True:
        return model,threshold
    return model
    
def sparsify_set0(model, prune_rate=50.):
    threshold = calculate_threshold(model, prune_rate)
    try:
        model.__prune__set0(threshold)
    except:
        model.module.__prune__set0(threshold)
    return model

def expand_model(model, layers=torch.Tensor()):
    #for name, layer in model.named_children():
    for name, layer in model.named_parameters():
        tmp = name.split('.')

        #if 'conv' not in name or 'weight' not in name or 'shortcut' in name or 'layer' not in name:
        if 'conv' not in name or 'weight' not in name or 'branch' not in name or 'bn' in name:
        #if 'conv' not in name or 'weight' not in name :
            continue 
        name = name.rstrip('.weight') # 'conv1_1'
        name_sp=name.split('.')
        #layers = torch.cat([layers,getattr(model, name_sp[0]).weight.view(-1)],0) #VGG
        layers = torch.cat([layers,getattr(getattr(getattr(model, name_sp[0]),name_sp[1]),name_sp[2]).weight.view(-1)],0) #VGG

    return layers

def get_neuron_value(model, x, neuron_idx,layer_input,layer_output):
    return get_layer(model,x,layer_output, layer_input)[:, neuron_idx].mean()
    
def get_layer_name(module, depth = -1, prefix = '',
                   use_filter = True, repeat = False,
                   seq_only = False, init = True) :
    layer_name_list: list[str] = []
    if init or (not seq_only or isinstance(module, nn.Sequential)) and depth != 0:
        for name, child in module.named_children():
            full_name = prefix + ('.' if prefix else '') + name  # prefix=full_name
            layer_name_list.extend(get_layer_name(child, depth - 1, full_name,
                                                  use_filter, repeat, seq_only, init=False))
    if prefix and (not use_filter or filter_layer(module)) \
            and (repeat or depth == 0 or not isinstance(module, nn.Sequential)):
        layer_name_list.append(prefix)
    return layer_name_list


def get_all_layer(module, x,
                  layer_input = 'input', depth = 0,
                  prefix='', use_filter = True, repeat = False,
                  seq_only = True, verbose = 0):
    layer_name_list = get_layer_name(module, depth=depth, prefix=prefix, use_filter=False)
    if layer_input == 'input':
        layer_input = 'record'
    elif layer_input not in layer_name_list:
        print('Model Layer Name List: ', layer_name_list)
        print('Input layer: ', layer_input)
        raise ValueError('Layer name not in model')
    if verbose:
        print(f'{ansi["green"]}{"layer name":<50s}{"output shape":<20}{"module information"}{ansi["reset"]}')
    return _get_all_layer(module, x, layer_input, depth,
                          prefix, use_filter, repeat,
                          seq_only, verbose=verbose, init=True)


def _get_all_layer(module, x, layer_input = 'record', depth = 0,
                   prefix = '', use_filter = True, repeat = False,
                   seq_only = True, verbose = 0, init = False
                   ):
    _dict: dict[str, torch.Tensor] = {}
    if init or (not seq_only or isinstance(module, nn.Sequential)) and depth != 0:
        for name, child in module.named_children():
            full_name = prefix + ('.' if prefix else '') + name  # prefix=full_name
            if layer_input == 'record' or layer_input.startswith(f'{full_name}.'):
                sub_dict, x = _get_all_layer(child, x, layer_input, depth - 1,
                                             full_name, use_filter, repeat, seq_only, verbose)
                _dict.update(sub_dict)
                layer_input = 'record'
            elif layer_input == full_name:
                layer_input = 'record'
    else:
        x = module(x)
    if prefix and (not use_filter or filter_layer(module)) \
            and (repeat or depth == 0 or not isinstance(module, nn.Sequential)):
        _dict[prefix] = x.clone()
        if verbose:
            shape_str = str(list(x.shape))
            module_str = ''
            if verbose == 1:
                module_str = module.__class__.__name__
            elif verbose == 2:
                module_str = type(module)
            elif verbose == 3:
                module_str = str(module).split('\n')[0].removesuffix('(')
            else:
                module_str = str(module)
            print(f'{ansi["blue_light"]}{prefix:<50s}{ansi["reset"]}{ansi["yellow"]}{shape_str:<20}{ansi["reset"]}{module_str}')
    return _dict, x


def get_layer(module, x, layer_output,
              layer_input, prefix = '',
              layer_name_list = None,
              seq_only = True):
    if layer_name_list is None:
        layer_name_list = get_layer_name(module,use_filter=False, repeat=True)
        layer_name_list=layer_name_list[:-1]
    if layer_input == layer_name_list[0] and layer_output ==  layer_name_list[-1]:
        return module(x)
    if layer_input not in layer_name_list or layer_output not in layer_name_list \
            or layer_name_list.index(layer_input) > layer_name_list.index(layer_output):
        print('Model Layer Name List: \n', layer_name_list)
        print('Input  layer: ', layer_input)
        print('Output layer: ', layer_output)
        raise ValueError('Layer name not correct')
    if layer_input == layer_name_list[0]:
        layer_input = 'record'
    return _get_layer(module, x, layer_output, layer_input, prefix, seq_only, init=True)


def _get_layer(module, x, layer_output = 'output',
               layer_input= 'record', prefix = '',
               seq_only= True, init= False):
    for name, child in module.named_children():
        x = x.to(device)
        if('layer' in name):
            x = child(x)
        if(name == layer_output[:-2]):
            break
    return x


def filter_layer(module):
    if isinstance(module, transforms.Normalize) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.ReLU) or isinstance(module, nn.Sigmoid):
        return False
    elif isinstance(module,nn.Dropout):
        return False
    return True
    
def read_config():
    f = open('./config.txt', encoding="utf-8")
    content = f.read()
    #print(content)
    params = json.loads(content)
    return params

###############################################################################################################
def Fuse(model_state_dict,ihufu,iufuh):
    hufu = ihufu
    ufuh = torch.load('weight/model_decoder_param.t7', map_location='cpu')
    tobe = model_state_dict
    
    print(tobe.keys())
    
    layers_size={}
    tmp2 = torch.tensor([]).to(device)
    
    #insert_channel=[2,5,8,12,15,18,20,22,25,28,31,34,42,50]
    
    for name2 in [key for key in tobe.keys() if 'conv' in key and 'weight' in key]:
        if(tobe[name2].size()[0]>=50):
            for i in range(tobe[name2].size()[0]):
                tmp2 = torch.cat([tmp2, tobe[name2][i].view(-1)], 0)
            tmpsze=tobe[name2].size()
            layers_size[name2]=tmpsze[3]*tmpsze[1]*tmpsze[2]
    # view(-1) function£ºflatten the matrix 1st dimention first, 2nd second, 3rd third
    
    grad2 = torch.ones(tmp2.size()).to(device)
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
        for i in range(tobe[name2].size()[0]):
            end_size = start_size + layers_size[name2]
            tobe[name2][i] +=tmp2[start_size:end_size].reshape(tobe[name2][0].size())
            tobe[name2][i] /=2
            tobe[name2.replace('conv','grad')][i] = grad2[start_size:end_size].reshape(tobe[name2][0].size())
            #tobe[name2.replace('conv','grad').replace('.0.','.')][i] = grad2[start_size:end_size].reshape(tobe[name2][0].size())
            start_size = end_size
    return tobe 
    print('Fusing Done!')