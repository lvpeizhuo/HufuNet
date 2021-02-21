from __future__ import print_function

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import time
import random
CUDA_VISIBLE_DEVICES=0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("cuda ia available")
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
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
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
        
def train_with_grad_control(model, trainloader, criterion, optimizer, list=['conv1_1']):
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
        
        
        for name,param in model.named_parameters():
            if 'conv' in name and 'weight' in name:
                name = name.lstrip('module.').rstrip('.weight') # 'conv1_1'
                getattr(model.module, name).weight.grad = torch.mul(getattr(model.module, name.replace('conv','grad')).weight, getattr(model.module, name).weight.grad)
                # validate the gradient don't change in some piece of one single layer
                #getattr(model.module, name).weight.grad[:10,0,0,0] = torch.zeros(getattr(model.module, name).weight.grad[:10,0,0,0].size())
        
        optimizer.step()


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

def validate(model, epoch, valloader, criterion, checkpoint=None):
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
    print('_:', _,'\n', 'correct:', correct, '_sum', _sum)
    print('accuracy: {:.4f}'.format(correct*1.0 / _sum))

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
        torch.save(state, 'checkpoints/%s.t7' % checkpoint)
        

def finetune(model, trainloader, criterion, optimizer, steps=100):
    # switch to train mode
    model.train()
    
    dataiter = iter(trainloader)
    for i in range(steps):
        try:
            input, target = dataiter.next()
            input = torch.squeeze(input, 2)
            target = torch.squeeze(target)
        except StopIteration:
            dataiter = iter(trainloader)
            input, target = dataiter.next()
            input = torch.squeeze(input, 2)
            target = torch.squeeze(target)

        input, target = input.to(device), target.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def expand_model(model, layers=torch.Tensor()):
    #for name, layer in model.named_children():
    for name, layer in model.named_parameters():
         tmp = name.split('.')
         if 'conv' not in name:
              continue 
         print(name)
         layers = torch.cat([layers.view(-1), getattr(model.module, tmp[1]).weight.view(-1)],0)
         
         '''
         if len(list(layer.children())) > 0:
             layers = expand_model(layer, layers)
         else:
             if 'mask' not in name and 'bias' not in name and 'grad' not in name: # isinstance(layer, nn.Conv2d) and 
                 #print(name)
                 layers = torch.cat((layers.view(-1), layer.weight.view(-1)))
          '''
    return layers

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

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
