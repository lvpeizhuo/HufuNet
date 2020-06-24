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
    print("cuda is available")
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
        target_np = target.cpu().detach().numpy()
        out_ys = np.argmax(output_np, axis = -1)

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
        torch.save(state, 'checkpoints/%s.t7' % checkpoint)
    return correct*1.0 / _sum
