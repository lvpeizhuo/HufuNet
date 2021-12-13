
import torch
from torch import nn

NUM_CLASSES = 10


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, mode = 'train', **kwargs):
        super(BasicConv2d, self).__init__()
        self.mode = mode

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.mask = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.mask.weight.data = torch.ones(self.mask.weight.size())
        
        self.grad = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.grad.weight.data = torch.ones(self.grad.weight.size())
        
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        self.conv.weight.data = torch.mul(self.conv.weight, self.mask.weight)
        x = self.conv(x)
        x = self.bn(x) 
        x = self.relu(x)
        return x

    def __prune__(self, threshold):
        self.mode = 'prune'
        self.mask.weight.data = torch.mul(torch.gt(torch.abs(self.conv.weight), threshold).float(), self.mask.weight)


class Inception(nn.Module):
    def __init__(self, in_channel, n1_1, n3x3red, n3x3, n5x5red, n5x5, pool_plane):
        super(Inception, self).__init__()
        # first line
        self.branch1x1 = BasicConv2d(in_channel, n1_1, kernel_size=1)

        # second line
        self.branch3x3_1 = BasicConv2d(in_channel, n3x3red, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(n3x3red, n3x3, kernel_size=3, padding=1)

        # third line
        self.branch5x5_1 = BasicConv2d(in_channel, n5x5red, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(n5x5red, n5x5, kernel_size=5, padding=2)

        # fourth line
        self.branch_pool_1 = nn.MaxPool2d(3, stride=1, padding=1)
        self.branch_pool_2 = BasicConv2d(in_channel, pool_plane, kernel_size=1)

    def forward(self, x):
        y1 = self.branch1x1(x)

        y2_1 = self.branch3x3_1(x)
        y2_2 = self.branch3x3_2(y2_1)

        y3_1 = self.branch5x5_1(x)
        y3_2 = self.branch5x5_2(y3_1)

        y4_1 = self.branch_pool_1(x)
        y4_2 = self.branch_pool_2(y4_1)

        output = torch.cat([y1, y2_2, y3_2, y4_2], 1)
        return output

    def __prune__(self, threshold):
        self.mode = 'prune'
        self.branch1x1.__prune__(threshold)
        self.branch3x3_1.__prune__(threshold)
        self.branch3x3_2.__prune__(threshold)
        self.branch5x5_1.__prune__(threshold)
        self.branch5x5_2.__prune__(threshold)
        self.branch_pool_2.__prune__(threshold)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(GoogLeNet, self).__init__()

        self.pre_layers = BasicConv2d(3, 192, kernel_size=3, padding=1) #

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32) #
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64) #

        self.max_pool3 = nn.MaxPool2d(3, stride=2, padding = 1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64) #
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64) #
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64) #
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64) #
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128) #


        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128) #
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128) #

        self.avg_pool = nn.AvgPool2d(8, stride = 1)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.max_pool3(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.max_pool3(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def __prune__(self, threshold):
        self.mode = 'prune'
        self.pre_layers.__prune__(threshold)
        self.a3.__prune__(threshold)
        self.b3.__prune__(threshold)

        self.a4.__prune__(threshold)
        self.b4.__prune__(threshold)
        self.c4.__prune__(threshold)
        self.d4.__prune__(threshold)
        self.e4.__prune__(threshold)

        self.a5.__prune__(threshold)
        self.b5.__prune__(threshold)

def googlenet():
    return GoogLeNet()