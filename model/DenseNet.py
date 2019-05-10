import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#path = 'C:/Users/Osvald/DenseNet-MNIST/data'
#testImages, testLabels = load.loadTest(path) # shapes (10000,28,28) and (10000,)
#trainImages, trainLabels = load.loadTrain(path) # shapes (60000,28,28) and (60000,)

class BottleneckBlock(nn.Module):
    '''
    Reduces model compexity by changing width (lxk) -> (k)
    Takes args in_planes, out_planes - number of channels into and out of block

    1x1 Conv transforms depth: in_planes -> inter_planes (4x output depth)
    3x3 Conv transforms depth: inter_planes -> out_planes

    After l layers, with growth rate k, this gives channel dims:
    (lxk) -> Bn,ReLU,Conv(1) -> (bn_sizexk) - > Bn,ReLU,Conv(3) -> (k)
    '''
    def __init__(self, in_planes, out_planes, growth_rate, bn_size=4, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = bn_size * growth_rate # number of intermediary channels in bottleneck
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes,
                               kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    '''
    Down-samples past states to match dims for concatenation
    Takes args in_planes, out_planes - number of channels into and out of block

    1x1 Conv transforms depth: in_planes -> out_planes
    2x2 Avg_Pool preserves depth -> out_planes

    After l layers, with growth rate k, this gives channel dims:
    (lxk) -> Bn,ReLU,Conv(1) -> (4xk) - > Bn,ReLU,Conv(3) -> (k)
    '''
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2) # can change pooling type and window here

class DenseBlock(nn.Sequential):
    '''
    Makes Densely Connected block with nb_layers layers
    Currently alway uses BottleNeck block TODO: add toggle for removal of bottleneck - new block implementation

    nb_lblayers bottleneck layers stacked on top of one another
    at each internal layer, 1, ko + k(l-1) channels are input and k channels are output
    k is passed in as growth_rate argument

    After this block k new higher level feature channels will be added to the "cummulative knowledge"
    providing a total of (Lxk) feature channels at the L'th denseblock
    '''
    def __init__(self, nb_layers, in_planes, growth_rate, bn_size=4, dropRate=0.0):
        super(DenseBlock, self).__init__()
        for i in range(nb_layers):
            # Each layer, l, has input ko + k(l-1) channels and outputs k channels
            layer = BottleneckBlock(in_planes = int(in_planes + i*growth_rate),
                                          out_planes = int(growth_rate),
                                          growth_rate=growth_rate,
                                          bn_size=bn_size,
                                          dropRate=float(dropRate))
            self.add_module('denselayer%d' % (i + 1), layer)

class SPP(nn.Sequential):
    def __init__(self, a, b, n, f): #a=width, b=height, n=size of pyramid layer, f= # of filters in layer being pooled from
        window = (self.ceiling(a/n), self.ceiling(b/n))
        stride = (int(a/n), int(b/n))
        self.add_module('pooling, n = ' + str(n), nn.MaxPool2d(kernel_size=window, stride=stride))

    def ceiling(self, x):
        return int(x) + (x>int(x))

class SegmentBranch(nn.Sequential):
    def __init__(self, f, hidden_layer_len_seg):
        self.add_module("conv1x1", nn.Conv2d(f, 1, kernel_size=1, stride=1))
        self.add_module("hidden layer", nn.Linear(f, hidden_layer_len_seg))
        self.add_module("output", torch.sigmoid(nn.Linear(f, 1))) #No rounding during training

class ClassifyBranch(nn.Sequential):
    def __init__(self, in_len, hidden_layer_len_cls, num_classes): #in_len = sum(k*f) across all layers of the pyramid. k = nxn, f = # of filters of feature maps pooled from.
        self.add_module("input layer to hidden", nn.Linear(in_len, hidden_layer_len_cls))
        self.add_module("hidden to output", torch.softmax(nn.Linear(hidden_layer_len_cls, num_classes)))

class DenseNet(nn.Module):
    def __init__(self, layers=[4,4], growth_rate=8, reduction=0.5, dropRate=0.0):
        super(DenseNet, self).__init__()
        self.in_planes = 2 * growth_rate
        self.n = layers
        # input Conv
        self.conv1 = nn.Conv2d(1, self.in_planes, kernel_size=3, stride=2,
                               padding=1, bias=False)

        # 1st block
        self.block1 = DenseBlock(self.n[0], self.in_planes, growth_rate, dropRate=dropRate)
        self.in_planes = int(self.in_planes+self.n[0]*growth_rate)
        #transition
        self.trans1 = TransitionBlock(self.in_planes, int(math.floor(self.in_planes*reduction)), dropRate=dropRate)
        self.in_planes = int(math.floor(self.in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(self.n[1], self.in_planes, growth_rate, dropRate=dropRate)
        self.in_planes = int(self.in_planes+self.n[1]*growth_rate)

        #initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.block2(out)
        return out

class DensePBR(nn.Module):
    def __init__(self, f=512, hidden_layer_len_cls=144, hidden_layer_len_seg=64, num_classes=2, n_layers=[1,4,16]):
        super(DensePBR, self).__init__()
        self.sum = 0
        self.a = 64 #Final width of feature map
        self.b = 64 #Final length of feature map, change code so it adjusts dynamically

        self.convolute = DenseNet()
        self.segment = SegmentBranch(f, hidden_layer_len_seg)
        self.classify = ClassifyBranch(self.sum, hidden_layer_len_cls, num_classes)

        self.l = n_layers
        self.f = f #Make it so we don't have to specify filter length, should be out[i].size() or something like that
        self.pools = []

        for i in self.l:
            pools.append(SPP(self.a, self.b, i, f))

    def forward(self, input):
        out = self.convolute.forward(input)

        #classify branch
        cls_in = []
        for pool in self.pools:
            cls_in.extend(pool(out).resize_(self.l[self.pools.index(i)]*self.f)) #change dimension to vertical
        self.classify(torch.tensor(cls_in)) #classification output

        #segment branch
        self.round(self.segment(out))

        #Don't know what to return since we have two branches

    def round(self, x):
        return (x >= 0.5 and x <= 1.0)
