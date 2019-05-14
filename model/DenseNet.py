import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BottleneckBlock(nn.Module):
    '''
    Reduces model compexity by changing width (lxk) -> (k)
    Takes args channels, out_channels - number of channels into and out of block

    1x1 Conv transforms depth: channels -> inter_channels (4x output depth)
    3x3 Conv transforms depth: inter_channels -> out_channels

    After l layers, with growth rate k, this gives channel dims:
    (lxk) -> Bn,ReLU,Conv(1) -> (bn_sizexk) - > Bn,ReLU,Conv(3) -> (k)
    '''
    def __init__(self, channels, out_channels, growth_rate, bn_size=4, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_channels = bn_size * growth_rate # number of intermediary channels in bottleneck
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, inter_channels,
                               kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, out_channels,
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
    Takes args channels, out_channels - number of channels into and out of block

    1x1 Conv transforms depth: channels -> out_channels
    2x2 Avg_Pool preserves depth -> out_channels

    After l layers, with growth rate k, this gives channel dims:
    (lxk) -> Bn,ReLU,Conv(1) -> (4xk) - > Bn,ReLU,Conv(3) -> (k)
    '''
    def __init__(self, channels, out_channels, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, out_channels, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, inplace=False, training=self.training)
        return F.avg_pool2d(out, stride=2) # can change pooling type and window here

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
    def __init__(self, nb_layers, channels, growth_rate, bn_size=4, dropRate=0.0):
        super(DenseBlock, self).__init__()
        for i in range(nb_layers):
            # Each layer, l, has input ko + k(l-1) channels and outputs k channels
            layer = BottleneckBlock(channels = int(channels + i*growth_rate),
                                          out_channels = int(growth_rate),
                                          growth_rate=growth_rate,
                                          bn_size=bn_size,
                                          dropRate=float(dropRate))
            self.add_module('denselayer%d' % (i + 1), layer)

class SPP(nn.Sequential):
    '''
    An SPP level class. Have multiple objects of these to create an SPP pyramid.
    a = width
    b = height
    n = side size of post-pool layer. n*n is number of bins for one feature map of layer n.
    '''
    def __init__(self, a, b, n):
        window = (self.ceiling(a/n), self.ceiling(b/n))
        stride = (int(a/n), int(b/n))
        self.add_module('pooling, n = ' + str(n), nn.MaxPool2d(kernel_size=window, stride=stride))

    def ceiling(self, x):
        return int(x) + (x>int(x))

class SegmentBranch(nn.Sequential):
    '''
    Segmentation FCN
    For pixel x_f(i,j) on each feature map, where f is feature map number
     are inputs to FCN with a hidden layer and one output node.
    '''
    def __init__(self, f, hidden_seg):
        self.add_module("conv1x1", nn.Conv2d(f, hidden_seg, kernel_size=1, stride=1))
        self.add_module("conv1x1_hidden", torch.sigmoid(nn.Conv2d(hidden_seg, 1, kernel_size=1, stride=1))) #apply sigmoid during training.

class ClassifyBranch(nn.Sequential):
    '''
    Classification branch that follows the SPP. SPP vector is input to this branch.
    in_channels = sum(k*f) across all layers of the final conv block. k = nxn, f = # of filters of feature maps pooled from.
    '''
    def __init__(self, in_channels, hidden_cls, num_classes):
        self.add_module("input layer to hidden", nn.Linear(in_channels, hidden_cls))
        self.add_module("hidden to output", torch.softmax(nn.Linear(hidden_cls, num_classes)))

class DenseNet(nn.Sequential):
    '''
    Vanilla DenseNet backbone as nn.Sequential.
    
    '''
    def __init__(self, layers=[4,4], growth_rate=8, reduction=0.5, dropRate=0.0):
        super(DenseNet, self).__init__()
        self.channels = 2 * growth_rate # first conv gives 2k channels
        self.n = layers
        # input Conv
        self.add_module("Conv 1", nn.Conv2d(1, self.channels, kernel_size=3, stride=2, padding=1, bias=False))

        for i in range(len(self.n)):
            if i < (len(self.n)-1):
                self.add_module("Block " + str(i+1), DenseBlock(self.n[i], self.channels, growth_rate=growth_rate, dropRate=dropRate))
                self.channels = int(self.channels*self.n[i]*growth_rate)

                #Transition
                self.add_module("Transition " + str(i+1), TransitionBlock.forward(self.channels, int(math.floor(self.channels*reduction)),
                                dropRate=dropRate))
                self.channels = int(math.floor(self.channels*reduction))

            else:
                self.add_module("Block " + str(i+1), DenseBlock(self.n[i], self.channels, growth_rate=growth_rate, dropRate=dropRate))
                self.channels = int(self.channels+self.n[i]*growth_rate)

        #initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

class DensePBR(nn.Module):
    '''
    Complete architecture of the framework we have proposed.
    DenseNet used for extracting feature maps.
    Feature maps are passed in parallel to a segmentation branch and classification branch:

    - Segmentation branch runs through 1x1 convolutions to binary classify each pixel as object of interest or not. May extend to having multiple filters to
    create multiple feature maps, each having a binary segmentation pixelwise encoding for a corresponding class. Sigmoid applied during training.

    - Classification branch is a sequence of spatial pyramid pooling, yields fixed output size which is passed into FCs for disease classification.

    '''
    def __init__(self, denseNetLayers=[4,4], hidden_cls=144, hidden_seg=64, num_classes=2, spp_layers=[1,2,4], segment=True, classify = True):
        super(DensePBR, self).__init__()
        self.seg = segment
        self.cls = classify
        self.pools = []
        self.pool_init = False
        self.map_dims = (0,0) #y,x
        self.prob_out = True

        # DenseNet
        self.conv = DenseNet(layers=denseNetLayers)
        self.channels = self.conv.channels # retrieve channel dim from DenseNet class

        # Segment Branch
        self.segment = SegmentBranch(f=self.channels, hidden_seg=hidden_seg)

        # Classification Branch
        self.spp_layers = spp_layers
        self.bins = sum((n**2) * self.channels for n in spp_layers) # total pooling bins * number of channels in final ConvBlock
        self.classify = ClassifyBranch(in_channels=self.bins, hidden_cls=hidden_cls, num_classes=num_classes)

    def SPP_init(self, convBlock):
        self.pool_init = True
        self.map_dims = (convBlock.shape[2], convBlock.shape[3]) # (batch, C, y, x)
        for i in self.spp_layers:
            self.pools = []
            self.pools.append(SPP(self.map_dims[2], self.map_dims[3], i))

    def forward(self, input):
        out = self.conv(input)

        seg_out = None
        cls_out = None

        # Classification Branch
        if self.cls:
            if pool_init == False or (out.shape[2], out.shape[3])!=self.map_dims: #If we haven't initialized the maxpooling pyramid yet, or the feature map changed shape.
                self.SPP_init(out) # creates SPP layers based on specific image dims
            cls_in = []
            for pool in self.pools: # n*n dimensions
                cls_in.extend(pool(out).view(-1)) # flatten for input into FC layer
            cls_out = self.classify(torch.tensor(cls_in).resize_(len(cls_in), 1)) #classification output, resize for vertical.

        # Segment Branch
        if self.seg:
            seg_out = self.segment(out) if self.prob_out else torch.round(self.segment(out))

        return seg_out, cls_out

    def toggleProbOut(self):
        self.prob_out = not self.prob_out
        print("Outputting segmentation pixelwise probability" if self.prob_out else "Outputting segmentation binary mask")
