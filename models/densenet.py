import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
Date: May, 2019

DenseNetPBR: DenseNet Parallel Branch Retina Experiments
Implementation of DenseNet/CondenseNet variation for parallel vessel segmentation and
classification of diatbetic retionpathy detection/diagnosis. Has two parallel branches:
one for segmentation, and one for classification, both share the same convolutional
feature map yielded by the DenseNet backbone.

Developed by:
    Jesse Sunc
    Osvald Nitski

Dr. Bo Wang's AI Lab, University Health Network.
'''

class BottleneckBlock(nn.Module):
    '''
    Reduces model compexity by changing width (lxk) -> (k)
    Takes args channels, out_channels - number of channels into and out of block

    1x1 Conv transforms depth: channels -> inter_channels (4x output depth)
    3x3 Conv transforms depth: inter_channels -> out_channels

    After l layers, with growth rate k, this gives channel dims:
    (lxk) -> Bn,ReLU,Conv(1) -> (bn_sizexk) - > Bn,ReLU,Conv(3) -> (k)
    '''
    def __init__(self, channels, out_channels, growth_rate, bn_size=4, dropRate=0):
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

class TransitionBlock(nn.Sequential):
    '''
    Down-samples past states to match dims for concatenation
    Takes args channels, out_channels - number of channels into and out of block

    1x1 Conv transforms depth: channels -> out_channels
    2x2 Avg_Pool preserves depth -> out_channels

    After l layers, with growth rate k, this gives channel dims:
    (lxk) -> Bn,ReLU,Conv(1) -> (4xk) - > Bn,ReLU,Conv(3) -> (k)
    '''
    def __init__(self, channels, out_channels, dropRate=0):
        super(TransitionBlock, self).__init__()
        self.dropRate = dropRate
        self.add_module("BN 1", nn.BatchNorm2d(channels))
        self.add_module("ReLU", nn.ReLU(inplace=True))
        self.add_module("Conv 1", nn.Conv2d(channels, out_channels, kernel_size=1,
                                            stride=1, padding=0, bias=False))
        if self.dropRate>0:
            self.add_module("Dropout", nn.Dropout(self.dropRate))

        self.add_module("Max pool", nn.MaxPool2d(kernel_size=2))

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
    def __init__(self, nb_layers, channels, growth_rate, bn_size=4, dropRate=0):
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
        super(SPP, self).__init__()
        window = (self.ceiling(a/n), self.ceiling(b/n))
        stride = (int(a/n), int(b/n))
        self.add_module('pooling, n = ' + str(n), nn.MaxPool2d(kernel_size=window, stride=stride))

    def ceiling(self, x):
        return int(x) + (x>int(x))

class SegmentBranch(nn.Module):
    '''
    Segmentation FCN
    For pixel x_f(i,j) on each feature map, where f is feature map number
     are inputs to FCN with a hidden layer and one output node.
    '''
    def __init__(self, f, hidden_seg):
        super(SegmentBranch, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(f, hidden_seg, kernel_size=1, stride=1, padding=0)
        self.hidden = nn.Conv2d(hidden_seg, 1, kernel_size=1, stride=1, padding=0)
    def forward(self, input):
        return self.hidden(self.relu(self.conv1x1(input)))

class ClassifyBranch(nn.Module):
    '''
    Classification branch that follows the SPP. SPP vector is input to this branch.
    in_channels = sum(k*f) across all layers of the final conv block. k = nxn, f = # of filters of feature maps pooled from.
    '''
    def __init__(self, in_channels, hidden_cls, num_classes):
        super(ClassifyBranch, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_cls)
        self.lin2 = nn.Linear(hidden_cls, num_classes)
    def forward(self, input):
        return torch.softmax(self.lin2(self.lin1(input)))

class DenseNet(nn.Module):
    '''
    DenseNet variant implementation: CondenseNet
    Each DenseBlock output is also concatenated with priors before feeding into the following TransitionBlock.
    '''
    def __init__(self, layers=[4,4,4,4], growth_rate=8, reduction=0.5, dropRate=0):
        super(DenseNet, self).__init__()
        self.channels = 2 * growth_rate
        self.n = layers
        self.instantiated = False

        self.bn0 = nn.BatchNorm2d(3) #Normalizes input w.r.t. minibatch
        self.conv1 = nn.Conv2d(3, self.channels, kernel_size=3, stride=2,
                               padding=1, bias=False)

        self.seq = nn.ModuleList()

        self.outputs = []

        self.in_trans_chs = [] #even numbers: {c0, c2, c4, ...}
        self.in_block_chs = [] #odd numbers:  {c1, c3, c5, ...}

        #if len(self.in_trans_chs) == 0:
        #    self.in_trans_chs.append(3)
        if len(self.in_block_chs) == 0:
            self.in_block_chs.append(self.channels)

        '''
        Add the blocks and transitions to the queue, depending on channels in and out which we use
        self.in_trans_chs and self.in_block_chs to help us with.
        '''
        for i in range(len(self.n)):
            self.seq.append(DenseBlock(self.n[i], self.in_block_chs[i], growth_rate=growth_rate, dropRate=dropRate))
            self.channels = int(self.in_block_chs[i]+self.n[i]*growth_rate)
            if not self.instantiated:
                self.in_trans_chs.append(self.channels)

            if i != len(self.n)-1:
                in_chs = sum(self.in_trans_chs)
                #print(in_chs)
                self.seq.append(TransitionBlock(in_chs, int(math.floor(in_chs*reduction)), dropRate=dropRate))
                self.channels = int(math.floor(in_chs*reduction))
                if not self.instantiated:
                    self.in_block_chs.append(self.channels)

        self.condense_ch_final = sum(self.in_trans_chs) #Final number of channels of feature map.
        self.instantiated = True
        #print(self.in_trans_chs)
        #print(self.in_block_chs)

        #initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        '''
        We permute because tensor is of shape (N, C, H, W), and we can only concatenate along dim 0, but we want to concat along the channels (dim 1) (with every dim != 0 equal between tensors).
        Hence, permute(1, 0, 2, 3) to get (C, N, H, W), concatenate as N, H, W are constant between tensors, to get (Cnew, N, H, W), and then permute(1, 0, 2, 3) again to get (N, Cnew, H, W).
        '''
        '''
        We also got rid of initial BN to save memory consumption, may re-add later to see how it may improve the model.
        '''
        #self.outputs.append(input.permute(1,0,2,3))
        in_dims = (input.shape[2], input.shape[3]) #To keep track of what to upsample the low resolution feature maps to.

        out = self.bn0(input)
        #self.outputs.append(out.permute(1,0,2,3)) #Normalize first, before propogating it into following transition blocks.
        out = self.conv1(out)

        for i, func in enumerate(self.seq):
            if i%2 == 0: #if it is a denseblock transformation
                out = func(out)
                self.outputs.append(F.interpolate(out, size=in_dims, mode="nearest").permute(1,0,2,3)) #Upsample the output to the correct input dims so we can concat before transition block.
            else:
                out = func(torch.cat(self.outputs).permute(1,0,2,3))
        val = torch.cat(self.outputs).permute(1,0,2,3) #Restore to (N, C, H, W)
        self.outputs = []
        return val


class DensePBR(nn.Module):
    '''
    Complete architecture of the framework we have proposed.
    DenseNet used for extracting feature maps.
    Feature maps are passed in parallel to a segmentation branch and classification branch:

    - Segmentation branch runs through 1x1 convolutions to binary classify each pixel as object of interest or not. May extend to having multiple filters to
    create multiple feature maps, each having a binary segmentation pixelwise encoding for a corresponding class. Sigmoid applied during training.

    - Classification branch is a sequence of spatial pyramid pooling, yields fixed output size which is passed into FCs for disease classification.

    '''
    def __init__(self, denseNetLayers=[4,4,4,4], hidden_cls=144, hidden_seg=64, num_classes=2, spp_layers=[1,2,4], segment=True, classify = False):
        super(DensePBR, self).__init__()
        self.seg = segment
        self.cls = classify
        self.pools = []
        self.pool_init = False
        self.map_dims = (0,0) #y,x
        self.prob_out = True

        # DenseNet
        self.conv = DenseNet(layers=denseNetLayers)
        self.channels = self.conv.condense_ch_final # retrieve channel dim from DenseNet class
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
        out = self.conv.forward(input)

        seg_out = None
        cls_out = None

        # Classification Branch
        if self.cls:
            if pool_init == False or (out.shape[2], out.shape[3])!=self.map_dims: #If we haven't initialized the maxpooling pyramid yet, or the feature map changed shape.
                self.SPP_init(out) # creates SPP layers based on specific image dims
            cls_in = []
            for pool in self.pools: # n*n dimensions
                cls_in.extend(pool(out).view(-1)) # flatten for input into FC layer
            cls_out = torch.log_softmax(self.classify.forward(torch.tensor(cls_in).resize_(len(cls_in), 1))) #classification output, resize for vertical.

        # Segment Branch
        if self.seg:
            seg_out = self.segment.forward(out) if self.prob_out else torch.round(self.segment.forward(out))

        return seg_out, cls_out

    def toggleProbOut(self):
        self.prob_out = not self.prob_out
        print("Outputting segmentation pixelwise probability" if self.prob_out else "Outputting segmentation binary mask")


'''
Old Code Storage Room:

class DenseNet(nn.Sequential):
    Vanilla DenseNet backbone as nn.Sequential.

    def __init__(self, layers=[4,4], growth_rate=8, reduction=0.5, dropRate=0):
        super(DenseNet, self).__init__()
        self.channels = 2 * growth_rate # first conv gives 2k channels
        self.n = layers

        # input transition
        print("Hello1")
        self.add_module("BN 1", nn.BatchNorm2d(3)) #Normalizes input w.r.t. minibatch
        print("Hello2")
        self.add_module("Conv 1", nn.Conv2d(3, self.channels, kernel_size=3, stride=2, padding=1, bias=False))

        for i in range(len(self.n)):
            if i < (len(self.n)-1):
                self.add_module("Block " + str(i+1), DenseBlock(self.n[i], self.channels, growth_rate=growth_rate, dropRate=dropRate))
                self.channels = int(self.channels*self.n[i]*growth_rate)
                #Transition
                self.add_module("Transition " + str(i+1), TransitionBlock(channels=self.channels, out_channels=int(math.floor(self.channels*reduction)),
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
'''
