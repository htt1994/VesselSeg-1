from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

import DenseNet as densenet

'''
TODO:
    1. Finish up L_tot()
'''

l = nn.NLLLoss()

def smoothL1(x):
    return (abs(x)<1).float()*0.5*x.float()**2 + (abs(x)>=1).float()*(abs(x).float()-0.5)

def L_tot(p_set, t_set, lambda):
    '''
    Custom loss function, L_tot = 1/Ncls * sum(Lcls(p_i, p*_i)) + lambda * 1/Nreg * sum(p*_i*Lreg(t_i, t*_i))
    Where i is the index of the anchor, i ~ {0, 1, ..., N_anchors-1}

    p_set is set of segmentation masking predicted, p_i is R(a*b) ~ [0, 1], and respective ground truth, p*_i is R(a*b) ~ {0, 1} for each anchor.
    t_set is set of vectors of dim(t_set(i)) = 8, normalized coordinates of proposal bbox.
        - t_i is predicted, t*_i is ground truth (generated from segmentation mask)

    Ncls = minibatch size
    Nreg = # of anchor locations

    Lcls = Binary cross entropy loss

    Lreg(t_i, t*_i) = Smooth_L1(t_i - t*_i)
        Smooth_L1(x) = {0.5x^2     | abs(x) < 1;
                        abs(x)-0.5 | otherwise}

    TODO:
        1. Implement with batch training compatibility: NOTE: This should already be batch training compatibile.
    '''
    Lcls = nn.BCELoss() #seems to be already normalized over Ncls

    return Lcls + lambda #*Others...


def train(args, model, device, train_loader, optimizer, epoch):
    global l
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = l(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    global l
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += l(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retina Segmentation and Classification with DenseNet-Based Architecture')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
    dset.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
                   batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
    dset.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
                   batch_size=args.test_batch_size, shuffle=True, **kwargs)

    epochs=100
    workers = 1
    ngpu = 0
    img_w = 28
    img_h = 28

    model = densenet.DenseNet().to(device) #Change to: model = densenet.DensePBR().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epochs)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_densenet.pt")
