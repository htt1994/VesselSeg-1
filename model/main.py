from __future__ import print_function
from PIL import Image
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import sys
import data_loader as dl
import densenet as dnet
import matplotlib.pyplot as plt

torch.set_printoptions(edgeitems=350)

'''
TODO:
    1. IMPLEMENT COSINE AND POLY LR DECAY
'''
loss_history = []
l_t = []
l_v = []
def smoothL1(x):
    '''
    Smooth_L1(x) = {0.5x^2     | abs(x) < 1;
                    abs(x)-0.5 | otherwise}
    '''
    return (abs(x)<1).float()*0.5*x.float()**2 + (abs(x)>=1).float()*(abs(x).float()-0.5)

def L_tot(p_set, p_ground, t_set, t_ground, cls_pred, cls_ground, phi, alpha, beta): #loss per minibatch
    '''
    Custom loss function, L_tot = 1/Ncls * sum(Lcls(p_i, p*_i)) + lambda * 1/Nreg * sum(p*_i*Lreg(t_i, t*_i))
    Where i is the index of the anchor, i ~ {0, 1, ..., N_anchors-1}

    p_set is set of segmentation masking predicted, p_i is R(a*b) ~ [0, 1]
    p_ground are the respective ground truth, p*_i is R(a*b) ~ {0, 1} for each anchor.
    t_set is set of vectors of dim(t_set(i)) = 8, normalized coordinates of proposal bbox, t_i is predicted.
    t_ground contains t*_i, the ground truth (generated from segmentation mask). {Don't need in retina experiments}
    cls_pred is the predicted class ~ {0, 1, ..., k-1}
    cls_ground is the ground truth class ~ {0, 1, ..., k-1}


    Ncls = minibatch size
    Nreg = # of anchor locations

    Lseg = Binary cross entropy loss              {Loss for segmentation}

    Lreg(t_i, t*_i) = Smooth_L1(t_i - t*_i)       {Loss for bbox, used to train the region proposal network}

    Lcls = Negative log loss                      {Loss for classification}
    TODO:
        1. Fix Lreg, check dimensionality, etc... (Don't worry too much about this until we MRI stuff though.)
    '''
    Lseg, Lreg, Lcls = (0, 0, 0)

    if phi:
        bce = nn.BCELoss() #seems to be already normalized over Ncls, avg BCEloss across minibatch.
        Lseg = bce(p_set, p_ground)

    #The following two losses do not matter for retina segmentation!

    #t_set in the form 2D tensor torch.tensor([[[t1_0, t2_0, t3_0, ..., t8_0], ..., [t1_i, t2_i, t3_i, ..., t8_i]], ... other iterations in minibatch])
    #where i = k-1, k = # of anchors.
    if alpha:
        Nreg = len(t_set[0][0]) #Number of anchors, can figure out w.r.t. variables given.
        Lreg = 1/Nreg * (p_ground * smoothL1(t_set - t_ground)) #don't think this is implmeneted correctly. Check over it

    if beta:
        logloss = nn.NLLLoss()
        Lcls = logloss(cls_pred, cls_ground) #Class prediction must also be log_softmax then. avg NLLLoss across minibatch.

    return phi*Lseg + alpha*Lreg, + beta*Lcls


#### EVALUATION METRICS
def jaccard(pred, target):
    '''
    J(A, B) = AnB/AuB
    '''
    pred = torch.round(pred) == 0
    target = target == 0

    #AnB = (torch.round(pred).long() & target.long()).view(-1).sum().float()
    #return (AnB/(torch.round(pred).long().view(-1).sum()+target.view(-1).sum().long()-AnB).float())
    AnB = (pred & target).view(-1).sum().float()
    return (AnB/(pred.long().view(-1).sum()+target.view(-1).sum().long()-AnB).float()).item()

def dice(pred, target):
    '''
    D(A, B) = 2(AnB)/(A + B)
    '''
    j = jaccard(pred, target)
    return 2*j/(j+1)


def IoU(pred, target):
    #### INVERT THE TENSORS FIRST SINCE WE INVERTED THE TARGET BIT MAP.
    pred = torch.round(pred) == 0
    target = target == 0

    truePos = (pred&target).view(-1).sum().float() #pred & target
    falsePos =  (pred & (target==0)).view(-1).sum().float() #pred & (not target)
    falseNegative = ((pred == 0) & target).view(-1).sum().float()  #(not pred) & target
    return (truePos/(truePos+falsePos+falseNegative)).item()


def restructure(pack):
    d = list(zip(*pack))
    return torch.stack(d[0]).permute(0, 3, 1, 2).float(), torch.stack(d[1]).float().unsqueeze(1)
    #We apply .unsqueeze(1) to target to turn (N,W,H) to (N, C, W, H) where C=1 (bit map), to match output/data shape.

#SHOW IMG WITH RGB VALUES BETWEEN [0, 255]
def show_img(img):
    if use_cuda:
        img = img.cpu()
    img = Image.fromarray(torch.Tensor.numpy(img.detach())[0][0], 'I')
    img.show()

#SHOW IMG WITH RGB VALUES BETWEEN [0, 11]
def show_img_mult255(img):
    if use_cuda:
        img = img.cpu()
    img = np.multiply(img, 255).astype('uint8')
    img = Image.fromarray(img, 'RGB')
    img.show()


### DATA AUGMENTATION FUNCTIONS
def brightness_change(img_arr, factor):
    return np.clip(np.multiply(img_arr, factor), 0.0, 1.0)

def contrast_change(img_arr, factor):
    return np.clip(0.5 + np.multiply((img_arr-0.5), factor), 0.0, 1.0)

def random_contrast_brightness(img_arr):
    #DRAW FROM A NORMAL DISTRIBUTION THE CHANGE FACTORS
    c_factor = np.random.normal(1, 0.15)
    b_factor = np.random.normal(1, 0.2)
    return contrast_change(brightness_change(img_arr, b_factor), c_factor)

### NETWORK TRAINING FUNCTIONS
def train(args, model, l, device, train_loader, optimizer, batch_size, epoch, print_rate=5):
    global loss_history
    global l_t

    print(len(train_loader)*batch_size)
    model.train()
    #avg_jaccard = 0
    avg_dice = 0
    train_loss = 0

    for batch_idx, pack in enumerate(train_loader):
        #print("BATCH ID: " + str(batch_idx+1))
        pack = restructure(pack)

        data, target = Variable(pack[0].to(device)), Variable(pack[1].to(device))

        #RANDOMIZE THE CONTRAST AND BRIGHTNESS OF IMGS IN THE BATCH
        for i in data:
            i = random_contrast_brightness(i)

        #OUTPUT DATA
        output = model(data)
        loss = l(output[0], target) #output[0] is segmentation prediction
        train_loss += loss
        #avg_jaccard += jaccard(torch.sigmoid(output[0]).detach(), target).item()
        avg_dice += dice(torch.sigmoid(output[0]).detach(), target).item()
       # loss_history.append(("Train", epoch, loss, avg_jaccard/(batch_idx+1), avg_dice/(batch_idx+1)))

        if epoch % 100 == 0 and batch_idx == 0:
            show_img(output[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % print_rate == 0:
        print('Train loss at epoch %s:' % epoch, str(train_loss.detach().item()/len(train_loader))
             + " Dice: " + str(avg_dice/len(train_loader)))

    l_t.append(train_loss.detach().item()/len(train_loader))

    if epoch % 200 == 0:
        show_img(output[0])
    if epoch == 1:
        show_img(target)
        del output
        del data
        del target

    #if torch.cuda.is_available():
    #    torch.cuda.empty_cache()

    #del avg_jaccard
    del avg_dice
    del train_loader

def test(args, model, l, device, test_loader, batch_size, epoch, print_rate=10):
    #if torch.cuda.is_available():
    #    torch.cuda.empty_cache()
    global l_v
    model.eval()

    test_loss = 0
    #avg_jaccard = 0
    avg_dice = 0

    with torch.no_grad():
        for batch_idx, pack in enumerate(test_loader):
            pack = restructure(pack)
            data, target = pack[0].to(device), pack[1].to(device)

            output = model(data)

            test_loss += l(output[0].detach(), target.float()).detach().item() # sum up batch loss
            #avg_jaccard += jaccard(torch.sigmoid(output[0]).detach(), target).item()#across minibatch
            avg_dice += dice(torch.sigmoid(output[0]).detach(), target).item()

            #TODO: print SAME image every
            if batch_idx == 0 and epoch%print_rate==0:
                show_img(output[0])

            del output
            del data
            del target

    #TODO: fix this avg dice and loss, make sure it prints correct numbers and is invariant to test batch size
    if epoch%print_rate==0 or epoch == 1:
        print("Test Loss is: " + str(test_loss/len(test_loader)) + " and the Avg. Dice is: " + str(avg_dice/len(test_loader)))

    l_v.append(test_loss/len(test_loader))
   # loss_history.append(("Test", epoch, test_loss, avg_jaccard/length, avg_dice/length))

    #del avg_jaccard
    del avg_dice
    del test_loss
    del test_loader

def minibatch_init(set, minibatch_size):
    #   Initializes the minibatches. Returns set,
    #   which is structured as of set = [[minibatch 0: composed of minibatch_size (data, target)], [minibatch 1:...], ..., [minibatch k-1]
    shuffle_set = list(zip(set[0], set[1])) #Randomize the groupings on the minibatches
    random.shuffle(shuffle_set)
    temp = list(zip(*shuffle_set))
    c = 0
    set = []
    minibatch = []
    for i in range(len(temp[0])): #loop through all elements in set, and assign it into a minibatch
        minibatch.append((temp[0][i], temp[1][i]))
        c += 1
        if not c%minibatch_size:
            set.append(minibatch)
            minibatch = []
    return set

def save(epoch, model, optimizer, loss, PATH, load_model):
    global loss_history
    lh = []
    if load_model: #If cumulative loss_history already exists, extend it with the current one.
        lh = torch.load(PATH)["loss_history"].extend(loss_history)
    else: #Otherwise define it as the loss_history of fprint_rate*2==0 irst training session.
        lh = loss_historyprint_rate*2==0
    torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'loss_history': lh,
            }, PATH)
    loss_history = []

def plot_graph(var_list):
    for var in var_list:
        var = np.array(var)
        plt.plot(var)
    plt.show()

def lr_decay(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=7, threshold = 0.0001, type="EXP"):
    if epoch %lr_decay_epoch:
        return optimizer
    for param_group in optimizer.param_groups:
        if param_group['lr'] <= threshold:
            return optimizery
        if type=="EXP":
            param_group['lr'] *= lr_decay
        elif type=="COS":
            return optimizer #ADD COSINE DECAY
        elif type=="POLY":
            return optimizer #ADD POLYNOMIAL DECAY
        print("Learning rate is now:print(len(train_loader)) " + str(param_group['lr']))
    return optimizer


#Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retina Segmentation and Classification with DenseNet-Based Architecture')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 1
    test_batch_size = 4
    epochs = 2000
    load_model = False
    #TODO: change folder name to match model, etc.
    model_path = "densenetpbr_t1.tar"
    e_count = 1
    train_loader = dl.loadTrain() #minibatch_init(dl.loadTrain(), batch_size)
    test_loader = minibatch_init(dl.loadTest(), test_batch_size)  #minibatch_init(dl.loadTest(), test_batch_size)
    lr = 1
    momentum = 0.9
    #TODO: try uninverted bitmap
    #loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10)).to(device) # for regular bitmap
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.1)).to(device) # for inverted bitmap

    model = dnet.DenseNetPBR(denseNetLayers=[4,4,4,4])
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    model.to(device)

    ### NUM OF PARAMETERS OF model
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("TOTAL NUM OF PARAMETERS: " + str(pytorch_total_params))

    #Load Model if true:
    if load_model:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        loss_history = checkpoint['loss_history']

   # try:
    for epoch in range(1, epochs+1):#epochs+1):
        e_count = epoch #Increase epoch count outside of loop.
        optimizer = lr_decay(optimizer, epoch, lr_decay_epoch=100)
        train(args, model, loss, device, minibatch_init(train_loader, batch_size), optimizer, batch_size, epoch)
        test(args, model, loss, device, test_loader, test_batch_size, epoch)
        if epoch % 100 == 0:
            plot_graph([l_t, l_v])
        #save(e_count, model, optimizer, l, model_path, load_model)
        #if torch.cuda.is_available():
        #    torch.cuda.empty_cache()

    if (args.save_model):
        save(e_count, model, optimizer, loss, model_path, load_model)
        print("Saved model!")
    # except:augment

     #    print("Saving the current model)
      #   save(e_count, model, optimizer, l, model_path, load_model)
       #  print("Saved the model!")
