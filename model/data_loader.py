import numpy as np
import torch
import os
import random
from PIL import Image

#CHANGE THIS ACCORDINGLY TO YOUR OWN FILE LOCATION
path = '/Users/Jesse/Desktop/DenseNetPBR/data/DRIVE700x605'
#path = '/home/wanglab/Osvald/Imaging/DenseNetPBR/data/DRIVE700x605'

def normalize(x, norm): #0 = -128, divide 128. 1 = /255
    return (x.astype(float)-128)/128  if norm == 0 else x.astype(float)/255

def img_to_bitmap(x):
    return x/255

def transform(data, labels):
    methods = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_180, Image.TRANSPOSE]

    x = random.randrange(0, len(methods))
    #data.show()
    data.transpose(methods[x])
    return data.transpose(methods[x]), labels.transpose(methods[x])

def loadData(folder, multiplier=3): #TOTAL NUMBER OF DATA PER EPOCH = Multiplier*OriginalLength.
    data = []
    labels = []

    data_img = []
    seg_img = []

    for imagePath in os.listdir(folder):
        if imagePath.endswith(".png"):
            image = Image.open(folder+'/'+imagePath)
            np_img = np.array(image)
            data.append(torch.tensor(normalize(np_img, 1)))
            data_img.append(image)

    for seg in os.listdir(folder+"/manual"):
        if seg.endswith(".png"):
            image = Image.open(folder+'/manual/'+seg)
            np_seg = (np.array(image) == 0).astype('long') * 255
            #np_seg = ((np.array(image) == 0) == 0).astype(long) * 255
            #np_seg = np.ones(np_seg.shape) - np_seg #TODO: remove
            labels.append(torch.tensor(img_to_bitmap(np_seg)))
            seg_img.append(image)

    for x in range(multiplier-1):
        for i in range(len(data_img)):
            img_d, img_s = transform(data_img[i], seg_img[i])
            np_img_d, np_img_s = np.array(img_d), np.array(img_s)
            data.append(torch.tensor(normalize(np_img_d, 1)))
            labels.append(torch.tensor(img_to_bitmap(np_img_s)))

    del data_img
    del seg_img

    return data, labels

def loadTrain(dataPath=path):
    return loadData(dataPath + "/training" )

def loadTest(dataPath=path):
    return loadData(dataPath + "/test")
