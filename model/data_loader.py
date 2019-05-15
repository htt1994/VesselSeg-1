import numpy as np
import torch
import os
import random
from PIL import Image

#CHANGE THIS ACCORDINGLY TO YOUR OWN FILE LOCATION
path = '/Users/Jesse/Desktop/DenseNetPBR/data/DRIVE700x605'

def normalize(x, norm): #0 = -128, divide 128. 1 = /255
    return (x.astype(float)-128)/128  if norm == 0 else x.astype(float)/255

def img_to_bitmap(x):
    return x/255

def loadData(folder):
    data = []
    labels = []

    for imagePath in os.listdir(folder):
        if imagePath.endswith(".png"):
            image = Image.open(folder+'/'+imagePath)
            np_img = np.array(image)
            data.append(torch.tensor(normalize(np_img, 1)))

    for seg in os.listdir(folder+"/manual"):
        if seg.endswith(".png"):
            image = Image.open(folder+'/manual/'+seg)
            np_seg = np.array(image)
            labels.append(torch.tensor(img_to_bitmap(np_seg)))

    return data, labels #, dict(zip(data, labels))

def loadTrain(dataPath=path):
    return loadData(dataPath + "/training" )

def loadTest(dataPath=path):
    return loadData(dataPath + "/test")
