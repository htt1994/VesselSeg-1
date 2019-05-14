import numpy as np
import torch
import os
from PIL import Image


path = '/Users/Jesse/Desktop/DenseNetPBR/data/DRIVE700x605'

def normalize(x):
    return (x.astype(float)-128)/128

def img_to_bitmap(x):
    return x/255

def loadData(folder):
    data = []
    labels = []

    for imagePath in os.listdir(folder):
        if imagePath.endswith(".png"):
            image = Image.open(folder+'/'+imagePath)
            np_img = np.array(image)
            data.append(torch.tensor(np_img))

    for seg in os.listdir(folder+"/manual"):
        if seg.endswith(".png"):
            image = Image.open(folder+'/manual/'+seg)
            np_seg = np.array(image)
            labels.append(torch.tensor(img_to_bitmap(np_seg)))

    return data, labels, dict(zip(data, labels))

def loadTrain(dataPath=path):
    return loadData(dataPath + "/training" )

def loadTest(dataPath=path):
    return loadData(dataPath + "/test")
