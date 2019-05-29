import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import glob

torch.set_printoptions(edgeitems=350)
path = "/Users/Jesse/Desktop/DenseNetPBR/data/DRIVE700x605/"

class RetinaSeg(Dataset):
    def __init__(self, img_path, seg_path):
        self.img_path = img_path
        self.seg_path = seg_path
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        img = Image.open(self.img_path[index])
        mask = Image.open(self.seg_path[index])

        tensor_img = self.to_tensor(img) # RGB
        tensor_mask = self.to_tensor(img) # RGB
        print(tensor_img)
        return tensor_img, tensor_mask
        #return img, mask

    def __len__(self):
        return len(self.img_path)

def loadTrain(dataPath=path):
    img_path = glob.glob(dataPath+"training/*.png")
    seg_path = glob.glob(dataPath+"training/manual/*.png")
    return RetinaSeg(img_path, seg_path)

def loadVal(dataPath=path):
    img_path = glob.glob(dataPath+"training/val/*.png")
    seg_path = glob.glob(dataPath+"training/val/manual/*.png")
    return RetinaSeg(img_path, seg_path)

def loadTest(dataPath=path):
    img_path = glob.glob(dataPath+"test/*.png")
    seg_path = glob.glob(dataPath+"test/manual/*.png")
    return RetinaSeg(img_path, seg_path)