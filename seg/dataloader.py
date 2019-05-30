import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import glob

torch.set_printoptions(edgeitems=350)
path = "/Users/Jesse/Desktop/DenseNetPBR/data/DRIVE700x605/"
#path = "/home/jessesun/Desktop/DenseNetPBR/data/DRIVE700x605/"
class RetinaSeg(Dataset):
    def __init__(self, img_path, seg_path):
        self.img_path = img_path
        self.seg_path = seg_path
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        img = Image.open(self.img_path[index])
        mask = Image.open(self.seg_path[index])

        tensor_img = np.array(img) / 255.0 # RGB
        tensor_mask = np.array(mask) / 255.0 # RGB

        output = dict()
        output['img_data'] = torch.from_numpy(tensor_img).float().permute(2, 0, 1)
        output['seg_label'] = torch.from_numpy(tensor_mask).long().unsqueeze(0)

        print(output['seg_label'].shape)
        #return tensor_img, tensor_mask
        return output

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
