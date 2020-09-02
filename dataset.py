import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import torchvision as tv
from utils.utils import *

from torch.utils.data import Dataset

labels = {'cats': 0, 'dogs': 1, 'horses': 2, 'Humans': 3}

class DataReader(Dataset):
    def __init__(self, img_folder, img_size):
        self.img_size= img_size
        self.img_list = sorted(glob.glob(r'data/*/*.*'))
        self.data_len = len(self.img_list)
        self.data_tranform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])

    def __getitem__(self, item):
        # Avoid out of range
        img_path = self.img_list[item % self.data_len]
        # get label number
        files = img_path.split('/')
        label = files[1]
        pil_img = Image.open(img_path)
        img = self.data_tranform(pil_img)
        img_ = pad_to_square(img, 0)
        img_ = resize(img_, self.img_size)
        # one_hot_labels = F.one_hot(labels[label], num_classes=4)
        return img_path, img_, labels[label]

    def __len__(self):
        return self.data_len


def data_loader(folder, img_size, batch_size, num_workers):
    data_reader = DataReader(folder, img_size)
    dataset_loader = DataLoader(data_reader, batch_size, shuffle = True, num_workers = num_workers)
    return dataset_loader


if __name__ == '__main__':
    x1 = torch.ones((2,2))
    x2 = torch.ones((2,2))
    print(x1 + x2)

    a = 38
    b = 36
    print(a%b)
    folder = r'E:\\dataset\\data\\'
    train_dataset_ = data_loader(folder, 224, 32, 4)

    list = glob.glob(r'data/*/*.*')
    for path in list:
        files = path.split('\\')
        labels = files[-1].split(r'/')


    

