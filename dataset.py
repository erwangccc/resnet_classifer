import os
import glob
import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import DataLoader
from utils.utils import *
from torch.utils.data import Dataset


labels = {'cats': 0, 'dogs': 1, 'horses': 2, 'Humans': 3}

class DataReader(Dataset):
    def __init__(self, img_folder, img_size):
        self.img_size= img_size
        self.img_list = sorted(glob.glob(r'{}/*/*.*'.format(img_folder)))
        self.data_len = len(self.img_list)
        self.data_tranform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])

    def __getitem__(self, item):
        # Avoid out of range
        img_path = self.img_list[item % self.data_len]
        # get label number
        files = img_path.split('/')
        label = files[2]
        pil_img = Image.open(img_path).convert('RGB')
        img = self.data_tranform(pil_img)
        img_ = pad_to_square(img, 0)
        img_ = resize(img_, self.img_size)
        # one_hot_labels = F.one_hot(labels[label], num_classes=4)
        return img_, labels[label]

    def __len__(self):
        return self.data_len


def get_dataset(default_path, img_size, batch_size, num_workers, data_name='custom'):
    if data_name == 'custom':
        data_reader = DataReader(default_path, img_size)
        train_loader = DataLoader(data_reader, batch_size, shuffle=True, num_workers=num_workers)
        test_loader = train_loader
    elif data_name == 'cifar10':
        transform = transforms.Compose(
            [transforms.Resize(img_size),
             # transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root=default_path, train=True,
                                                download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.CIFAR10(root=default_path, train=False,
                                               download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)
    else:
        raise ValueError('Invalid data set name!!!')

    return train_loader, test_loader


if __name__ == '__main__':
    x1 = torch.ones((2,2))
    x2 = torch.ones((2,2))
    print(x1 + x2)


    

