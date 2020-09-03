import argparse
import torchvision
from dataset import *
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from network import *
# from resnet import ResNet, BasicBlock
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn as nn


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default = 4, help= "number of classified classes")
    parser.add_argument("--input_size", type=int, default = 224, help= "input size of network")
    parser.add_argument("--batch_size", type=int, default = 1, help= "training batch sizes")
    parser.add_argument("--n_cpu", type=int, default = 8, help= "cpu number used when training")
    parser.add_argument("--log_path", type=str, default= "logs/resnet_classifier", help= "path of saving logs")
    parser.add_argument("--weights_path", type=str, default= "weights/custom/resnet18_custom_200.pth", help= "path of saving weights")
    parser.add_argument("--test_path", type=str, default="data/custom/", help= "cpu number used when training")
    parser.add_argument("--data_name", type=str, default="custom", help="the data name we want to load")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # resnet = ResNet(BasicBlock, [2, 2, 2, 2], 4)
    resnet = ResNet18(Basicblock, [2, 2, 2, 2], args.input_size, 3, args.num_classes)
    resnet.load_state_dict(torch.load(args.weights_path))

    # load train dataset
    train_loader, test_loader = get_dataset(args.test_path, args.input_size,
                                            args.batch_size, args.n_cpu)

    resnet = resnet.to(device)

    progressbar = tqdm(test_loader)
    test_accuracy = 0
    resnet.eval()  # set evaluation mode
    for batch_idx, (imgs, targets) in enumerate(progressbar):
        imgs = imgs.to(device)
        targets = targets.to(device)
        # optimizer set to zero
        output = resnet(imgs)
        res = output.argmax()
        if res.item() == targets.item():
            test_accuracy+=1

    print('Acc/{}-{}-{}'.format(args.data_name, test_accuracy, test_accuracy / len(test_loader.dataset)))






