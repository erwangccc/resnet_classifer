import argparse
import torchvision
from dataset import *
import torch.optim as optim
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter
from network import *
from resnet import ResNet, BasicBlock
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn as nn


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoches", type=int, default=200, help= "number of epochs")
    parser.add_argument("--num_classes", type=int, default=4, help= "number of classified classes")
    parser.add_argument("--input_size", type=int, default=224, help= "input size of network")
    parser.add_argument("--batch_size", type=int, default=32, help= "training batch sizes")
    parser.add_argument("--lr", type=int, default=0.001, help= "learning rate of training")
    parser.add_argument("--n_cpu", type=int, default=16, help= "cpu number used when training")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help= "interval between saving model weights")
    parser.add_argument("--pretrained_weights", type=str, default=None, help= "path of pretrained weights")
    parser.add_argument("--log_path", type=str, default= "logs/resnet_classifier", help= "path of saving logs")
    parser.add_argument("--weights_path", type=str, default="./weights", help= "path of saving weights")
    parser.add_argument("--train_path", type=str, default="data/custom/", help= "cpu number used when training")
    parser.add_argument("--data_name", type=str, default="custom", help= "the data name we want to load")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(args.log_path, 'Training process of cars detection via ResNet18!')
    CEloss = nn.CrossEntropyLoss()
    # resnet = ResNet(BasicBlock, [2, 2, 2, 2], args.num_classes)
    resnet = ResNet18(Basicblock, [2, 2, 2, 2], args.input_size, 3, args.num_classes)
    if args.pretrained_weights:
        resnet.load_state_dict(torch.load(args.pretrained_weights))
    else:
        resnet.apply(weights_init)
    optimizer = optim.SGD(resnet.parameters(), lr=args.lr, weight_decay=0.9999)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    # load train dataset
    train_loader, test_loader = get_dataset(args.train_path, args.input_size,
                                args.batch_size, args.n_cpu, args.data_name)

    # write model structure to logs
    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # write to tensorboard
    writer.add_image('Data we train', img_grid)
    with writer as w:
        w.add_graph(resnet, images)

    resnet = resnet.to(device)


    for epoch in range(1, args.epoches + 1):
        resnet.train()

        train_loss = 0
        progressbar = tqdm(train_loader)

        #######
        # train
        #######
        for batch_idx, (imgs, targets) in enumerate(progressbar):
            imgs = imgs.to(device)
            targets = targets.to(device)
            # optimizer set to zero
            optimizer.zero_grad()
            outputs = resnet(imgs)
            # print(output)
            # print(one_hot_targets)
            loss = CEloss(outputs, targets)
            loss.backward()  # loss value backward
            optimizer.step()  # update parameters at optimizer
            train_loss += loss.item()
            progressbar.set_description('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(imgs), 100. * (batch_idx+1) / len(train_loader),
                       loss.item()))
        scheduler.step()
        writer.add_scalar('loss/{}-train'.format(args.data_name), train_loss/len(train_loader.dataset), epoch)

        #######
        # test
        #######
        progressbar1 = tqdm(test_loader)
        test_accuracy = 0
        for batch_idx, (imgs, targets) in enumerate(progressbar1):
            imgs = imgs.to(device)
            targets = targets.to(device)
            # optimizer set to zero
            output = resnet(imgs)
            for i in range(len(imgs)):
                if output[i].argmax().item() == targets[i].item():
                    test_accuracy += 1
            progressbar1.set_description('Test Epoch: {} Acc: {:.0f}%'.format(
                epoch, 100. * test_accuracy / len(test_loader.dataset)))
        writer.add_scalar('Acc/{}-test'.format(args.data_name), test_accuracy / len(test_loader.dataset), epoch)

        if epoch%args.checkpoint_interval == 0:
            torch.save(resnet.state_dict(), args.weights_path+'/{}/resnet18_{}_{}.pth'.format(args.data_name,
                                                                                             args.data_name, epoch))

    writer.close()





