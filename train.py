import argparse
from dataset import *
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from network import *
from tqdm import tqdm


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoches", type=int, default = 100, help= "number of epochs")
    parser.add_argument("--num_classes", type=int, default = 4, help= "number of classified classes")
    parser.add_argument("--input_size", type=int, default = 224, help= "input size of network")
    parser.add_argument("--batch_size", type=int, default = 32, help= "training batch sizes")
    parser.add_argument("--lr", type=int, default = 0.001, help= "learning rate of training")
    parser.add_argument("--n_cpu", type=int, default = 8, help= "cpu number used when training")
    parser.add_argument("--checkpoint_interval", type=int, default = 10, help= "interval between saving model weights")
    parser.add_argument("--pretrained_weights", type=str, default = None, help= "path of pretrained weights")
    parser.add_argument("--log_path", type=str, default= "logs", help= "path of saving logs")
    parser.add_argument("--weights_path", type=str, default= "./", help= "path of saving weights")
    parser.add_argument("--train_path", type=str, default = "E://dataset//data//", help= "cpu number used when training")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(args.log_path, 'Training process of cars detection via ResNet18!')
    CEloss = nn.CrossEntropyLoss()
    resnet = ResNet18(Basicblock, [2, 2, 2, 2], 224, 3, 4)
    resnet = resnet.to(device)
    resnet.apply(weights_init)
    optimizer = optim.Adam(resnet.parameters(), lr=args.lr, weight_decay = 0.9999)
    # load train dataset
    train_dataset = data_loader(args.train_path, args.input_size,
                                args.batch_size, args.n_cpu)

    # write model structure to logs
    # dummy_input = torch.randn(1, 3, 224, 224, requires_grad = True)
    # grid = torchvision.utils.make_grid(dummy_input)
    # writer.add_graph(resnet, grid)
    for epoch in range(1, args.epoches + 1):
        resnet.train()
        train_loss = 0
        progressbar = tqdm(train_dataset)
        for batch_idx, (path, imgs, targets) in enumerate(progressbar):
            imgs = imgs.to(device)
            # optimizer set to zero
            optimizer.zero_grad()
            output = resnet(imgs)
            # one_hot code
            one_hot_targets = F.one_hot(targets, args.num_classes)
            loss = CEloss(output, targets)
            loss.backward()  # loss value backward
            optimizer.step()  # update parameters at optimizer
            train_loss += loss.item()
            progressbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(imgs), len(train_dataset.dataset),
                       100. * batch_idx / len(train_dataset),
                       loss.item() / len(imgs)))
        writer.add_scalar('loss/train', train_loss/len(train_dataset.dataset), epoch)

        if epoch%args.checkpoint_interval == 0:
            torch.save(resnet.state_dict(), args.weights_path+'/resnet18_clf_{}.pth'.format(epoch))

    writer.close()




