# -*- coding: utf-8 -*-
# Author: ShaoChang Wang
# Time: 2020-08-25 23:20:08
import math
import torch
import torchvision
from torch import nn
import datetime
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


def conv3x3(input_ch, output_ch, stride=1):
    return nn.Conv2d(input_ch, output_ch, kernel_size=3,
              stride=stride, padding=1, bias=False)


class Basicblock(nn.Module):
    expansion= 1

    def __init__(self, in_channels, out_channels, layer_id, stride=1, downsample=None):
        super(Basicblock, self).__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.layer_id = layer_id
        self.downsample = downsample

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual  # The key of residual network
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, blocks, repeats, input_size, num_classes=1000):
        super(ResNet18, self).__init__()
        self.input_size = input_size
        self.first_channel = 64
        self.input_channel = self.first_channel

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.first_layer = nn.Sequential(OrderedDict([
        #     ('conv{}_{}'.format(1, 1), nn.Conv2d(in_channels=3,
        #                                          out_channels=self.first_channel, kernel_size=7, stride=2,
        #                                          padding=3, bias=False)),
        #     ('bn{}_{}'.format(1, 1), nn.BatchNorm2d(self.first_channel)),
        #     ('relu{}_{}'.format(1, 1), nn.ReLU(inplace=True)),
        #     ('pool{}_{}'.format(1, 1), nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))]))

        self.layer1 = self.make_layer(blocks, 64, repeats[0], 2)
        self.layer2 = self.make_layer(blocks, 128, repeats[1], 3, stride=2)
        self.layer3 = self.make_layer(blocks, 256, repeats[2], 4, stride=2)
        self.layer4 = self.make_layer(blocks, 512, repeats[3], 5, stride=2)
        self.avgp = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layer(self, res_block, out_channels, repeat, layer_id, stride=1):
        downsample = None
        if stride != 1 or self.input_channel != out_channels * res_block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.input_channel, out_channels * res_block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * res_block.expansion)
            )
        layers = [res_block(self.input_channel, out_channels * res_block.expansion,
                            layer_id, stride, downsample)]
        self.input_channel = out_channels * res_block.expansion  # update input channel of next layer
        for i in range(1, repeat):
            # second convolution layer does not need to downsample
            layers.append(res_block(self.input_channel, out_channels * res_block.expansion, layer_id))
        # return the residual layer
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgp(out)
        out = out.view(out.size(0), -1)  # convert to [Batch, N]
        out = self.fc(out)

        return out  # cross_entropy_loss will do softmax automatically


if __name__=='__main__':
    resnet = ResNet18(Basicblock, [2, 2, 2, 2], 224, 3, 10)
    print(resnet)
    log_path = 'logs'
    writer = SummaryWriter(log_path, 'Training process of cars detection via ResNet18!')
    # Save model structure
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad = True)

    grid = torchvision.utils.make_grid(dummy_input)
    writer.add_image('images', grid, 0)
    writer.add_graph(resnet, grid)
    writer.close()