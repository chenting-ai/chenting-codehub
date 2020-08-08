import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from unets.unet_blocks import *
from unets.resnet_blocks import _resnet, BasicBlock, Bottleneck

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


class MyDataset(data.Dataset):
    def __init__(self, parameter_data, parameter_target):
        self.data = np.array(parameter_data)
        self.target = np.array(parameter_target)

    def __getitem__(self, index):  # 返回的是tensor
        d, target = self.data[index, :], self.target[index, :]
        d = torch.from_numpy(d)
        target = torch.from_numpy(target)
        return d, target

    def __len__(self):
        return self.target.shape[0]


class MyDataset2(data.Dataset):
    def __init__(self, parameter_data, parameter_target):
        self.data = np.array(parameter_data)
        self.target = np.array(parameter_target)

    def __getitem__(self, index):  # 返回的是tensor
        d, target = self.data[:, :], self.target[index, :]
        d = torch.from_numpy(d)
        target = torch.from_numpy(target)
        return d, target

    def __len__(self):
        return self.target.shape[0]



