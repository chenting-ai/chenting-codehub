import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
import scipy.io as scipio
import numpy as np
from net_and_data import MyDataset2, ConvNet1_1, device, weights_init

date_num = 4  # 第几次


def pre_trained(judge):

    model = ConvNet1_1().to(device)
    mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_SMI_2020_05_18_0%d.pth' % date_num
    model.load_state_dict(torch.load(mode_patch))
    temp_weight = model.conv1.weight.data  # 检验网络权重是否变化的初始网络参数
    print(temp_weight)
    temp_weight = temp_weight.cpu().detach().numpy()
    print(temp_weight)
    pathmat = './net_parameter/net_conv1_parameter_2020_05_18_0%d.mat' % date_num
    scipio.savemat(pathmat, {'temp_weight_2020_05_18_0%d' % date_num: temp_weight})


if __name__ == '__main__':
    pre_trained(0)
    # inversion(0)
    # tensorboard --logdir E:\Python_project\Predict_Impedence\loss\pre_train_loss_marmousi2019_12_10_11
