import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
import scipy.io as scipio
import numpy as np
from net_and_data import MyDataset, ConvNet, RNN, ConvNet2, device

BATCH_SIZE = 5  # BATCH_SIZE大小
# 加载测试用地震数据
seisnicFile = 'E://Python_project/Predict_Impedence/marmousi_data/poststack_normal.mat'
poststack = scipio.loadmat(seisnicFile)
test_seismic = poststack['poststack']


def generate_the_initial_model():
    model = RNN().to(device)
    model.load_state_dict(torch.load('./model_file/pre_trained_network_model_marmousi2019_12_16_19.pth'))
    model.eval()

    test_dataset = MyDataset(test_seismic, test_seismic)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=3, shuffle=True, drop_last=False)

    pre_inital_ip = np.zeros((test_seismic.shape[0], test_seismic.shape[1]))

    for itr, (test_dt, test_lable) in enumerate(test_dataloader):
        test_dt, test_lable = test_dt.to(device), test_lable.to(device)
        test_dt = test_dt.float()
        output = model(test_dt)
        np_output = output.cpu().detach().numpy()
        pre_inital_ip[(itr*BATCH_SIZE):((itr+1)*BATCH_SIZE), :] = np_output
    pathmat = './marmousi_out/pre_test2019_12_16_19.mat'
    scipio.savemat(pathmat, {'pre_test2019_12_16_19': pre_inital_ip})


if __name__ == '__main__':
    generate_the_initial_model()
