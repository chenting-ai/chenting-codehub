import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
import scipy.io as scipio
import numpy as np
from tensorboardX import SummaryWriter
from net_and_data import MyDataset2, ConvNet, RNN, ConvNet3, device
from my_loss_function import my_mse_loss_fun, syn_seismic_fun

BATCH_SIZE = 1  # BATCH_SIZE大小
EPOCHS = 200  # 总共训练批次
lr = 0.001  # 学习步长

# 加载训练用地震数据2
dataFile1 = 'D://Python_project/Predict_Impedence_SMI/SMI_data/Xline76_seismic_1_1.mat'
Xline76_seismic = scipio.loadmat(dataFile1)
train_seismic = Xline76_seismic['Xline76_seismic_1_1']

# 加载初始模型数据数据
m0_File = 'D://Python_project/Predict_Impedence_SMI/SMI_out/pre_Impedance2020_04_06_01.mat'
m0 = scipio.loadmat(m0_File)
m0 = m0['pre_Impedance2020_04_06_01']

# 加载子波
waveFile = 'D://Python_project/Predict_Impedence_SMI/SMI_data/wavelet_1_1.mat'
wavel = scipio.loadmat(waveFile)
wavele = wavel['wavelet_1_1']


def inversion(judge):
    trace_number = m0.shape[0]
    point_number = m0.shape[1]
    impedance = np.zeros((m0.shape[0], m0.shape[1]))
    wavelet = torch.from_numpy(wavele)
    wavelet = wavelet.float()
    wavelet = wavelet.view(wavelet.size(1))
    wavelet = wavelet.to(device)
    for k in range(0, trace_number):
        writer = SummaryWriter(log_dir='./loss/inversion_loss_SMI2020_04_12_02/inversion_loss_SMI_%d' % k)
        train_dataset = MyDataset2(m0[k, :].reshape(1, point_number), train_seismic[k, :].reshape(1, point_number))
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True,
                                      drop_last=False)

        if judge == 0:
            model = ConvNet3().to(device)
            temp = 10000000000000
            epoch_num = 1
        else:
            model = ConvNet3().to(device)
            mode_patch = './model_file/inversion_model_SMI2020_04_12_02/inversion_model_SMI_%d.pth' % k
            model.load_state_dict(torch.load(mode_patch))
            path_temp = './Temporary_parameters/inversion_temp.mat'
            temp = scipio.loadmat(path_temp)
            temp = temp['temp'].item()
            path_epoch = './Temporary_parameters/inversion_epoch_num.mat'
            epoch_num = scipio.loadmat(path_epoch)
            epoch_num = epoch_num['epoch_num'].item() + 1

        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epoch_num, EPOCHS + 1):

            print(k, epoch)
            epoch_loss = []

            for itr, (train_dt, train_lable) in enumerate(train_dataloader):
                train_dt, train_lable = train_dt.to(device), train_lable.to(device)
                train_dt = train_dt.float()
                train_lable = train_lable.float()

                model.train()
                optimizer.zero_grad()
                output = model(train_dt)
                syn_seismic = syn_seismic_fun(output, wavelet)
                syn_seismic = syn_seismic.float()
                loss = F.mse_loss(syn_seismic, train_lable) + F.mse_loss(output, train_dt)
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())
            writer.add_graph(model, (train_dt,))

            epoch_loss = np.sum(np.array(epoch_loss))
            writer.add_scalar('Train/MSE', epoch_loss, epoch)

            epoch_num = epoch
            print('Train set: Average loss: {:.15f}'.format(epoch_loss))
            if epoch_loss < temp:
                path = './model_file/inversion_model_SMI2020_04_12_02/inversion_model_SMI_%d.pth' % k
                torch.save(model.state_dict(), path)
                temp = epoch_loss
                np_output = output.cpu().detach().numpy()
                impedance[k, :] = np_output
                writer.close()
                pathmat = './SMI_out/inversion_Impedance2020_04_12_02.mat'
                scipio.savemat(pathmat, {'inversion_Impedance2020_04_12_02': impedance})
            path_temp = './Temporary_parameters/inversion_temp.mat'
            path_epoch = './Temporary_parameters/inversion_epoch_num.mat'
            scipio.savemat(path_temp, {'temp': temp})
            scipio.savemat(path_epoch, {'epoch_num': epoch_num})


if __name__ == '__main__':
    inversion(0)
    # tensorboard --logdir E:\Python_project\Predict_Impedence\train_logs
