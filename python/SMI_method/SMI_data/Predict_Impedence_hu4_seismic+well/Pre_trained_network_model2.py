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
from net_and_data import MyDataset2, ConvNet2_2, device

BATCH_SIZE = 2  # BATCH_SIZE大小
BATCH_LEN = 60  # BATCH_LEN长度
EPOCHS = 20  # 总共训练批次

number = np.int(60/BATCH_LEN)  # 一道有几个batch长度
lr = 0.001  # 学习步长

# 加载训练用地震数据
dataFile1 = 'D://Python_project/Predict_Impedence_hu/SMI_data/well_seismic.mat'
well_seismic = scipio.loadmat(dataFile1)
train_well_seismic = well_seismic['well_seismic']

# 加载训练用测井数据
dataFile2 = 'D://Python_project/Predict_Impedence_hu/SMI_data/well.mat'
well = scipio.loadmat(dataFile2)
train_well = well['well']

# 加载训练用地震数据2
dataFile3 = 'D://Python_project/Predict_Impedence_hu/SMI_data/Xline76_seismic.mat'
Xline76_seismic = scipio.loadmat(dataFile3)
train_seismic = Xline76_seismic['Xline76_seismic']


def pre_trained(judge):

    writer = SummaryWriter(log_dir='./loss/pre_train_loss_model2/pre_train_loss_SMI_2020_04_17_01')

    if judge == 0:
        model = ConvNet2_2().to(device)
        temp = 10000000000000
        epoch_num = 1
    else:
        model = ConvNet2_2().to(device)
        mode_patch = './model_file/pre_trained_network_model_model2/pre_trained_network_model_SMI_2020_04_17_01.pth'
        model.load_state_dict(torch.load(mode_patch))
        path_temp = './Temporary_parameters/pre_temp_model2.mat'
        temp = scipio.loadmat(path_temp)
        temp = temp['temp'].item()
        path_epoch = './Temporary_parameters/pre_epoch_num_model2.mat'
        epoch_num = scipio.loadmat(path_epoch)
        epoch_num = epoch_num['epoch_num'].item()+1

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoch_num, EPOCHS+1):
        rand = np.random.randint(0, 60-BATCH_LEN+1, 1)

        print(epoch)
        epoch_loss = []

        train_dataset = MyDataset2(train_well_seismic[:, rand[0]:rand[0]+BATCH_LEN], train_well[:, rand[0]:rand[0]+BATCH_LEN])
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=5, shuffle=True,
                                      drop_last=False)

        for itr, (train_dt, train_lable) in enumerate(train_dataloader):

            train_dt, train_lable = train_dt.to(device), train_lable.to(device)
            train_dt = train_dt.float()
            train_lable = train_lable.float()

            model.train()
            optimizer.zero_grad()
            output = model(train_dt)
            loss = F.mse_loss(output, train_lable)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
        writer.add_graph(model, (train_dt,))

        epoch_loss = np.sum(np.array(epoch_loss))
        writer.add_scalar('Train/MSE', epoch_loss, epoch)

        epoch_num = epoch
        print('Train set: Average loss: {:.15f}'.format(epoch_loss))
        if epoch_loss < temp:
            path = './model_file/pre_trained_network_model_model2/pre_trained_network_model_SMI_2020_04_17_01.pth'
            torch.save(model.state_dict(), path)
            temp = epoch_loss
        path_temp = './Temporary_parameters/pre_temp_model2.mat'
        path_epoch = './Temporary_parameters/pre_epoch_num_model2.mat'
        scipio.savemat(path_temp, {'temp': temp})
        scipio.savemat(path_epoch, {'epoch_num': epoch_num})
    writer.close()


def tested():

    impedance_xline76 = np.zeros((train_seismic.shape[0], train_seismic.shape[1]))
    for num in range(0, number):

        test_dataset = MyDataset2(train_seismic[:, (num * BATCH_LEN):((num + 1) * BATCH_LEN)], train_seismic[:, (num * BATCH_LEN):((num + 1) * BATCH_LEN)])
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=5, shuffle=True, drop_last=False)

        model = ConvNet2_2().to(device)
        mode_patch = './model_file/pre_trained_network_model_model2/pre_trained_network_model_SMI_2020_04_17_01.pth'
        model.load_state_dict(torch.load(mode_patch))

        for itr, (test_dt, test_lable) in enumerate(test_dataloader):
            test_dt, test_lable = test_dt.to(device), test_lable.to(device)
            test_dt = test_dt.float()
            output = model(test_dt)
            np_output = output.cpu().detach().numpy()
            impedance_xline76[(itr * BATCH_SIZE):((itr + 1) * BATCH_SIZE), (num * BATCH_LEN):((num + 1) * BATCH_LEN)] = np_output

    pathmat = './SMI_out/pre_X16Impedance_model2_2020_04_17_01.mat'
    scipio.savemat(pathmat, {'pre_X16Impedance_model2_2020_04_17_01': impedance_xline76})


def tested2():

    impedance_well = np.zeros((train_well_seismic.shape[0], train_well_seismic.shape[1]))

    for num in range(0, number):

        test_dataset = MyDataset2(train_well_seismic[:, (num * BATCH_LEN):((num + 1) * BATCH_LEN)], train_well_seismic[:, (num * BATCH_LEN):((num + 1) * BATCH_LEN)])
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=5, shuffle=True, drop_last=False)

        model = ConvNet2_2().to(device)
        mode_patch = './model_file/pre_trained_network_model_model2/pre_trained_network_model_SMI_2020_04_17_01.pth'
        model.load_state_dict(torch.load(mode_patch))

        for itr, (test_dt, test_lable) in enumerate(test_dataloader):
            test_dt, test_lable = test_dt.to(device), test_lable.to(device)
            test_dt = test_dt.float()
            output = model(test_dt)
            np_output = output.cpu().detach().numpy()
            impedance_well[(itr * BATCH_SIZE):((itr + 1) * BATCH_SIZE), (num * BATCH_LEN):((num + 1) * BATCH_LEN)] = np_output

    pathmat = './SMI_out/test_Impedance_model2_2020_04_17_01.mat'
    scipio.savemat(pathmat, {'test_Impedance_model2_2020_04_17_01': impedance_well})


if __name__ == '__main__':
    pre_trained(0)
    tested()
    tested2()
    # tensorboard --logdir E:\Python_project\Predict_Impedence\loss\pre_train_loss_marmousi2019_12_10_11
