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
from net_and_data import MyDataset2, ConvNet1_3, device
from my_loss_function import my_mse_loss_fun, syn_seismic_fun2

BATCH_SIZE = 1  # BATCH_SIZE大小
BATCH_LEN = 60  # BATCH_LEN长度
EPOCHS = 1000  # 总共训练批次
coefval = 0.4  # 相关系数阈值
Rval = 50  # 半径阈值
data_rate = 0.5  # 选用前50%的xline线训练

number = np.int(60/BATCH_LEN)  # 一道有几个batch长度
lr = 0.001  # 学习步长

# 加载训练用井旁地震数据
dataFile1 = 'D://Python_project/Predict_Impedence_hu/SMI_data/well_seismic.mat'
well_seismic = scipio.loadmat(dataFile1)
train_well_seismic = well_seismic['well_seismic']
# train_well_seismic = torch.from_numpy(train_well_seismic)
# train_well_seismic = train_well_seismic.float()
# train_well_seismic = train_well_seismic.to(device)

# 加载训练用测井数据
dataFile2 = 'D://Python_project/Predict_Impedence_hu2/SMI_data/well.mat'
well = scipio.loadmat(dataFile2)
train_well = well['well']
# train_well = torch.from_numpy(train_well)
# train_well = train_well.float()
# train_well = train_well.to(device)

# 加载训练用地震数据Xline1_75
dataFile3 = 'D://Python_project/Predict_Impedence_hu2/SMI_data/Xline1_110_seismic.mat'
Xline1_110_seismic = scipio.loadmat(dataFile3)
train1_110_seismic = Xline1_110_seismic['Xline1_110_seismic']
# train1_75_seismic = torch.from_numpy(train1_75_seismic)
# train1_75_seismic = train1_75_seismic.float()
# train1_75_seismic = train1_75_seismic.to(device)

# 加载训练用标签数据
dataFile4 = 'D://Python_project/Predict_Impedence_hu2/SMI_data/Xline1_110_SMI_impedance.mat'
Xline1_110_SMI_impedance = scipio.loadmat(dataFile4)
Xline1_110_label_impedance = Xline1_110_SMI_impedance['Xline1_110_SMI_impedance']
# Xline1_75_label_impedance = torch.from_numpy(Xline1_75_label_impedance)
# Xline1_75_label_impedance = Xline1_75_label_impedance.float()
# Xline1_75_label_impedance = Xline1_75_label_impedance.to(device)

# 加载线道号
dataFile5 = 'D://Python_project/Predict_Impedence_hu2/SMI_data/Xline_Inline_number.mat'
Xline_Inline = scipio.loadmat(dataFile5)
Xline_Inline_number = Xline_Inline['Xline_Inline_number']
# Xline_Inline_number = torch.from_numpy(Xline_Inline_number)
# Xline_Inline_number = Xline_Inline_number.float()
# Xline_Inline_number = Xline_Inline_number.to(device)

# 加载训练用地震数据3
dataFile6 = 'D://Python_project/Predict_Impedence_hu2/SMI_data/Xline16_seismic.mat'
Xline16_seismic = scipio.loadmat(dataFile6)
test_Xline16_seismic = Xline16_seismic['Xline16_seismic']
# test_Xline16_seismic = torch.from_numpy(test_Xline16_seismic)
# test_Xline16_seismic = test_Xline16_seismic.float()
# test_Xline16_seismic = test_Xline16_seismic.to(device)

# 加载训练用地震数据3
dataFile7 = 'D://Python_project/Predict_Impedence_hu2/SMI_data/Xline76_seismic.mat'
Xline76_seismic = scipio.loadmat(dataFile7)
test_Xline76_seismic = Xline76_seismic['Xline76_seismic']
# test_Xline76_seismic = torch.from_numpy(test_Xline76_seismic)
# test_Xline76_seismic = test_Xline76_seismic.float()
# test_Xline76_seismic = test_Xline76_seismic.to(device)
# 加载子波
waveFile = 'D://Python_project/Predict_Impedence_SMI/SMI_data/wavelet_1_1.mat'
wavel = scipio.loadmat(waveFile)
wavele = wavel['wavelet_1_1']
wavelet = torch.from_numpy(wavele)
wavelet = wavelet.float()
wavelet = wavelet.view(wavelet.size(1))
wavelet = wavelet.to(device)


def pre_trained(judge):

    writer = SummaryWriter(log_dir='./loss/pre_train_loss_model1/pre_train_loss_SMI_2020_04_23_01')

    if judge == 0:
        model = ConvNet1_3().to(device)
        temp = 10000000000000
        epoch_num = 1
    else:
        model = ConvNet1_3().to(device)
        mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_SMI_2020_04_23_01.pth'
        model.load_state_dict(torch.load(mode_patch))
        path_temp = './Temporary_parameters/pre_temp_model1.mat'
        temp = scipio.loadmat(path_temp)
        temp = temp['temp'].item()
        path_epoch = './Temporary_parameters/pre_epoch_num_model1.mat'
        epoch_num = scipio.loadmat(path_epoch)
        epoch_num = epoch_num['epoch_num'].item()+1

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoch_num, EPOCHS+1):
        print(epoch)
        trace_number = np.random.randint(0, 142*110*data_rate, 1)
        # print(trace_number)

        # 计算相关系数
        coef_seismic = np.zeros((105, Xline1_110_label_impedance.shape[1]))
        coef_seismic[0, :] = train1_110_seismic[trace_number, :]
        coef_seismic[1:105, :] = train_well_seismic[:, :]
        temp_coef = np.corrcoef(coef_seismic)

        # 优选出相关系数大于阈值并且半径范围内的井
        num = 0
        tempval = np.zeros(0)
        temp_train_well = np.zeros(0)
        temp_train_well_seisic = np.zeros(0)
        absCORcoef = np.abs(temp_coef[0, 1:105])
        for k in range(0, 104):
            if absCORcoef[k] > coefval:
                # 井数据的坐标
                wellxline = Xline_Inline_number[0, k]
                wellinline = Xline_Inline_number[1, k]
                # 目标地震数据的坐标
                seismicinline = np.mod(trace_number + 1, 142)
                seismicxline = (trace_number + 1 - seismicinline) / 142 + 1
                R = np.sqrt((seismicxline - wellxline) * (seismicxline - wellxline) + (seismicinline - wellinline) * (
                        seismicinline - wellinline))
                if R < Rval:
                    tempval = np.append(tempval, absCORcoef[k])
                    temp_train_well = np.append(temp_train_well, train_well[k, :])
                    temp_train_well_seisic = np.append(temp_train_well_seisic, train_well_seismic[k, :])
                    num = num + 1

        if num < 1:
            num = 104
            tempval = np.zeros(0)
            for max_num in range(0, num):
                temp_tempval = max(absCORcoef)
                tempval = np.append(tempval, temp_tempval)
                for max_num2 in range(0, 104):
                    if temp_tempval == absCORcoef[max_num2]:
                        absCORcoef[max_num2] = 0
                        temp_train_well = np.append(temp_train_well, train_well[max_num2, :])
                        temp_train_well_seisic = np.append(temp_train_well_seisic, train_well_seismic[max_num2, :])

        if num > 1:
            maxval = max(tempval)
            minval = min(tempval)
            max_minlen = maxval - minval

            tempval = (tempval - minval) / max_minlen
        else:
            tempval = 1
        valsum = sum(tempval)
        tempval = tempval / valsum
        tempval = torch.from_numpy(tempval)
        tempval = tempval.view(1, -1)
        tempval = tempval.float()
        tempval = tempval.to(device)

        temp_train_well = torch.from_numpy(temp_train_well)
        temp_train_well = temp_train_well.view(num, -1)
        # temp_train_well = tempval.mm(temp_train_well)

        temp_train_well = temp_train_well.float()
        temp_train_well = temp_train_well.to(device)
        # temp_train_well = temp_train_well.view(num, -1)

        # temp_train_well_seisic = torch.from_numpy(temp_train_well_seisic)
        # temp_train_well_seisic = temp_train_well_seisic.float()
        # temp_train_well_seisic = temp_train_well_seisic.to(device)
        # temp_train_well_seisic = temp_train_well_seisic.view(num, -1)
        # temp_seismic = torch.from_numpy(train1_75_seismic[trace_number, :])
        # temp_seismic = temp_seismic.float()
        # temp_seismic = temp_seismic.to(device)
        # temp_seismic = temp_seismic.view(1, -1)

        temp_lable = torch.from_numpy(Xline1_110_label_impedance[trace_number, :])
        temp_lable = temp_lable.float()
        temp_lable = temp_lable.to(device)
        temp_lable = temp_lable.view(1, -1)

        temp_train_seismic = train1_110_seismic[trace_number, :]
        temp_train_seismic = torch.from_numpy(temp_train_seismic)
        temp_train_seismic = temp_train_seismic.float()
        temp_train_seismic = temp_train_seismic.to(device)
        # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入
        rand = np.random.randint(0, 60 - BATCH_LEN + 1, 1)
        train_dataset = MyDataset2(temp_train_well[:, rand[0]:rand[0] + BATCH_LEN],
                                   temp_lable[:, rand[0]:rand[0] + BATCH_LEN])
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True,
                                      drop_last=False)
        epoch_loss = []

        for itr, (train_dt, train_lable) in enumerate(train_dataloader):
            train_dt, train_lable = train_dt.to(device), train_lable.to(device)
            train_dt = train_dt.float()
            train_lable = train_lable.float()

            model.train()
            optimizer.zero_grad()
            output = model(train_dt, tempval)
            # syn_seismic = syn_seismic_fun2(output, wavelet)
            # syn_seismic = syn_seismic.float()
            # loss = F.mse_loss(syn_seismic, temp_train_seismic) + F.mse_loss(output, train_lable)
            loss = F.mse_loss(output, train_lable)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        epoch_loss = np.sum(np.array(epoch_loss))
        writer.add_scalar('Train/MSE', epoch_loss, epoch)
        epoch_num = epoch
        print('Train set: Average loss: {:.15f}'.format(epoch_loss))
        if epoch_loss < temp:
            path = './model_file/pre_trained_network_model_model1/pre_trained_network_model_SMI_2020_04_23_01.pth'
            torch.save(model.state_dict(), path)
            temp = epoch_loss
        path_temp = './Temporary_parameters/pre_temp_model1.mat'
        path_epoch = './Temporary_parameters/pre_epoch_num_model1.mat'
        scipio.savemat(path_temp, {'temp': temp})
        scipio.savemat(path_epoch, {'epoch_num': epoch_num})
    writer.add_graph(model, (train_dt, tempval))


def tested():

    impedance_xline16 = np.zeros((test_Xline16_seismic.shape[0], test_Xline16_seismic.shape[1]))

    model = ConvNet1_3().to(device)
    mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_SMI_2020_04_23_01.pth'
    model.load_state_dict(torch.load(mode_patch))
    for num_batchlen in range(0, number):
        for trace_number in range(0, test_Xline16_seismic.shape[0]):
            print(trace_number)

            # 计算相关系数
            coef_seismic = np.zeros((105, test_Xline16_seismic.shape[1]))
            coef_seismic[0, :] = test_Xline16_seismic[trace_number, :]
            coef_seismic[1:105, :] = train_well_seismic[:, :]
            temp_coef = np.corrcoef(coef_seismic)

            # 优选出相关系数大于阈值并且半径范围内的井
            num = 0
            tempval = np.zeros(0)
            temp_train_well = np.zeros(0)
            temp_train_well_seisic = np.zeros(0)
            absCORcoef = np.abs(temp_coef[0, 1:105])
            temp_trace_number = trace_number*110+16
            for k in range(0, 104):
                if absCORcoef[k] > coefval:
                    # 井数据的坐标
                    wellxline = Xline_Inline_number[0, k]
                    wellinline = Xline_Inline_number[1, k]
                    # 目标地震数据的坐标
                    seismicxline = np.mod(temp_trace_number, 110)
                    seismicinline = (temp_trace_number - seismicxline) / 110 + 1
                    R = np.sqrt(
                        (seismicxline - wellxline) * (seismicxline - wellxline) + (seismicinline - wellinline) * (
                                seismicinline - wellinline))
                    if R < Rval:
                        tempval = np.append(tempval, absCORcoef[k])
                        temp_train_well = np.append(temp_train_well, train_well[k, :])
                        temp_train_well_seisic = np.append(temp_train_well_seisic, train_well_seismic[k, :])
                        num = num + 1

            if num < 1:
                num = 104
                tempval = np.zeros(0)
                for max_num in range(0, num):
                    temp_tempval = max(absCORcoef)
                    tempval = np.append(tempval, temp_tempval)
                    for max_num2 in range(0, 104):
                        if temp_tempval == absCORcoef[max_num2]:
                            absCORcoef[max_num2] = 0
                            temp_train_well = np.append(temp_train_well, train_well[max_num2, :])
                            temp_train_well_seisic = np.append(temp_train_well_seisic, train_well_seismic[max_num2, :])
            if num > 1:
                maxval = max(tempval)
                minval = min(tempval)
                max_minlen = maxval - minval

                tempval = (tempval - minval) / max_minlen
            else:
                tempval=1
            valsum = sum(tempval)
            tempval = tempval/valsum

            tempval = torch.from_numpy(tempval)
            tempval = tempval.view(1, -1)
            tempval = tempval.float()
            tempval = tempval.to(device)

            temp_train_well = torch.from_numpy(temp_train_well)
            temp_train_well = temp_train_well.view(num, -1)
            # temp_train_well = tempval.mm(temp_train_well)

            temp_train_well = temp_train_well.float()
            temp_train_well = temp_train_well.to(device)

            temp_lable = torch.from_numpy(test_Xline16_seismic[trace_number, :])
            temp_lable = temp_lable.float()
            temp_lable = temp_lable.to(device)
            temp_lable = temp_lable.view(1, -1)
            # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入
            train_dataset = MyDataset2(temp_train_well[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)],
                                       temp_lable[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)])
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True,
                                          drop_last=False)

            for itr, (train_dt, train_lable) in enumerate(train_dataloader):
                train_dt, train_lable = train_dt.to(device), train_lable.to(device)
                train_dt = train_dt.float()

                # model.train()
                output = model(train_dt, tempval)
                # train_dt = train_dt.view(tempval.size(1), -1)
                # output = tempval.mm(train_dt)
                np_output = output.cpu().detach().numpy()
                impedance_xline16[(trace_number * BATCH_SIZE):((trace_number + 1) * BATCH_SIZE), (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)] = np_output

    pathmat = './SMI_out/test_X16_Impedance_model1_2020_04_23_01.mat'
    scipio.savemat(pathmat, {'test_X16_Impedance_model1_2020_04_23_01': impedance_xline16})


def tested2():
    impedance_xline76 = np.zeros((test_Xline76_seismic.shape[0], test_Xline76_seismic.shape[1]))

    model = ConvNet1_3().to(device)
    mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_SMI_2020_04_23_01.pth'
    model.load_state_dict(torch.load(mode_patch))
    for num_batchlen in range(0, number):
        for trace_number in range(0, test_Xline76_seismic.shape[0]):
            print(trace_number)

            # 计算相关系数
            coef_seismic = np.zeros((105, test_Xline76_seismic.shape[1]))
            coef_seismic[0, :] = test_Xline76_seismic[trace_number, :]
            coef_seismic[1:105, :] = train_well_seismic[:, :]
            temp_coef = np.corrcoef(coef_seismic)

            # 优选出相关系数大于阈值并且半径范围内的井
            num = 0
            tempval = np.zeros(0)
            temp_train_well = np.zeros(0)
            temp_train_well_seisic = np.zeros(0)
            absCORcoef = np.abs(temp_coef[0, 1:105])
            temp_trace_number = trace_number*110+76
            for k in range(0, 104):
                if absCORcoef[k] > coefval:
                    # 井数据的坐标
                    wellxline = Xline_Inline_number[0, k]
                    wellinline = Xline_Inline_number[1, k]
                    # 目标地震数据的坐标
                    seismicxline = np.mod(temp_trace_number, 110)
                    seismicinline = (temp_trace_number - seismicxline) / 110 + 1
                    R = np.sqrt(
                        (seismicxline - wellxline) * (seismicxline - wellxline) + (seismicinline - wellinline) * (
                                seismicinline - wellinline))
                    if R < Rval:
                        tempval = np.append(tempval, absCORcoef[k])
                        temp_train_well = np.append(temp_train_well, train_well[k, :])
                        temp_train_well_seisic = np.append(temp_train_well_seisic, train_well_seismic[k, :])
                        num = num + 1

            if num < 1:
                num = 104
                tempval = np.zeros(0)
                for max_num in range(0, num):
                    temp_tempval = max(absCORcoef)
                    tempval = np.append(tempval, temp_tempval)
                    for max_num2 in range(0, 104):
                        if temp_tempval == absCORcoef[max_num2]:
                            absCORcoef[max_num2] = 0
                            temp_train_well = np.append(temp_train_well, train_well[max_num2, :])
                            temp_train_well_seisic= np.append(temp_train_well_seisic, train_well_seismic[max_num2, :])
            if num > 1:
                maxval = max(tempval)
                minval = min(tempval)
                max_minlen = maxval - minval

                tempval = (tempval - minval) / max_minlen
            else:
                tempval=1
            valsum = sum(tempval)
            tempval = tempval/valsum
            tempval = torch.from_numpy(tempval)
            tempval = tempval.view(1, -1)
            tempval = tempval.float()
            tempval = tempval.to(device)

            temp_train_well = torch.from_numpy(temp_train_well)
            temp_train_well = temp_train_well.view(num, -1)
            # temp_train_well = tempval.mm(temp_train_well)

            temp_train_well = temp_train_well.float()
            temp_train_well = temp_train_well.to(device)

            temp_lable = torch.from_numpy(test_Xline76_seismic[trace_number, :])
            temp_lable = temp_lable.float()
            temp_lable = temp_lable.to(device)
            temp_lable = temp_lable.view(1, -1)
            # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入
            train_dataset = MyDataset2(temp_train_well[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)],
                                       temp_lable[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)])
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True,
                                          drop_last=False)

            for itr, (train_dt, train_lable) in enumerate(train_dataloader):
                train_dt, train_lable = train_dt.to(device), train_lable.to(device)
                train_dt = train_dt.float()

                # model.train()
                output = model(train_dt, tempval)
                # train_dt = train_dt.view(tempval.size(1), -1)
                # output = tempval.mm(train_dt)
                np_output = output.cpu().detach().numpy()
                impedance_xline76[(trace_number * BATCH_SIZE):((trace_number + 1) * BATCH_SIZE), (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)] = np_output

    pathmat = './SMI_out/pre_X76_Impedance_model1_2020_04_23_01.mat'
    scipio.savemat(pathmat, {'pre_X76_Impedance_model1_2020_04_23_01': impedance_xline76})


if __name__ == '__main__':
    pre_trained(0)
    tested()
    tested2()
    # inversion(0)
    # tensorboard --logdir E:\Python_project\Predict_Impedence\loss\pre_train_loss_marmousi2019_12_10_11
