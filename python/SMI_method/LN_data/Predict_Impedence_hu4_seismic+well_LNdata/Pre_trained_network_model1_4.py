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
from net_and_data import MyDataset2, BSsequential_net_D, device, weights_init
from my_loss_function import my_mse_loss_fun, syn_seismic_fun2
#  将井数据作为标签


BATCH_SIZE = 1  # BATCH_SIZE大小
BATCH_LEN = 60  # BATCH_LEN长度
EPOCHS = 10000  # 总共训练批次
coefval = 0.4  # 相关系数阈值
Rval = 50  # 半径阈值
num_well = 5  # 井数量
data_rate = 1  # 选用前50%的xline线训练
number = np.int(60/BATCH_LEN)  # 一道有几个batch长度

is_consistent = 0  # 是否要确定的随机路径
is_synseismic = 0  # 是否增加合成地震记录约束
is_inherit_net = 0  # 是否继承之前的网络
which_choose_well = 1  # 第几种选井方式 0.选前num_well口，1.在阈值和半径范围内选前num_well口，

date_num = 2  # 今天第几次跑
# 加载数据
if 1:
    # 加载训练用井旁地震数据
    dataFile1 = './SMI_data/well_seismic.mat'
    well_seismic = scipio.loadmat(dataFile1)
    train_well_seismic = well_seismic['well_seismic']
    # train_well_seismic = torch.from_numpy(train_well_seismic)
    # train_well_seismic = train_well_seismic.float()
    # train_well_seismic = train_well_seismic.to(device)

    # 加载训练用测井数据
    dataFile2 = './SMI_data/well.mat'
    well = scipio.loadmat(dataFile2)
    train_well = well['well']
    # train_well = torch.from_numpy(train_well)
    # train_well = train_well.float()
    # train_well = train_well.to(device)

    # 加载训练用地震数据Xline1_75
    dataFile3 = './SMI_data/Xline1_110_seismic.mat'
    Xline1_110_seismic = scipio.loadmat(dataFile3)
    train1_110_seismic = Xline1_110_seismic['Xline1_110_seismic']
    # train1_75_seismic = torch.from_numpy(train1_75_seismic)
    # train1_75_seismic = train1_75_seismic.float()
    # train1_75_seismic = train1_75_seismic.to(device)

    # 加载训练用标签数据
    dataFile4 = './SMI_data/Xline1_110_SMI_impedance.mat'
    Xline1_110_SMI_impedance = scipio.loadmat(dataFile4)
    Xline1_110_label_impedance = Xline1_110_SMI_impedance['Xline1_110_SMI_impedance']
    # Xline1_75_label_impedance = torch.from_numpy(Xline1_75_label_impedance)
    # Xline1_75_label_impedance = Xline1_75_label_impedance.float()
    # Xline1_75_label_impedance = Xline1_75_label_impedance.to(device)

    # 加载线道号
    dataFile5 = './SMI_data/Xline_Inline_number.mat'
    Xline_Inline = scipio.loadmat(dataFile5)
    Xline_Inline_number = Xline_Inline['Xline_Inline_number']
    # Xline_Inline_number = torch.from_numpy(Xline_Inline_number)
    # Xline_Inline_number = Xline_Inline_number.float()
    # Xline_Inline_number = Xline_Inline_number.to(device)

    # 加载训练用地震数据3
    dataFile6 = './SMI_data/Xline16_seismic.mat'
    Xline16_seismic = scipio.loadmat(dataFile6)
    test_Xline16_seismic = Xline16_seismic['Xline16_seismic']
    # test_Xline16_seismic = torch.from_numpy(test_Xline16_seismic)
    # test_Xline16_seismic = test_Xline16_seismic.float()
    # test_Xline16_seismic = test_Xline16_seismic.to(device)

    # 加载训练用地震数据3
    dataFile7 = './SMI_data/Xline76_seismic.mat'
    Xline76_seismic = scipio.loadmat(dataFile7)
    test_Xline76_seismic = Xline76_seismic['Xline76_seismic']
    # test_Xline76_seismic = torch.from_numpy(test_Xline76_seismic)
    # test_Xline76_seismic = test_Xline76_seismic.float()
    # test_Xline76_seismic = test_Xline76_seismic.to(device)
    # 加载训练用地震数据4
    dataFile8 = './SMI_data/Inline26_seismic.mat'
    Inline26_seismic = scipio.loadmat(dataFile8)
    test_Inline26_seismic = Inline26_seismic['Inline26_seismic']
    # 加载训练用地震数据4
    dataFile9 = './SMI_data/Inline99_seismic.mat'
    Inline99_seismic = scipio.loadmat(dataFile9)
    test_Inline99_seismic = Inline99_seismic['Inline99_seismic']
    # 加载子波
    waveFile = 'D://Python_project/Predict_Impedence_SMI/SMI_data/wavelet_1_1.mat'
    wavel = scipio.loadmat(waveFile)
    wavele = wavel['wavelet_1_1']
    wavelet = torch.from_numpy(wavele)
    wavelet = wavelet.float()
    wavelet = wavelet.view(wavelet.size(1))
    wavelet = wavelet.to(device)


def pre_trained(judge):

    writer = SummaryWriter(log_dir='./loss/pre_train_loss_model1/pre_train_loss_SMI_2020_07_28_0%d' % date_num)

    if judge == 0:
        model = BSsequential_net_D().to(device)
        model.apply(weights_init)
        temp = 10000000000000
        epoch_num = 1
    else:
        model = BSsequential_net_D().to(device)
        mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_SMI_2020_07_28_02.pth'
        model.load_state_dict(torch.load(mode_patch))
        temp = 10000000000000
        epoch_num = 1
        # path_temp = './Temporary_parameters/pre_temp_model1.mat'
        # temp = scipio.loadmat(path_temp)
        # temp = temp['temp'].item()
        # path_epoch = './Temporary_parameters/pre_epoch_num_model1.mat'
        # epoch_num = scipio.loadmat(path_epoch)
        # epoch_num = epoch_num['epoch_num'].item()+1
    if is_consistent == 0:
        temp_wellnumber = np.zeros(0)
    else:
        is_path = 'D://Python_project/Predict_Impedence_hu2/SMI_out/map_number_2020_05_18_07.mat'
        Random_path = scipio.loadmat(is_path)
        temp_wellnumber = Random_path['temp_wellnumber']
    count = 0  # 检验网络权重是否变化的计数器
    lr = 0.001  # 学习步长
    for epoch in range(epoch_num, EPOCHS+1):
        print(epoch, count)
        temp_weight = model.fc60.weight   # 检验网络权重是否变化的初始网络参数
        temp_a = torch.sum(temp_weight.data)
        # print(temp_weight)
        # temp_a = torch.sum(temp_weight)
        # print(a)

        if np.mod(epoch + 1, 10) == 0:
            lr = lr * 0.99
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # if is_consistent == 1:
        #     trace_number = np.int(map_xline[0, epoch-1]*142+map_inline[0, epoch-1])
        # else:
        #     temp_1 = np.random.randint(0, 29, 1)
        #     temp_2 = np.random.randint(0, 22, 1)
        #     trace_number = temp_2*5*142+temp_1*5
        #     map_xline = np.append(map_xline, temp_2 * 5)
        #     map_inline = np.append(map_inline, temp_1 * 5)
        trace_number = np.random.randint(0, 104, 1)
        temp_wellnumber = np.append(temp_wellnumber, trace_number)
        # print(trace_number)

        # 计算相关系数
        coef_seismic = np.zeros((105, train_well_seismic.shape[1]))
        coef_seismic[0, :] = train_well_seismic[trace_number, :]
        coef_seismic[1:105, :] = train_well_seismic[:, :]
        temp_coef = np.corrcoef(coef_seismic)

        # 优选出相关系数大于阈值并且半径范围内的井
        tempval_1 = np.zeros(0)
        temp_train_well_1 = np.zeros(0)
        temp_train_well_seisic_1 = np.zeros(0)
        absCORcoef = np.abs(temp_coef[0, 1:105])
        if which_choose_well == 1:
            num = 0
            for k in range(0, 104):
                if absCORcoef[k] > coefval:
                    # 井数据的坐标
                    wellxline = Xline_Inline_number[0, k]
                    wellinline = Xline_Inline_number[1, k]
                    # 目标地震数据的坐标
                    seismicinline = np.mod(trace_number + 1, 142)
                    seismicxline = (trace_number + 1 - seismicinline) / 142 + 1
                    R = np.sqrt(
                        (seismicxline - wellxline) * (seismicxline - wellxline) + (seismicinline - wellinline) * (
                                seismicinline - wellinline))
                    if R < Rval:
                        tempval_1 = np.append(tempval_1, absCORcoef[k])
                        temp_train_well_1 = np.append(temp_train_well_1, train_well[k, :])
                        temp_train_well_seisic_1 = np.append(temp_train_well_seisic_1, train_well_seismic[k, :])
                        num = num + 1

            temp_train_well = np.zeros(0)
            temp_train_well_seisic = np.zeros(0)
            if num < num_well:
                num = num_well
                tempval = np.zeros(0)
                for max_num in range(0, num):
                    temp_tempval = max(absCORcoef)
                    tempval = np.append(tempval, temp_tempval)
                    for max_num2 in range(0, 104):
                        if temp_tempval == absCORcoef[max_num2]:
                            absCORcoef[max_num2] = 0
                            temp_train_well = np.append(temp_train_well, train_well[max_num2, :])
                            temp_train_well_seisic = np.append(temp_train_well_seisic, train_well_seismic[max_num2, :])
            else:
                tempval = np.zeros(0)
                temp_train_well_1 = torch.from_numpy(temp_train_well_1)
                temp_train_well_1 = temp_train_well_1.view(num, -1)
                temp_train_well_1 = temp_train_well_1.cpu().detach().numpy()
                temp_train_well_seisic_1 = torch.from_numpy(temp_train_well_seisic_1)
                temp_train_well_seisic_1 = temp_train_well_seisic_1.view(num, -1)
                temp_train_well_seisic_1 = temp_train_well_seisic_1.cpu().detach().numpy()
                for max_num in range(0, num_well):
                    temp_tempval = max(tempval_1)
                    tempval = np.append(tempval, temp_tempval)
                    for max_num2 in range(0, num):
                        if temp_tempval == tempval_1[max_num2]:
                            tempval_1[max_num2] = 0
                            temp_train_well = np.append(temp_train_well, temp_train_well_1[max_num2, :])
                            temp_train_well_seisic = np.append(temp_train_well_seisic,
                                                               temp_train_well_seisic_1[max_num2, :])
        else:
            num = num_well
            tempval = np.zeros(0)
            temp_train_well = np.zeros(0)
            temp_train_well_seisic = np.zeros(0)
            for max_num in range(0, num):
                temp_tempval = max(absCORcoef)
                tempval = np.append(tempval, temp_tempval)
                for max_num2 in range(0, 104):
                    if temp_tempval == absCORcoef[max_num2]:
                        absCORcoef[max_num2] = 0
                        temp_train_well = np.append(temp_train_well, train_well[max_num2, :])
                        temp_train_well_seisic = np.append(temp_train_well_seisic, train_well_seismic[max_num2, :])

        num = num_well
        maxval = max(tempval)
        minval = min(tempval)
        max_minlen = maxval - minval
        tempval = (tempval - minval) / max_minlen
        valsum = sum(tempval)
        tempval = tempval / valsum

        tempval = torch.from_numpy(tempval)
        tempval = tempval.view(1, -1)
        tempval = tempval.float()
        tempval = tempval.to(device)

        temp_train_well = torch.from_numpy(temp_train_well)
        temp_train_well = temp_train_well.view(num, -1)
        temp_train_well = temp_train_well.float()
        # temp_train_well = temp_train_well.to(device)
        # temp_train_well = temp_train_well.view(num, -1)

        # temp_train_well_seisic = torch.from_numpy(temp_train_well_seisic)
        # temp_train_well_seisic = temp_train_well_seisic.float()
        # temp_train_well_seisic = temp_train_well_seisic.to(device)
        # temp_train_well_seisic = temp_train_well_seisic.view(num, -1)
        # temp_seismic = torch.from_numpy(train1_75_seismic[trace_number, :])
        # temp_seismic = temp_seismic.float()
        # temp_seismic = temp_seismic.to(device)
        # temp_seismic = temp_seismic.view(1, -1)

        temp_lable = torch.from_numpy(train_well[trace_number, :])
        temp_lable = temp_lable.float()
        # temp_lable = temp_lable.to(device)
        temp_lable = temp_lable.view(1, -1)
        # for rand in range(0, 60 - BATCH_LEN + 1):
        for num_rand in range(0, number):
            rand = np.random.randint(0, 60 - BATCH_LEN + 1, 1)
            temp_train_seismic = train_well_seismic[trace_number, rand[0]:rand[0] + BATCH_LEN]
            temp_train_seismic = torch.from_numpy(temp_train_seismic)
            temp_train_seismic = temp_train_seismic.float()
            temp_train_seismic = temp_train_seismic.to(device)
            temp_train_seismic = temp_train_seismic.view(1, -1)

            # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入

            train_dataset = MyDataset2(temp_train_well[:, rand[0]:rand[0] + BATCH_LEN], temp_lable[:, rand[0]:rand[0] + BATCH_LEN])
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True, drop_last=False)
            epoch_loss = []

            for itr, (train_dt, train_lable) in enumerate(train_dataloader):
                train_dt, train_lable = train_dt.to(device), train_lable.to(device)
                train_dt = train_dt.float()
                train_lable = train_lable.float()

                model.train()
                optimizer.zero_grad()
                output = model(train_dt, temp_train_seismic)

                if is_synseismic == 1:
                    syn_seismic = syn_seismic_fun2(output, wavelet)
                    syn_seismic = syn_seismic.float()
                    loss = F.mse_loss(syn_seismic, temp_train_seismic) + F.mse_loss(output, train_lable)
                else:
                    loss = F.mse_loss(output, train_lable)

                loss.backward()
                optimizer.step()

                # print(model.conv1.weight)
                # print(model.conv2.weight)
                # print(model.lstm.weight)
                # print(model.fc1.weight.data[:, 0])

                epoch_loss.append(loss.item())

        temp_b = torch.sum(model.fc60.weight.data)
        # print(b)
        if temp_a == temp_b:
            count = count + 1
        else:
            count = 0
        if count > 50:
            break

        epoch_loss = np.sum(np.array(epoch_loss))
        writer.add_scalar('Train/MSE', epoch_loss, epoch)
        epoch_num = epoch
        print('Train set: Average loss: {:.15f}'.format(epoch_loss))
        if epoch_loss < temp:
            path = './model_file/pre_trained_network_model_model1/pre_trained_network_model_SMI_2020_07_28_0%d.pth' % date_num
            torch.save(model.state_dict(), path)
        path_temp = './Temporary_parameters/pre_temp_model1.mat'
        path_epoch = './Temporary_parameters/pre_epoch_num_model1.mat'
        scipio.savemat(path_temp, {'temp': temp})
        scipio.savemat(path_epoch, {'epoch_num': epoch_num})
    if is_consistent == 0:
        pathmat = './SMI_out/map_number_2020_07_28_0%d.mat' % date_num
        scipio.savemat(pathmat, {'temp_wellnumber': temp_wellnumber})
    writer.add_graph(model, (train_dt, temp_train_seismic))
    writer.close()


def tested():

    impedance_xline16 = np.zeros((test_Xline16_seismic.shape[0], test_Xline16_seismic.shape[1]))

    model = BSsequential_net_D().to(device)
    mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_SMI_2020_07_28_0%d.pth' % date_num
    model.load_state_dict(torch.load(mode_patch))
    for trace_number in range(0, test_Xline16_seismic.shape[0]):
        print(trace_number)

        # 计算相关系数
        coef_seismic = np.zeros((105, Xline1_110_label_impedance.shape[1]))
        coef_seismic[0, :] = test_Xline16_seismic[trace_number, :]
        coef_seismic[1:105, :] = train_well_seismic[:, :]
        temp_coef = np.corrcoef(coef_seismic)

        # 优选出相关系数大于阈值并且半径范围内的井
        tempval_1 = np.zeros(0)
        temp_train_well_1 = np.zeros(0)
        temp_train_well_seisic_1 = np.zeros(0)
        absCORcoef = np.abs(temp_coef[0, 1:105])
        if which_choose_well == 1:
            num = 0
            for k in range(0, 104):
                if absCORcoef[k] > coefval:
                    # 井数据的坐标
                    wellxline = Xline_Inline_number[0, k]
                    wellinline = Xline_Inline_number[1, k]
                    # 目标地震数据的坐标
                    seismicinline = np.mod(trace_number + 1, 142)
                    seismicxline = (trace_number + 1 - seismicinline) / 142 + 1
                    R = np.sqrt(
                        (seismicxline - wellxline) * (seismicxline - wellxline) + (seismicinline - wellinline) * (
                                seismicinline - wellinline))
                    if R < Rval:
                        tempval_1 = np.append(tempval_1, absCORcoef[k])
                        temp_train_well_1 = np.append(temp_train_well_1, train_well[k, :])
                        temp_train_well_seisic_1 = np.append(temp_train_well_seisic_1, train_well_seismic[k, :])
                        num = num + 1

            temp_train_well = np.zeros(0)
            temp_train_well_seisic = np.zeros(0)
            if num < num_well:
                num = num_well
                tempval = np.zeros(0)
                for max_num in range(0, num):
                    temp_tempval = max(absCORcoef)
                    tempval = np.append(tempval, temp_tempval)
                    for max_num2 in range(0, 104):
                        if temp_tempval == absCORcoef[max_num2]:
                            absCORcoef[max_num2] = 0
                            temp_train_well = np.append(temp_train_well, train_well[max_num2, :])
                            temp_train_well_seisic = np.append(temp_train_well_seisic, train_well_seismic[max_num2, :])
            else:
                tempval = np.zeros(0)
                temp_train_well_1 = torch.from_numpy(temp_train_well_1)
                temp_train_well_1 = temp_train_well_1.view(num, -1)
                temp_train_well_1 = temp_train_well_1.cpu().detach().numpy()
                temp_train_well_seisic_1 = torch.from_numpy(temp_train_well_seisic_1)
                temp_train_well_seisic_1 = temp_train_well_seisic_1.view(num, -1)
                temp_train_well_seisic_1 = temp_train_well_seisic_1.cpu().detach().numpy()
                for max_num in range(0, num_well):
                    temp_tempval = max(tempval_1)
                    tempval = np.append(tempval, temp_tempval)
                    for max_num2 in range(0, num):
                        if temp_tempval == tempval_1[max_num2]:
                            tempval_1[max_num2] = 0
                            temp_train_well = np.append(temp_train_well, temp_train_well_1[max_num2, :])
                            temp_train_well_seisic = np.append(temp_train_well_seisic,
                                                               temp_train_well_seisic_1[max_num2, :])
        else:
            num = num_well
            tempval = np.zeros(0)
            temp_train_well = np.zeros(0)
            temp_train_well_seisic = np.zeros(0)
            for max_num in range(0, num):
                temp_tempval = max(absCORcoef)
                tempval = np.append(tempval, temp_tempval)
                for max_num2 in range(0, 104):
                    if temp_tempval == absCORcoef[max_num2]:
                        absCORcoef[max_num2] = 0
                        temp_train_well = np.append(temp_train_well, train_well[max_num2, :])
                        temp_train_well_seisic = np.append(temp_train_well_seisic, train_well_seismic[max_num2, :])

        num = num_well
        maxval = max(tempval)
        minval = min(tempval)
        max_minlen = maxval - minval
        tempval = (tempval - minval) / max_minlen
        valsum = sum(tempval)
        tempval = tempval / valsum

        tempval = torch.from_numpy(tempval)
        tempval = tempval.view(1, -1)
        tempval = tempval.float()
        tempval = tempval.to(device)

        temp_train_well = torch.from_numpy(temp_train_well)
        temp_train_well = temp_train_well.view(num, -1)

        temp_train_well = temp_train_well.float()
        # temp_train_well = temp_train_well.to(device)

        temp_lable = torch.from_numpy(test_Xline16_seismic[trace_number, :])
        temp_lable = temp_lable.float()
        # temp_lable = temp_lable.to(device)
        temp_lable = temp_lable.view(1, -1)

        for num_batchlen in range(0, number):
            # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入
            train_dataset = MyDataset2(temp_train_well[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)],
                                       temp_lable[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)])
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True,
                                          drop_last=False)

            temp_seismic = test_Xline16_seismic[trace_number, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)]
            temp_seismic = torch.from_numpy(temp_seismic)
            temp_seismic = temp_seismic.float()
            temp_seismic = temp_seismic.to(device)
            temp_seismic = temp_seismic.view(1, -1)

            for itr, (train_dt, train_lable) in enumerate(train_dataloader):
                train_dt, train_lable = train_dt.to(device), train_lable.to(device)
                train_dt = train_dt.float()

                output = model(train_dt, temp_seismic)
                # train_dt = train_dt.view(num_well, -1)
                # output = tempval.mm(train_dt)
                np_output = output.cpu().detach().numpy()
                impedance_xline16[(trace_number * BATCH_SIZE):((trace_number + 1) * BATCH_SIZE), (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)] = np_output

    pathmat = './SMI_out/test_X16_Impedance_model1_2020_07_28_0%d.mat' % date_num
    scipio.savemat(pathmat, {'test_X16_Impedance_model1_2020_07_28_0%d' % date_num: impedance_xline16})


def tested2():
    impedance_xline76 = np.zeros((test_Xline76_seismic.shape[0], test_Xline76_seismic.shape[1]))

    model = BSsequential_net_D().to(device)
    mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_SMI_2020_07_28_0%d.pth' % date_num
    model.load_state_dict(torch.load(mode_patch))
    for trace_number in range(0, test_Xline76_seismic.shape[0]):
        print(trace_number)

        # 计算相关系数
        coef_seismic = np.zeros((105, Xline1_110_label_impedance.shape[1]))
        coef_seismic[0, :] = test_Xline76_seismic[trace_number, :]
        coef_seismic[1:105, :] = train_well_seismic[:, :]
        temp_coef = np.corrcoef(coef_seismic)

        # 优选出相关系数大于阈值并且半径范围内的井
        tempval_1 = np.zeros(0)
        temp_train_well_1 = np.zeros(0)
        temp_train_well_seisic_1 = np.zeros(0)
        absCORcoef = np.abs(temp_coef[0, 1:105])
        if which_choose_well == 1:
            num = 0
            for k in range(0, 104):
                if absCORcoef[k] > coefval:
                    # 井数据的坐标
                    wellxline = Xline_Inline_number[0, k]
                    wellinline = Xline_Inline_number[1, k]
                    # 目标地震数据的坐标
                    seismicinline = np.mod(trace_number + 1, 142)
                    seismicxline = (trace_number + 1 - seismicinline) / 142 + 1
                    R = np.sqrt(
                        (seismicxline - wellxline) * (seismicxline - wellxline) + (seismicinline - wellinline) * (
                                seismicinline - wellinline))
                    if R < Rval:
                        tempval_1 = np.append(tempval_1, absCORcoef[k])
                        temp_train_well_1 = np.append(temp_train_well_1, train_well[k, :])
                        temp_train_well_seisic_1 = np.append(temp_train_well_seisic_1, train_well_seismic[k, :])
                        num = num + 1

            temp_train_well = np.zeros(0)
            temp_train_well_seisic = np.zeros(0)
            if num < num_well:
                num = num_well
                tempval = np.zeros(0)
                for max_num in range(0, num):
                    temp_tempval = max(absCORcoef)
                    tempval = np.append(tempval, temp_tempval)
                    for max_num2 in range(0, 104):
                        if temp_tempval == absCORcoef[max_num2]:
                            absCORcoef[max_num2] = 0
                            temp_train_well = np.append(temp_train_well, train_well[max_num2, :])
                            temp_train_well_seisic = np.append(temp_train_well_seisic,
                                                               train_well_seismic[max_num2, :])
            else:
                tempval = np.zeros(0)
                temp_train_well_1 = torch.from_numpy(temp_train_well_1)
                temp_train_well_1 = temp_train_well_1.view(num, -1)
                temp_train_well_1 = temp_train_well_1.cpu().detach().numpy()
                temp_train_well_seisic_1 = torch.from_numpy(temp_train_well_seisic_1)
                temp_train_well_seisic_1 = temp_train_well_seisic_1.view(num, -1)
                temp_train_well_seisic_1 = temp_train_well_seisic_1.cpu().detach().numpy()
                for max_num in range(0, num_well):
                    temp_tempval = max(tempval_1)
                    tempval = np.append(tempval, temp_tempval)
                    for max_num2 in range(0, num):
                        if temp_tempval == tempval_1[max_num2]:
                            tempval_1[max_num2] = 0
                            temp_train_well = np.append(temp_train_well, temp_train_well_1[max_num2, :])
                            temp_train_well_seisic = np.append(temp_train_well_seisic,
                                                               temp_train_well_seisic_1[max_num2, :])
        else:
            num = num_well
            tempval = np.zeros(0)
            temp_train_well = np.zeros(0)
            temp_train_well_seisic = np.zeros(0)
            for max_num in range(0, num):
                temp_tempval = max(absCORcoef)
                tempval = np.append(tempval, temp_tempval)
                for max_num2 in range(0, 104):
                    if temp_tempval == absCORcoef[max_num2]:
                        absCORcoef[max_num2] = 0
                        temp_train_well = np.append(temp_train_well, train_well[max_num2, :])
                        temp_train_well_seisic = np.append(temp_train_well_seisic, train_well_seismic[max_num2, :])

        num = num_well
        maxval = max(tempval)
        minval = min(tempval)
        max_minlen = maxval - minval
        tempval = (tempval - minval) / max_minlen
        valsum = sum(tempval)
        tempval = tempval / valsum

        tempval = torch.from_numpy(tempval)
        tempval = tempval.view(1, -1)
        tempval = tempval.float()
        tempval = tempval.to(device)

        temp_train_well = torch.from_numpy(temp_train_well)
        temp_train_well = temp_train_well.view(num, -1)
        temp_train_well = temp_train_well.float()
        # temp_train_well = temp_train_well.to(device)

        temp_lable = torch.from_numpy(test_Xline76_seismic[trace_number, :])
        temp_lable = temp_lable.float()
        # temp_lable = temp_lable.to(device)
        temp_lable = temp_lable.view(1, -1)
        for num_batchlen in range(0, number):
            # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入
            train_dataset = MyDataset2(
                temp_train_well[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)],
                temp_lable[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)])
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True,
                                          drop_last=False)

            temp_seismic = test_Xline76_seismic[trace_number, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)]
            temp_seismic = torch.from_numpy(temp_seismic)
            temp_seismic = temp_seismic.float()
            temp_seismic = temp_seismic.to(device)
            temp_seismic = temp_seismic.view(1, -1)

            for itr, (train_dt, train_lable) in enumerate(train_dataloader):
                train_dt, train_lable = train_dt.to(device), train_lable.to(device)
                train_dt = train_dt.float()

                output = model(train_dt, temp_seismic)
                # train_dt = train_dt.view(num_well, -1)
                # output = tempval.mm(train_dt)
                np_output = output.cpu().detach().numpy()
                impedance_xline76[(trace_number * BATCH_SIZE):((trace_number + 1) * BATCH_SIZE),
                (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)] = np_output

    pathmat = './SMI_out/pre_X76_Impedance_model1_2020_07_28_0%d.mat' % date_num
    scipio.savemat(pathmat, {'pre_X76_Impedance_model1_2020_07_28_0%d' % date_num: impedance_xline76})


def tested3():
    impedance_inline26 = np.zeros((test_Inline26_seismic.shape[0], test_Inline26_seismic.shape[1]))

    model = BSsequential_net_D().to(device)
    mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_SMI_2020_07_28_0%d.pth' % date_num
    model.load_state_dict(torch.load(mode_patch))
    for trace_number in range(0, test_Inline26_seismic.shape[0]):
        print(trace_number)

        # 计算相关系数
        coef_seismic = np.zeros((105, Xline1_110_label_impedance.shape[1]))
        coef_seismic[0, :] = test_Inline26_seismic[trace_number, :]
        coef_seismic[1:105, :] = train_well_seismic[:, :]
        temp_coef = np.corrcoef(coef_seismic)

        # 优选出相关系数大于阈值并且半径范围内的井
        tempval_1 = np.zeros(0)
        temp_train_well_1 = np.zeros(0)
        temp_train_well_seisic_1 = np.zeros(0)
        absCORcoef = np.abs(temp_coef[0, 1:105])
        if which_choose_well == 1:
            num = 0
            for k in range(0, 104):
                if absCORcoef[k] > coefval:
                    # 井数据的坐标
                    wellxline = Xline_Inline_number[0, k]
                    wellinline = Xline_Inline_number[1, k]
                    # 目标地震数据的坐标
                    seismicinline = np.mod(trace_number + 1, 142)
                    seismicxline = (trace_number + 1 - seismicinline) / 142 + 1
                    R = np.sqrt(
                        (seismicxline - wellxline) * (seismicxline - wellxline) + (seismicinline - wellinline) * (
                                seismicinline - wellinline))
                    if R < Rval:
                        tempval_1 = np.append(tempval_1, absCORcoef[k])
                        temp_train_well_1 = np.append(temp_train_well_1, train_well[k, :])
                        temp_train_well_seisic_1 = np.append(temp_train_well_seisic_1, train_well_seismic[k, :])
                        num = num + 1

            temp_train_well = np.zeros(0)
            temp_train_well_seisic = np.zeros(0)
            if num < num_well:
                num = num_well
                tempval = np.zeros(0)
                for max_num in range(0, num):
                    temp_tempval = max(absCORcoef)
                    tempval = np.append(tempval, temp_tempval)
                    for max_num2 in range(0, 104):
                        if temp_tempval == absCORcoef[max_num2]:
                            absCORcoef[max_num2] = 0
                            temp_train_well = np.append(temp_train_well, train_well[max_num2, :])
                            temp_train_well_seisic = np.append(temp_train_well_seisic, train_well_seismic[max_num2, :])
            else:
                tempval = np.zeros(0)
                temp_train_well_1 = torch.from_numpy(temp_train_well_1)
                temp_train_well_1 = temp_train_well_1.view(num, -1)
                temp_train_well_1 = temp_train_well_1.cpu().detach().numpy()
                temp_train_well_seisic_1 = torch.from_numpy(temp_train_well_seisic_1)
                temp_train_well_seisic_1 = temp_train_well_seisic_1.view(num, -1)
                temp_train_well_seisic_1 = temp_train_well_seisic_1.cpu().detach().numpy()
                for max_num in range(0, num_well):
                    temp_tempval = max(tempval_1)
                    tempval = np.append(tempval, temp_tempval)
                    for max_num2 in range(0, num):
                        if temp_tempval == tempval_1[max_num2]:
                            tempval_1[max_num2] = 0
                            temp_train_well = np.append(temp_train_well, temp_train_well_1[max_num2, :])
                            temp_train_well_seisic = np.append(temp_train_well_seisic,
                                                               temp_train_well_seisic_1[max_num2, :])
        else:
            num = num_well
            tempval = np.zeros(0)
            temp_train_well = np.zeros(0)
            temp_train_well_seisic = np.zeros(0)
            for max_num in range(0, num):
                temp_tempval = max(absCORcoef)
                tempval = np.append(tempval, temp_tempval)
                for max_num2 in range(0, 104):
                    if temp_tempval == absCORcoef[max_num2]:
                        absCORcoef[max_num2] = 0
                        temp_train_well = np.append(temp_train_well, train_well[max_num2, :])
                        temp_train_well_seisic = np.append(temp_train_well_seisic, train_well_seismic[max_num2, :])

        num = num_well
        maxval = max(tempval)
        minval = min(tempval)
        max_minlen = maxval - minval
        tempval = (tempval - minval) / max_minlen
        valsum = sum(tempval)
        tempval = tempval / valsum

        tempval = torch.from_numpy(tempval)
        tempval = tempval.view(1, -1)
        tempval = tempval.float()
        tempval = tempval.to(device)

        temp_train_well = torch.from_numpy(temp_train_well)
        temp_train_well = temp_train_well.view(num, -1)
        temp_train_well = temp_train_well.float()
        # temp_train_well = temp_train_well.to(device)

        temp_lable = torch.from_numpy(test_Inline26_seismic[trace_number, :])
        temp_lable = temp_lable.float()
        # temp_lable = temp_lable.to(device)
        temp_lable = temp_lable.view(1, -1)
        for num_batchlen in range(0, number):
            # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入
            train_dataset = MyDataset2(temp_train_well[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)],
                                       temp_lable[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)])
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True,
                                          drop_last=False)

            temp_seismic = test_Inline26_seismic[trace_number, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)]
            temp_seismic = torch.from_numpy(temp_seismic)
            temp_seismic = temp_seismic.float()
            temp_seismic = temp_seismic.to(device)
            temp_seismic = temp_seismic.view(1, -1)

            for itr, (train_dt, train_lable) in enumerate(train_dataloader):
                train_dt, train_lable = train_dt.to(device), train_lable.to(device)
                train_dt = train_dt.float()

                output = model(train_dt, temp_seismic)
                # train_dt = train_dt.view(num_well, -1)
                # output = tempval.mm(train_dt)
                np_output = output.cpu().detach().numpy()
                impedance_inline26[(trace_number * BATCH_SIZE):((trace_number + 1) * BATCH_SIZE), (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)] = np_output

    pathmat = './SMI_out/pre_In26_Impedance_model1_2020_07_28_0%d.mat' % date_num
    scipio.savemat(pathmat, {'pre_In26_Impedance_model1_2020_07_28_0%d' % date_num: impedance_inline26})


def tested4():
    impedance_inline99 = np.zeros((test_Inline99_seismic.shape[0], test_Inline99_seismic.shape[1]))

    model = BSsequential_net_D().to(device)
    mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_SMI_2020_07_28_0%d.pth' % date_num
    model.load_state_dict(torch.load(mode_patch))
    # num_params = 0
    # for param in model.parameters():
    #     num_params += param.numel()
    # print(num_params)
    for trace_number in range(0, test_Inline99_seismic.shape[0]):
        print(trace_number)

        # 计算相关系数
        coef_seismic = np.zeros((105, Xline1_110_label_impedance.shape[1]))
        coef_seismic[0, :] = test_Inline99_seismic[trace_number, :]
        coef_seismic[1:105, :] = train_well_seismic[:, :]
        temp_coef = np.corrcoef(coef_seismic)

        # 优选出相关系数大于阈值并且半径范围内的井
        tempval_1 = np.zeros(0)
        temp_train_well_1 = np.zeros(0)
        temp_train_well_seisic_1 = np.zeros(0)
        absCORcoef = np.abs(temp_coef[0, 1:105])
        if which_choose_well == 1:
            num = 0
            for k in range(0, 104):
                if absCORcoef[k] > coefval:
                    # 井数据的坐标
                    wellxline = Xline_Inline_number[0, k]
                    wellinline = Xline_Inline_number[1, k]
                    # 目标地震数据的坐标
                    seismicinline = np.mod(trace_number + 1, 142)
                    seismicxline = (trace_number + 1 - seismicinline) / 142 + 1
                    R = np.sqrt(
                        (seismicxline - wellxline) * (seismicxline - wellxline) + (seismicinline - wellinline) * (
                                seismicinline - wellinline))
                    if R < Rval:
                        tempval_1 = np.append(tempval_1, absCORcoef[k])
                        temp_train_well_1 = np.append(temp_train_well_1, train_well[k, :])
                        temp_train_well_seisic_1 = np.append(temp_train_well_seisic_1, train_well_seismic[k, :])
                        num = num + 1

            temp_train_well = np.zeros(0)
            temp_train_well_seisic = np.zeros(0)
            if num < num_well:
                num = num_well
                tempval = np.zeros(0)
                for max_num in range(0, num):
                    temp_tempval = max(absCORcoef)
                    tempval = np.append(tempval, temp_tempval)
                    for max_num2 in range(0, 104):
                        if temp_tempval == absCORcoef[max_num2]:
                            absCORcoef[max_num2] = 0
                            temp_train_well = np.append(temp_train_well, train_well[max_num2, :])
                            temp_train_well_seisic = np.append(temp_train_well_seisic, train_well_seismic[max_num2, :])
            else:
                tempval = np.zeros(0)
                temp_train_well_1 = torch.from_numpy(temp_train_well_1)
                temp_train_well_1 = temp_train_well_1.view(num, -1)
                temp_train_well_1 = temp_train_well_1.cpu().detach().numpy()
                temp_train_well_seisic_1 = torch.from_numpy(temp_train_well_seisic_1)
                temp_train_well_seisic_1 = temp_train_well_seisic_1.view(num, -1)
                temp_train_well_seisic_1 = temp_train_well_seisic_1.cpu().detach().numpy()
                for max_num in range(0, num_well):
                    temp_tempval = max(tempval_1)
                    tempval = np.append(tempval, temp_tempval)
                    for max_num2 in range(0, num):
                        if temp_tempval == tempval_1[max_num2]:
                            tempval_1[max_num2] = 0
                            temp_train_well = np.append(temp_train_well, temp_train_well_1[max_num2, :])
                            temp_train_well_seisic = np.append(temp_train_well_seisic,
                                                               temp_train_well_seisic_1[max_num2, :])
        else:
            num = num_well
            tempval = np.zeros(0)
            temp_train_well = np.zeros(0)
            temp_train_well_seisic = np.zeros(0)
            for max_num in range(0, num):
                temp_tempval = max(absCORcoef)
                tempval = np.append(tempval, temp_tempval)
                for max_num2 in range(0, 104):
                    if temp_tempval == absCORcoef[max_num2]:
                        absCORcoef[max_num2] = 0
                        temp_train_well = np.append(temp_train_well, train_well[max_num2, :])
                        temp_train_well_seisic = np.append(temp_train_well_seisic, train_well_seismic[max_num2, :])

        num = num_well
        maxval = max(tempval)
        minval = min(tempval)
        max_minlen = maxval - minval
        tempval = (tempval - minval) / max_minlen
        valsum = sum(tempval)
        tempval = tempval / valsum

        tempval = torch.from_numpy(tempval)
        tempval = tempval.view(1, -1)
        tempval = tempval.float()
        tempval = tempval.to(device)

        temp_train_well = torch.from_numpy(temp_train_well)
        temp_train_well = temp_train_well.view(num, -1)
        temp_train_well = temp_train_well.float()
        # temp_train_well = temp_train_well.to(device)

        temp_lable = torch.from_numpy(test_Inline99_seismic[trace_number, :])
        temp_lable = temp_lable.float()
        # temp_lable = temp_lable.to(device)
        temp_lable = temp_lable.view(1, -1)
        for num_batchlen in range(0, number):
            # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入
            train_dataset = MyDataset2(temp_train_well[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)],
                                       temp_lable[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)])
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True,
                                          drop_last=False)

            temp_seismic = test_Inline99_seismic[trace_number, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)]
            temp_seismic = torch.from_numpy(temp_seismic)
            temp_seismic = temp_seismic.float()
            temp_seismic = temp_seismic.to(device)
            temp_seismic = temp_seismic.view(1, -1)

            for itr, (train_dt, train_lable) in enumerate(train_dataloader):
                train_dt, train_lable = train_dt.to(device), train_lable.to(device)
                train_dt = train_dt.float()

                output = model(train_dt, temp_seismic)
                # train_dt = train_dt.view(num_well, -1)
                # output = tempval.mm(train_dt)
                np_output = output.cpu().detach().numpy()
                impedance_inline99[(trace_number * BATCH_SIZE):((trace_number + 1) * BATCH_SIZE), (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)] = np_output

    pathmat = './SMI_out/pre_In99_Impedance_model1_2020_07_28_0%d.mat' % date_num
    scipio.savemat(pathmat, {'pre_In99_Impedance_model1_2020_07_28_0%d' % date_num: impedance_inline99})


if __name__ == '__main__':
    pre_trained(is_inherit_net)
    tested()
    tested2()
    tested3()
    tested4()
    # inversion(0)
    # tensorboard --logdir E:\Python_project\Predict_Impedence\loss\pre_train_loss_marmousi2019_12_10_11
