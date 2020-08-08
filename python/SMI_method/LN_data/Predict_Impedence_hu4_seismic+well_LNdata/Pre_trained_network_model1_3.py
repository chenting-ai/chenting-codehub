import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
import scipy.io as scipio
import numpy as np
import os
from tensorboardX import SummaryWriter
from net_and_data import MyDataset, MyDataset2, BSsequential_net, device, weights_init
from my_loss_function import my_mse_loss_fun, syn_seismic_fun2
#  输入为地震数据和井数据

# device = "cpu"
BATCH_SIZE = 64  # BATCH_SIZE大小
BATCH_LEN = 100  # BATCH_LEN长度
EPOCHS = 30  # 总共训练批次
num_well = 5  # 井数量

data_rate = 1  # 选用前50%的xline线训练
number = np.int(100/BATCH_LEN)  # 一道有几个batch长度
inline_num = 501
xline_num = 631
totall_number = xline_num * inline_num

is_consistent = 0  # 是否要确定的随机路径
is_synseismic = 0  # 是否增加合成地震记录约束
is_inherit_net = 0  # 是否继承之前的网络
which_choose_well = 0  # 第几种选井方式 0.选前num_well口，1.在阈值和半径范围内选前num_well口，

date_num = 1  # 今天第几次跑
# 加载数据
if 1:
    # 加载地震数据 seisic_well_number
    dataFile = './LN_data/seisic_well_number_%d.mat' % num_well
    seisic_well_number = scipio.loadmat(dataFile)
    seisic_well_number = seisic_well_number['seisic_well_number']
    # 加载训练用井旁地震数据
    dataFile1 = './LN_data/well_seismic.mat'
    well_seismic = scipio.loadmat(dataFile1)
    train_well_seismic = well_seismic['well_seismic']
    # 加载训练用测井数据
    dataFile2 = './LN_data/well_vp_vs.mat'
    well = scipio.loadmat(dataFile2)
    train_well = well['well_vp_vs']
    # 加载训练用地震数据
    dataFile3 = './LN_data/seismic.mat'
    seismic = scipio.loadmat(dataFile3)
    seismic = seismic['seismic']
    # 加载训练用标签数据
    dataFile4 = './LN_data/lable.mat'
    vp_vs_lable = scipio.loadmat(dataFile4)
    vp_vs_lable = vp_vs_lable['lable']
    # 加载子波
    waveFile = './LN_data/wavelet.mat'
    wavel = scipio.loadmat(waveFile)
    wavele = wavel['wavelet']
    wavelet = torch.from_numpy(wavele)
    wavelet = wavelet.float()
    wavelet = wavelet.view(wavelet.size(1))
    wavelet = wavelet.to(device)


def process_well_cor():
    seisic_well_number = np.zeros((totall_number, seismic.shape[1] + num_well))
    for trace_number in range(0, totall_number):
        print(trace_number, totall_number)

        # 计算相关系数
        coef_seismic = np.zeros((train_well.shape[0]+1, train_well.shape[1]))
        coef_seismic[0, :] = seismic[trace_number, :]
        coef_seismic[1:coef_seismic.shape[0], :] = train_well_seismic[:, :]
        temp_coef = np.corrcoef(coef_seismic)

        # 优选出相关系数大于阈值并且半径范围内的井
        tempval_1 = np.zeros(0)
        temp_well_num = np.zeros(0)
        well_num = np.zeros(0)
        absCORcoef = np.abs(temp_coef[0, 1:coef_seismic.shape[0]])
        if which_choose_well == 1:
            num = 0
            for k in range(0, train_well.shape[0]):
                if absCORcoef[k] > coefval:
                    # 井数据的坐标
                    wellxline = Xline_Inline_number[0, k]
                    wellinline = Xline_Inline_number[1, k]
                    # 目标地震数据的坐标
                    seismicinline = np.mod(trace_number + 1, xline_num)
                    seismicxline = (trace_number + 1 - seismicinline) / xline_num + 1
                    R = np.sqrt((seismicxline - wellxline) * (seismicxline - wellxline) + (seismicinline - wellinline) * (
                            seismicinline - wellinline))
                    if R < Rval:
                        tempval_1 = np.append(tempval_1, absCORcoef[k])
                        temp_well_num = np.append(temp_well_num, k)
                        num = num + 1

            if num < num_well:
                num = num_well
                for max_num in range(0, num):
                    temp_tempval = max(absCORcoef)
                    for max_num2 in range(0, train_well.shape[0]):
                        if temp_tempval == absCORcoef[max_num2]:
                            absCORcoef[max_num2] = 0
                            well_num = np.append(well_num, max_num2)
            else:
                for max_num in range(0, num_well):
                    temp_tempval = max(tempval_1)
                    for max_num2 in range(0, num):
                        if temp_tempval == tempval_1[max_num2]:
                            tempval_1[max_num2] = 0
                            well_num = np.append(well_num, temp_well_num[max_num2, :])
        else:
            num = num_well
            for max_num in range(0, num):
                temp_tempval = max(absCORcoef)
                for max_num2 in range(0, train_well.shape[0]):
                    if temp_tempval == absCORcoef[max_num2]:
                        absCORcoef[max_num2] = 0
                        well_num = np.append(well_num, max_num2)

        well_num = torch.from_numpy(well_num)
        well_num = well_num.view(1, -1)
        seisic_well_number[trace_number, 0:seismic.shape[1]] = seismic[trace_number, :]
        seisic_well_number[trace_number, seismic.shape[1]:seisic_well_number.shape[1]] = well_num

    pathmat = './LN_data/seisic_well_number_%d.mat' % num_well
    scipio.savemat(pathmat, {'seisic_well_number': seisic_well_number})


def pre_trained(judge):

    writer = SummaryWriter(log_dir='./loss/pre_train_loss_model1/pre_train_loss_LN_2020_08_08_0%d' % date_num)

    if judge == 0:
        model = BSsequential_net(BATCH_LEN, num_well).to(device)
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
        device1 = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        print(device1)
        # model.apply(weights_init)
        temp = 10000000000000
        epoch_num = 1
    else:
        model = BSsequential_net(BATCH_LEN, num_well).to(device)
        mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_LN_2020_08_07_0%d.pth' % date_num
        model.load_state_dict(torch.load(mode_patch))
        device1 = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        print(device1)
        temp = 10000000000000
        epoch_num = 50001
    if is_consistent == 0:
        map_xline = np.zeros(0)
        map_inline = np.zeros(0)
    else:
        is_path = './SMI_out/map_number_2020_05_14_06.mat'
        Random_path = scipio.loadmat(is_path)
        map_xline = Random_path['map_xline']
        map_inline = Random_path['map_inline']
    count = 0  # 检验网络权重是否变化的计数器
    lr = 0.001  # 学习步长
    temp_epoch = 1
    for epoch in range(epoch_num, EPOCHS+1):
        temp_weight = model.fc60.weight   # 检验网络权重是否变化的初始网络参数
        temp_a = torch.sum(temp_weight.data)
        # print(temp_weight)
        # temp_weight = model.lstm60
        # temp_a = torch.sum(temp_weight.weight_hh_l0.data) + torch.sum(temp_weight.weight_ih_l0.data)
        # print(a)

        lr = lr * 0.9
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for num_rand in range(0, number):
            rand = np.random.randint(0, train_well.shape[1] - BATCH_LEN + 1, 1)

            train_seisic_well_number = np.zeros((totall_number, BATCH_LEN + num_well))
            train_seisic_well_number[:, 0:BATCH_LEN] = seisic_well_number[:, rand[0]:rand[0] + BATCH_LEN]
            train_seisic_well_number[:, BATCH_LEN:BATCH_LEN + num_well] = seisic_well_number[:, seisic_well_number.shape[1]-num_well:seisic_well_number.shape[1]]
            train_seisic_well_number = torch.from_numpy(train_seisic_well_number)
            train_seisic_well_number = train_seisic_well_number.float()

            train_vp_vs_lable = torch.from_numpy(vp_vs_lable[:, rand[0]:rand[0] + BATCH_LEN])
            train_vp_vs_lable = train_vp_vs_lable.float()

            # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入

            train_dataset = MyDataset2(train_seisic_well_number, train_vp_vs_lable)
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=BATCH_SIZE, shuffle=True, drop_last=False)

            for itr, (train_dt, train_lable) in enumerate(train_dataloader):
                epoch_loss = []

                temp_train_well = train_well[train_dt[:, BATCH_LEN:train_dt.shape[1]].int(), rand[0]:rand[0] + BATCH_LEN]
                temp_train_well = torch.from_numpy(temp_train_well)
                train_dt, train_lable = train_dt[:, 0:BATCH_LEN].to(device), train_lable.to(device)
                temp_train_well = temp_train_well.to(device)
                train_dt = train_dt.float()
                train_lable = train_lable.float()
                temp_train_well = temp_train_well.float()

                model.train()
                optimizer.zero_grad()
                output = model(temp_train_well, train_dt)
                if is_synseismic == 1:
                    syn_seismic = syn_seismic_fun2(output, wavelet)
                    syn_seismic = syn_seismic.float()
                    loss = F.mse_loss(syn_seismic, temp_train_seismic) + F.mse_loss(output, train_lable)
                else:
                    loss = F.mse_loss(output, train_lable) + F.l1_loss(output, train_lable)

                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())

                temp_b = torch.sum(model.fc60.weight.data)
                # temp_b = torch.sum(model.lstm60.weight_hh_l0.data) + torch.sum(model.lstm60.weight_ih_l0.data)
                # print(b)
                if temp_a == temp_b:
                    count = count + 1
                else:
                    count = 0
                if count > 100:
                    break

                epoch_loss = np.sum(np.array(epoch_loss))
                writer.add_scalar('Train/MSE', epoch_loss, temp_epoch)
                temp_epoch = temp_epoch+1
                print("epoch-itr-count:", epoch, itr, count, 'Train set: Average loss: {:.15f}'.format(epoch_loss))
                path = './model_file/pre_trained_network_model_model1/pre_trained_network_model_LN_2020_08_08_0%d.pth' % date_num
                torch.save(model.state_dict(), path)
    # writer.add_graph(model, (train_dt, temp_train_seismic))
    writer.close()


def tested1():

    Inline1134_vp_vs = np.zeros((xline_num, train_well.shape[1]))
    Inline1134_seismic = seismic[(1134-750)*xline_num:(1135-750)*xline_num, :]

    model = BSsequential_net(BATCH_LEN, num_well).to(device)
    mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_LN_2020_08_08_0%d.pth' % date_num
    model.load_state_dict(torch.load(mode_patch))   # , map_location='cpu'
    for trace_number in range(0, xline_num):
        print(trace_number)

        # 计算相关系数
        coef_seismic = np.zeros((train_well.shape[0] + 1, train_well.shape[1]))
        coef_seismic[0, :] = Inline1134_seismic[trace_number, :]
        coef_seismic[1:coef_seismic.shape[0], :] = train_well_seismic[:, :]
        temp_coef = np.corrcoef(coef_seismic)

        # 优选出相关系数大于阈值并且半径范围内的井
        tempval_1 = np.zeros(0)
        temp_train_well_1 = np.zeros(0)
        temp_train_well_seisic_1 = np.zeros(0)
        absCORcoef = np.abs(temp_coef[0, 1:coef_seismic.shape[0]])
        if which_choose_well == 1:
            num = 0
            for k in range(0, 15):
                if absCORcoef[k] > coefval:
                    # 井数据的坐标
                    wellxline = Xline_Inline_number[0, k]
                    wellinline = Xline_Inline_number[1, k]
                    # 目标地震数据的坐标
                    seismicinline = np.mod(trace_number + 1, xline_num)
                    seismicxline = (trace_number + 1 - seismicinline) / xline_num + 1
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
                    for max_num2 in range(0, 15):
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
                for max_num2 in range(0, train_well.shape[0]):
                    if temp_tempval == absCORcoef[max_num2]:
                        absCORcoef[max_num2] = 0
                        temp_train_well = np.append(temp_train_well, train_well[max_num2, :])
                        temp_train_well_seisic = np.append(temp_train_well_seisic, train_well_seismic[max_num2, :])

        num = num_well
        temp_train_well = torch.from_numpy(temp_train_well)
        temp_train_well = temp_train_well.view(num, -1)
        temp_train_well = temp_train_well.float()

        temp_seismic = Inline1134_seismic[trace_number, :]
        temp_seismic = torch.from_numpy(temp_seismic)
        temp_seismic = temp_seismic.float()
        temp_seismic = temp_seismic.view(1, -1)

        for num_batchlen in range(0, number):
            # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入
            train_dataset = MyDataset(temp_train_well[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)],
                                       temp_seismic[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)])
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True,
                                          drop_last=False)

            for itr, (train_dt, train_lable) in enumerate(train_dataloader):
                train_dt, train_lable = train_dt.to(device), train_lable.to(device)
                train_dt = train_dt.float()

                output = model(train_dt, train_lable)
                np_output = output.cpu().detach().numpy()
                Inline1134_vp_vs[(trace_number * BATCH_SIZE):((trace_number + 1) * BATCH_SIZE), (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)] = np_output

    pathmat = './LN_out/Inline1134_vp_vs_model1_2020_08_08_0%d.mat' % date_num
    scipio.savemat(pathmat, {'Inline1134_vp_vs_model1_2020_08_08_0%d' % date_num: Inline1134_vp_vs})


def tested2():

    Inline1250_vp_vs = np.zeros((xline_num, train_well.shape[1]))
    Inline1250_seismic = seismic[(1250-750)*xline_num:(1251-750)*xline_num, :]

    model = BSsequential_net(BATCH_LEN, num_well).to(device)
    mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_LN_2020_08_08_0%d.pth' % date_num
    model.load_state_dict(torch.load(mode_patch))
    for trace_number in range(0, xline_num):
        print(trace_number)

        # 计算相关系数
        coef_seismic = np.zeros((train_well.shape[0] + 1, train_well.shape[1]))
        coef_seismic[0, :] = Inline1250_seismic[trace_number, :]
        coef_seismic[1:coef_seismic.shape[0], :] = train_well_seismic[:, :]
        temp_coef = np.corrcoef(coef_seismic)

        # 优选出相关系数大于阈值并且半径范围内的井
        tempval_1 = np.zeros(0)
        temp_train_well_1 = np.zeros(0)
        temp_train_well_seisic_1 = np.zeros(0)
        absCORcoef = np.abs(temp_coef[0, 1:coef_seismic.shape[0]])
        if which_choose_well == 1:
            num = 0
            for k in range(0, 15):
                if absCORcoef[k] > coefval:
                    # 井数据的坐标
                    wellxline = Xline_Inline_number[0, k]
                    wellinline = Xline_Inline_number[1, k]
                    # 目标地震数据的坐标
                    seismicinline = np.mod(trace_number + 1, 631)
                    seismicxline = (trace_number + 1 - seismicinline) / 631 + 1
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
                    for max_num2 in range(0, 15):
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
                for max_num2 in range(0, train_well.shape[0]):
                    if temp_tempval == absCORcoef[max_num2]:
                        absCORcoef[max_num2] = 0
                        temp_train_well = np.append(temp_train_well, train_well[max_num2, :])
                        temp_train_well_seisic = np.append(temp_train_well_seisic, train_well_seismic[max_num2, :])

        num = num_well
        temp_train_well = torch.from_numpy(temp_train_well)
        temp_train_well = temp_train_well.view(num, -1)
        temp_train_well = temp_train_well.float()

        temp_seismic = Inline1250_seismic[trace_number, :]
        temp_seismic = torch.from_numpy(temp_seismic)
        temp_seismic = temp_seismic.float()
        temp_seismic = temp_seismic.view(1, -1)

        for num_batchlen in range(0, number):
            # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入
            train_dataset = MyDataset(temp_train_well[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)],
                                       temp_seismic[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)])
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True,
                                          drop_last=False)

            for itr, (train_dt, train_lable) in enumerate(train_dataloader):
                train_dt, train_lable = train_dt.to(device), train_lable.to(device)
                train_dt = train_dt.float()

                output = model(train_dt, train_lable)
                np_output = output.cpu().detach().numpy()
                Inline1250_vp_vs[(trace_number * BATCH_SIZE):((trace_number + 1) * BATCH_SIZE), (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)] = np_output

    pathmat = './LN_out/Inline1250_vp_vs_model1_2020_08_08_0%d.mat' % date_num
    scipio.savemat(pathmat, {'Inline1250_vp_vs_model1_2020_08_08_0%d' % date_num: Inline1250_vp_vs})


def tested3():

    Inline750_vp_vs = np.zeros((xline_num, train_well.shape[1]))
    Inline750_seismic = seismic[(750-750)*xline_num:(751-750)*xline_num, :]

    model = BSsequential_net(BATCH_LEN, num_well).to(device)
    mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_LN_2020_08_08_0%d.pth' % date_num
    model.load_state_dict(torch.load(mode_patch))
    for trace_number in range(0, xline_num):
        print(trace_number)

        # 计算相关系数
        coef_seismic = np.zeros((train_well.shape[0] + 1, train_well.shape[1]))
        coef_seismic[0, :] = Inline750_seismic[trace_number, :]
        coef_seismic[1:coef_seismic.shape[0], :] = train_well_seismic[:, :]
        temp_coef = np.corrcoef(coef_seismic)

        # 优选出相关系数大于阈值并且半径范围内的井
        tempval_1 = np.zeros(0)
        temp_train_well_1 = np.zeros(0)
        temp_train_well_seisic_1 = np.zeros(0)
        absCORcoef = np.abs(temp_coef[0, 1:coef_seismic.shape[0]])
        if which_choose_well == 1:
            num = 0
            for k in range(0, 15):
                if absCORcoef[k] > coefval:
                    # 井数据的坐标
                    wellxline = Xline_Inline_number[0, k]
                    wellinline = Xline_Inline_number[1, k]
                    # 目标地震数据的坐标
                    seismicinline = np.mod(trace_number + 1, xline_num)
                    seismicxline = (trace_number + 1 - seismicinline) / xline_num + 1
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
                    for max_num2 in range(0, 15):
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
                for max_num2 in range(0, train_well.shape[0]):
                    if temp_tempval == absCORcoef[max_num2]:
                        absCORcoef[max_num2] = 0
                        temp_train_well = np.append(temp_train_well, train_well[max_num2, :])
                        temp_train_well_seisic = np.append(temp_train_well_seisic, train_well_seismic[max_num2, :])

        num = num_well
        temp_train_well = torch.from_numpy(temp_train_well)
        temp_train_well = temp_train_well.view(num, -1)
        temp_train_well = temp_train_well.float()

        temp_seismic = Inline750_seismic[trace_number, :]
        temp_seismic = torch.from_numpy(temp_seismic)
        temp_seismic = temp_seismic.float()
        temp_seismic = temp_seismic.view(1, -1)

        for num_batchlen in range(0, number):
            # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入
            train_dataset = MyDataset(temp_train_well[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)],
                                       temp_seismic[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)])
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True,
                                          drop_last=False)

            for itr, (train_dt, train_lable) in enumerate(train_dataloader):
                train_dt, train_lable = train_dt.to(device), train_lable.to(device)
                train_dt = train_dt.float()

                output = model(train_dt, train_lable)
                np_output = output.cpu().detach().numpy()
                Inline750_vp_vs[(trace_number * BATCH_SIZE):((trace_number + 1) * BATCH_SIZE), (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)] = np_output

    pathmat = './LN_out/Inline750_vp_vs_model1_2020_08_08_0%d.mat' % date_num
    scipio.savemat(pathmat, {'Inline750_vp_vs_model1_2020_08_08_0%d' % date_num: Inline750_vp_vs})


if __name__ == '__main__':
    # process_well_cor()
    pre_trained(is_inherit_net)
    tested1()
    tested2()
    tested3()
    # inversion(0)
    # tensorboard --logdir E:\Python_project\Predict_Impedence\loss\pre_train_loss_marmousi2019_12_10_11
