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
from tensorboardX import SummaryWriter
from net_and_data import MyDataset2, BSsequential_net_seismic, device, weights_init
from my_loss_function import my_mse_loss_fun, syn_seismic_fun2
#  输入为地震数据
BATCH_SIZE = 1  # BATCH_SIZE大小
BATCH_LEN = 100  # BATCH_LEN长度
EPOCHS = 500000  # 总共训练批次
number = np.int(100/BATCH_LEN)  # 一道有几个batch长度
inline_num = 501
xline_num = 631

is_consistent = 0  # 是否要确定的随机路径
is_synseismic = 0  # 是否增加合成地震记录约束
is_inherit_net = 0  # 是否继承之前的网络

date_num = 3  # 今天第几次跑
# 加载数据
if 1:
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


def pre_trained(judge):

    writer = SummaryWriter(log_dir='./loss/pre_train_loss_model1/pre_train_loss_LN_2020_08_05_0%d' % date_num)

    if judge == 0:
        model = BSsequential_net_seismic(BATCH_LEN).to(device)
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
        device1 = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        print(device1)
        # model.apply(weights_init)
        temp = 10000000000000
        epoch_num = 1
    else:
        model = BSsequential_net(BATCH_LEN).to(device)
        mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_LN_2020_08_04_02.pth'
        model.load_state_dict(torch.load(mode_patch))
        temp = 10000000000000
        epoch_num = 1
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
    for epoch in range(epoch_num, EPOCHS+1):
        print(epoch, count)
        temp_weight = model.fc60.weight   # 检验网络权重是否变化的初始网络参数
        temp_a = torch.sum(temp_weight.data)
        # print(temp_weight)
        # temp_weight = model.lstm60
        # temp_a = torch.sum(temp_weight.weight_hh_l0.data) + torch.sum(temp_weight.weight_ih_l0.data)
        # print(a)

        if np.mod(epoch + 1, 200) == 0:
            lr = lr * 0.99
        optimizer = optim.Adam(model.parameters(), lr=lr)
        if is_consistent == 1:
            trace_number = np.int(map_inline[0, epoch-1]*xline_num+map_xline[0, epoch-1])
        else:
            temp_1 = np.random.randint(0, 501, 1)
            temp_2 = np.random.randint(0, 631, 1)
            trace_number = temp_1*631+temp_2
            map_inline = np.append(map_inline, temp_1)
            map_xline = np.append(map_xline, temp_2)

        temp_train_seismic = seismic[trace_number, :]
        temp_train_seismic = torch.from_numpy(temp_train_seismic)
        temp_train_seismic = temp_train_seismic.float()
        temp_train_seismic = temp_train_seismic.view(1, -1)
        temp_lable = torch.from_numpy(vp_vs_lable[trace_number, :])
        temp_lable = temp_lable.float()
        temp_lable = temp_lable.view(1, -1)
        for num_rand in range(0, number):
            # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入
            rand = np.random.randint(0, seismic.shape[1] - BATCH_LEN + 1, 1)
            train_dataset = MyDataset2(temp_train_seismic[:, rand[0]:rand[0] + BATCH_LEN], temp_lable[:, rand[0]:rand[0] + BATCH_LEN])
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True, drop_last=False)
            epoch_loss = []

            for itr, (train_dt, train_lable) in enumerate(train_dataloader):
                train_dt, train_lable = train_dt.to(device), train_lable.to(device)
                train_dt = train_dt.float()
                train_lable = train_lable.float()

                model.train()
                optimizer.zero_grad()
                output = model(train_dt, BATCH_LEN)
                if is_synseismic == 1:
                    syn_seismic = syn_seismic_fun2(output, wavelet)
                    syn_seismic = syn_seismic.float()
                    loss = F.mse_loss(syn_seismic, temp_train_seismic) + F.mse_loss(output, train_lable)
                else:
                    loss = F.mse_loss(output, train_lable)

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
        if count > 50:
            break

        epoch_loss = np.sum(np.array(epoch_loss))
        writer.add_scalar('Train/MSE', epoch_loss, epoch)
        epoch_num = epoch
        print('Train set: Average loss: {:.15f}'.format(epoch_loss))
        if epoch_loss < temp:
            path = './model_file/pre_trained_network_model_model1/pre_trained_network_model_LN_2020_08_05_0%d.pth' % date_num
            torch.save(model.state_dict(), path)
        path_loss = './Temporary_parameters/pre_temp_model1.mat'
        path_epoch = './Temporary_parameters/pre_epoch_num_model1.mat'
        scipio.savemat(path_loss, {'epoch_loss': epoch_loss})
        scipio.savemat(path_epoch, {'epoch_num': epoch_num})
    if is_consistent == 0:
        pathmat = './LN_out/map_number_2020_08_05_0%d.mat' % date_num
        scipio.savemat(pathmat, {'map_xline': map_xline, 'map_inline': map_inline})
    writer.add_graph(model, (train_dt, torch.tensor(BATCH_LEN)))
    writer.close()


def tested1():

    Inline1134_vp_vs = np.zeros((xline_num, seismic.shape[1]))
    Inline1134_seismic = seismic[(1134-750)*xline_num:(1135-750)*xline_num, :]

    model = BSsequential_net_seismic(BATCH_LEN).to(device)
    mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_LN_2020_08_05_0%d.pth' % date_num
    model.load_state_dict(torch.load(mode_patch))   # , map_location='cpu'
    for trace_number in range(0, xline_num):
        print(trace_number)

        temp_seismic = Inline1134_seismic[trace_number, :]
        temp_seismic = torch.from_numpy(temp_seismic)
        temp_seismic = temp_seismic.float()
        temp_seismic = temp_seismic.view(1, -1)

        for num_batchlen in range(0, number):
            # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入
            train_dataset = MyDataset2(temp_seismic[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)],
                                       temp_seismic[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)])
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True,
                                          drop_last=False)

            for itr, (train_dt, train_lable) in enumerate(train_dataloader):
                train_dt, train_lable = train_dt.to(device), train_lable.to(device)
                train_dt = train_dt.float()

                output = model(train_dt, BATCH_LEN)
                np_output = output.cpu().detach().numpy()
                Inline1134_vp_vs[(trace_number * BATCH_SIZE):((trace_number + 1) * BATCH_SIZE), (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)] = np_output

    pathmat = './LN_out/Inline1134_vp_vs_model1_2020_08_05_0%d.mat' % date_num
    scipio.savemat(pathmat, {'Inline1134_vp_vs_model1_2020_08_05_0%d' % date_num: Inline1134_vp_vs})


def tested2():

    Inline1250_vp_vs = np.zeros((xline_num, seismic.shape[1]))
    Inline1250_seismic = seismic[(1250-750)*xline_num:(1251-750)*xline_num, :]

    model = BSsequential_net_seismic(BATCH_LEN).to(device)
    mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_LN_2020_08_05_0%d.pth' % date_num
    model.load_state_dict(torch.load(mode_patch))
    for trace_number in range(0, xline_num):
        print(trace_number)

        temp_seismic = Inline1250_seismic[trace_number, :]
        temp_seismic = torch.from_numpy(temp_seismic)
        temp_seismic = temp_seismic.float()
        temp_seismic = temp_seismic.view(1, -1)

        for num_batchlen in range(0, number):
            # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入
            train_dataset = MyDataset2(temp_seismic[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)],
                                       temp_seismic[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)])
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True,
                                          drop_last=False)

            for itr, (train_dt, train_lable) in enumerate(train_dataloader):
                train_dt, train_lable = train_dt.to(device), train_lable.to(device)
                train_dt = train_dt.float()

                output = model(train_dt, BATCH_LEN)
                np_output = output.cpu().detach().numpy()
                Inline1250_vp_vs[(trace_number * BATCH_SIZE):((trace_number + 1) * BATCH_SIZE), (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)] = np_output

    pathmat = './LN_out/Inline1250_vp_vs_model1_2020_08_05_0%d.mat' % date_num
    scipio.savemat(pathmat, {'Inline1250_vp_vs_model1_2020_08_05_0%d' % date_num: Inline1250_vp_vs})


def tested3():

    Inline750_vp_vs = np.zeros((xline_num, seismic.shape[1]))
    Inline750_seismic = seismic[(750-750)*xline_num:(751-750)*xline_num, :]

    model = BSsequential_net_seismic(BATCH_LEN).to(device)
    mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_LN_2020_08_05_0%d.pth' % date_num
    model.load_state_dict(torch.load(mode_patch))
    for trace_number in range(0, xline_num):
        print(trace_number)

        temp_seismic = Inline750_seismic[trace_number, :]
        temp_seismic = torch.from_numpy(temp_seismic)
        temp_seismic = temp_seismic.float()
        temp_seismic = temp_seismic.view(1, -1)

        for num_batchlen in range(0, number):
            # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入
            train_dataset = MyDataset2(temp_seismic[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)],
                                       temp_seismic[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)])
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True,
                                          drop_last=False)

            for itr, (train_dt, train_lable) in enumerate(train_dataloader):
                train_dt, train_lable = train_dt.to(device), train_lable.to(device)
                train_dt = train_dt.float()

                output = model(train_dt, BATCH_LEN)
                np_output = output.cpu().detach().numpy()
                Inline750_vp_vs[(trace_number * BATCH_SIZE):((trace_number + 1) * BATCH_SIZE), (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)] = np_output

    pathmat = './LN_out/Inline750_vp_vs_model1_2020_08_05_0%d.mat' % date_num
    scipio.savemat(pathmat, {'Inline750_vp_vs_model1_2020_08_05_0%d' % date_num: Inline750_vp_vs})


if __name__ == '__main__':
    # pre_trained(is_inherit_net)
    tested1()
    tested2()
    tested3()
    # inversion(0)
    # tensorboard --logdir E:\Python_project\Predict_Impedence\loss\pre_train_loss_marmousi2019_12_10_11
