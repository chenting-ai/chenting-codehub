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
#  输入数据只有地震数据
BATCH_SIZE = 1  # BATCH_SIZE大小
BATCH_LEN = 60  # BATCH_LEN长度
EPOCHS = 50000  # 总共训练批次
data_rate = 1  # 选用前50%的xline线训练
number = np.int(60/BATCH_LEN)  # 一道有几个batch长度

is_consistent = 0  # 是否要确定的随机路径
is_synseismic = 0  # 是否增加合成地震记录约束
is_inherit_net = 0  # 是否继承之前的网络
which_choose_well = 1  # 第几种选井方式 0.选前num_well口，1.在阈值和半径范围内选前num_well口，

date_num = 1  # 今天第几次跑
# 加载数据
if 1:
    # 加载训练用井旁地震数据
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

    # Xline1_110_GEO_impedance,Xline1_110_SMI_impedance
    # Xline1_75_label_impedance = torch.from_numpy(Xline1_75_label_impedance)
    # Xline1_75_label_impedance = Xline1_75_label_impedance.float()
    # Xline1_75_label_impedance = Xline1_75_label_impedance.to(device)

    # 加载线道号
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
    waveFile = './SMI_data/wavelet_1_1.mat'
    wavel = scipio.loadmat(waveFile)
    wavele = wavel['wavelet_1_1']
    wavelet = torch.from_numpy(wavele)
    wavelet = wavelet.float()
    wavelet = wavelet.view(wavelet.size(1))
    wavelet = wavelet.to(device)


def pre_trained(judge):

    writer = SummaryWriter(log_dir='./loss/pre_train_loss_model1/pre_train_loss_SMI_2020_08_03_0%d' % date_num)

    if judge == 0:
        model = BSsequential_net_seismic().to(device)
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
        # model.apply(weights_init)
        temp = 10000000000000
        epoch_num = 1
    else:
        model = BSsequential_net_lstm().to(device)
        mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_SMI_2020_08_03_02.pth'
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
            trace_number = np.int(map_xline[0, epoch-1]*142+map_inline[0, epoch-1])
        else:
            temp_1 = np.random.randint(0, 142, 1)  # 29
            temp_2 = np.random.randint(0, 110, 1)  # 22
            trace_number = temp_2*142+temp_1
            map_xline = np.append(map_xline, temp_2)
            map_inline = np.append(map_inline, temp_1)
        temp_lable = torch.from_numpy(Xline1_110_label_impedance[trace_number, :])
        temp_lable = temp_lable.float()
        temp_lable = temp_lable.view(1, -1)
        for num_rand in range(0, number):
            rand = np.random.randint(0, 60 - BATCH_LEN + 1, 1)
            temp_train_seismic = train1_110_seismic[trace_number, rand[0]:rand[0] + BATCH_LEN]
            temp_train_seismic = torch.from_numpy(temp_train_seismic)
            temp_train_seismic = temp_train_seismic.float()
            temp_train_seismic = temp_train_seismic.view(1, -1)

            # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入

            train_dataset = MyDataset2(temp_train_seismic[:, rand[0]:rand[0] + BATCH_LEN], temp_lable[:, rand[0]:rand[0] + BATCH_LEN])
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True, drop_last=False)
            epoch_loss = []

            for itr, (train_dt, train_lable) in enumerate(train_dataloader):
                train_dt, train_lable = train_dt.to(device), train_lable.to(device)
                train_dt = train_dt.float()
                train_lable = train_lable.float()

                model.train()
                optimizer.zero_grad()
                output = model(train_dt)
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
            path = './model_file/pre_trained_network_model_model1/pre_trained_network_model_SMI_2020_08_03_0%d.pth' % date_num
            torch.save(model.state_dict(), path)
        path_loss = './Temporary_parameters/pre_temp_model1.mat'
        path_epoch = './Temporary_parameters/pre_epoch_num_model1.mat'
        scipio.savemat(path_loss, {'epoch_loss': epoch_loss})
        scipio.savemat(path_epoch, {'epoch_num': epoch_num})
    if is_consistent == 0:
        pathmat = './SMI_out/map_number_2020_08_03_0%d.mat' % date_num
        scipio.savemat(pathmat, {'map_xline': map_xline, 'map_inline': map_inline})
    writer.add_graph(model, (train_dt))
    writer.close()


def tested():

    impedance_xline16 = np.zeros((test_Xline16_seismic.shape[0], test_Xline16_seismic.shape[1]))

    model = BSsequential_net_seismic().to(device)
    mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_SMI_2020_08_03_0%d.pth' % date_num
    model.load_state_dict(torch.load(mode_patch))
    for trace_number in range(0, test_Xline16_seismic.shape[0]):
        print(trace_number)

        temp_lable = torch.from_numpy(test_Xline16_seismic[trace_number, :])
        temp_lable = temp_lable.float()
        temp_lable = temp_lable.view(1, -1)

        for num_batchlen in range(0, number):
            # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入
            train_dataset = MyDataset2(temp_lable[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)],
                                       temp_lable[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)])
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True,
                                          drop_last=False)

            for itr, (train_dt, train_lable) in enumerate(train_dataloader):
                train_dt, train_lable = train_dt.to(device), train_lable.to(device)
                train_dt = train_dt.float()

                output = model(train_dt)
                # train_dt = train_dt.view(num_well, -1)
                # output = tempval.mm(train_dt)
                np_output = output.cpu().detach().numpy()
                impedance_xline16[(trace_number * BATCH_SIZE):((trace_number + 1) * BATCH_SIZE), (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)] = np_output

    pathmat = './SMI_out/test_X16_Impedance_model1_2020_08_03_0%d.mat' % date_num
    scipio.savemat(pathmat, {'test_X16_Impedance_model1_2020_08_03_0%d' % date_num: impedance_xline16})


def tested2():
    impedance_xline76 = np.zeros((test_Xline76_seismic.shape[0], test_Xline76_seismic.shape[1]))

    model = BSsequential_net_seismic().to(device)
    mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_SMI_2020_08_03_0%d.pth' % date_num
    model.load_state_dict(torch.load(mode_patch))
    for trace_number in range(0, test_Xline76_seismic.shape[0]):
        print(trace_number)

        temp_lable = torch.from_numpy(test_Xline76_seismic[trace_number, :])
        temp_lable = temp_lable.float()
        # temp_lable = temp_lable.to(device)
        temp_lable = temp_lable.view(1, -1)
        for num_batchlen in range(0, number):
            # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入
            train_dataset = MyDataset2(
                temp_lable[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)],
                temp_lable[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)])
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True,
                                          drop_last=False)

            for itr, (train_dt, train_lable) in enumerate(train_dataloader):
                train_dt, train_lable = train_dt.to(device), train_lable.to(device)
                train_dt = train_dt.float()

                output = model(train_dt)
                # train_dt = train_dt.view(num_well, -1)
                # output = tempval.mm(train_dt)
                np_output = output.cpu().detach().numpy()
                impedance_xline76[(trace_number * BATCH_SIZE):((trace_number + 1) * BATCH_SIZE),
                (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)] = np_output

    pathmat = './SMI_out/pre_X76_Impedance_model1_2020_08_03_0%d.mat' % date_num
    scipio.savemat(pathmat, {'pre_X76_Impedance_model1_2020_08_03_0%d' % date_num: impedance_xline76})


def tested3():
    impedance_inline26 = np.zeros((test_Inline26_seismic.shape[0], test_Inline26_seismic.shape[1]))

    model = BSsequential_net_seismic().to(device)
    mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_SMI_2020_08_03_0%d.pth' % date_num
    model.load_state_dict(torch.load(mode_patch))
    for trace_number in range(0, test_Inline26_seismic.shape[0]):
        print(trace_number)

        temp_lable = torch.from_numpy(test_Inline26_seismic[trace_number, :])
        temp_lable = temp_lable.float()
        # temp_lable = temp_lable.to(device)
        temp_lable = temp_lable.view(1, -1)
        for num_batchlen in range(0, number):
            # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入
            train_dataset = MyDataset2(temp_lable[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)],
                                       temp_lable[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)])
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True,
                                          drop_last=False)

            for itr, (train_dt, train_lable) in enumerate(train_dataloader):
                train_dt, train_lable = train_dt.to(device), train_lable.to(device)
                train_dt = train_dt.float()

                output = model(train_dt)
                # train_dt = train_dt.view(num_well, -1)
                # output = tempval.mm(train_dt)
                np_output = output.cpu().detach().numpy()
                impedance_inline26[(trace_number * BATCH_SIZE):((trace_number + 1) * BATCH_SIZE), (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)] = np_output

    pathmat = './SMI_out/pre_In26_Impedance_model1_2020_08_03_0%d.mat' % date_num
    scipio.savemat(pathmat, {'pre_In26_Impedance_model1_2020_08_03_0%d' % date_num: impedance_inline26})


def tested4():
    impedance_inline99 = np.zeros((test_Inline99_seismic.shape[0], test_Inline99_seismic.shape[1]))

    model = BSsequential_net_seismic().to(device)
    mode_patch = './model_file/pre_trained_network_model_model1/pre_trained_network_model_SMI_2020_08_03_0%d.pth' % date_num
    model.load_state_dict(torch.load(mode_patch))
    # num_params = 0
    # for param in model.parameters():
    #     num_params += param.numel()
    # print(num_params)
    for trace_number in range(0, test_Inline99_seismic.shape[0]):
        print(trace_number)

        temp_lable = torch.from_numpy(test_Inline99_seismic[trace_number, :])
        temp_lable = temp_lable.float()
        # temp_lable = temp_lable.to(device)
        temp_lable = temp_lable.view(1, -1)
        for num_batchlen in range(0, number):
            # 利用优选出来的井数据，井旁道，加上一个目标道组成网络的输入
            train_dataset = MyDataset2(temp_lable[:, (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)],
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

                output = model(train_dt)
                # train_dt = train_dt.view(num_well, -1)
                # output = tempval.mm(train_dt)
                np_output = output.cpu().detach().numpy()
                impedance_inline99[(trace_number * BATCH_SIZE):((trace_number + 1) * BATCH_SIZE), (num_batchlen * BATCH_LEN):((num_batchlen + 1) * BATCH_LEN)] = np_output

    pathmat = './SMI_out/pre_In99_Impedance_model1_2020_08_03_0%d.mat' % date_num
    scipio.savemat(pathmat, {'pre_In99_Impedance_model1_2020_08_03_0%d' % date_num: impedance_inline99})


if __name__ == '__main__':
    pre_trained(is_inherit_net)
    tested()
    tested2()
    tested3()
    tested4()
    # inversion(0)
    # tensorboard --logdir E:\Python_project\Predict_Impedence\loss\pre_train_loss_marmousi2019_12_10_11
