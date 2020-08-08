import math
import torch
import numpy as np

# from net_and_data import MyDataset
# from torch.utils.data import DataLoader
# import scipy.io as scipio
# import matplotlib.pyplot as plt


def convmtx(wavelet, modelLength):

    c = torch.zeros(modelLength - 1)
    c = torch.cat([wavelet, c])
    r = torch.zeros(modelLength-1)
    m = c.size(0)
    x = torch.cat([r, c])
    x = x.view(x.size(0), 1)

    cidx = torch.linspace(0, m-1, m)  # 等分切割start和end之间为该数量的list  #[0 : m-1]  # 一个0到m-1的数列
    cidx = cidx.view(cidx.size(0), 1)
    cidx = torch.Tensor.repeat(cidx, [1, modelLength])
    ridx = torch.linspace(modelLength-1, 0, modelLength)  # modelLength:-1:1  # 一个modelLength到1的数列
    ridx = torch.Tensor.repeat(ridx, [m, 1])
    t_temp = cidx + ridx
    t_temp = t_temp.int()
    t = torch.zeros(t_temp.size())
    for i in range(t_temp.size(0)):
        for j in range(t_temp.size(1)):
            t[i, j] = x[t_temp[i, j], 0]

    return t


def g_generate(wavelet, seismic_len):
    # 生成正演算子G的函数，输入要求：
    # 子波：wavelet必须是tensor[1, length]的float32的类型
    # 地震数据长度seismic_len
    # 输出正演算子G

    # 读取地震数据的长度
    len_seis = seismic_len
    # 生成导数矩阵D
    d = torch.zeros(len_seis - 1, len_seis)
    for i in range(0, len_seis-1):
        d[i, i] = -1
        d[i, i + 1] = 1
    # 生成子波的褶积矩阵W
    ix = wavelet.argmax()
    ix = ix.item()
    w_matrix = convmtx(wavelet, len_seis-1)
    w = w_matrix[ix:ix + len_seis, :]
    # 生成正演算子矩阵G
    g = 0.5 * w.mm(d)

    return g


# if __name__ == '__main__':
#
#     waveFile = 'E://Python_project/Predict_Impedence/Out_Predictdata/wavelet1225.mat'
#     wave = scipio.loadmat(waveFile)
#     wavelet = wave['wavelet1225']
#     wavelet = torch.from_numpy(wavelet)
#     wavelet = wavelet.float()
#     wavelet = wavelet.view(wavelet.size(1))
#
#
#     # BATCH_SIZE = 1  # BATCH_SIZE大小
#     # dataFile = 'E://Python_project/Predict_Impedence/Out_Predictdata/well_well52_len30.mat'
#     # welldata = scipio.loadmat(dataFile)
#     # train_welldata = welldata['well_well52_len30']
#     #
#     # train_dataset = MyDataset(train_welldata, train_welldata)
#     # train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=3, shuffle=True, drop_last=False)
#     # for itr, (train_dt, train_lable) in enumerate(train_dataloader):
#
#
#     ModelFile = 'E://Python_project/Predict_Impedence/Out_Predictdata/trueModel.mat'
#     trueModel = scipio.loadmat(ModelFile)
#     trueModel = trueModel['trueModel']
#     trueModel = torch.from_numpy(trueModel)
#     trueModel = trueModel.float()
#     trueModel = trueModel.view(1, trueModel.size(1))
#
#     g = g_generate(wavelet, trueModel.size(1))
#     trueModel = trueModel.view(trueModel.size(1), 1)
#     trueModel = trueModel.float()
#     y = g.mm(trueModel)
#     x = torch.linspace(0, y.size(0)-1, y.size(0))
#     plt.plot(x, y, '-')
#     a = x









