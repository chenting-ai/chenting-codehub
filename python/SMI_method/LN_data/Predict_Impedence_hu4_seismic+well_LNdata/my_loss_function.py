import math
import torch
from G_function import g_generate
import numpy as np
import scipy.signal


# scipy.signal.convolve(x,h)  两个行向量的卷积

def syn_seismic_fun(pre_impede, wavelet):
    # 利用子波和地震数据长度构建矩阵G
    gg = g_generate(wavelet, pre_impede.size(1))

    # 合成地震记录syn_seismic  要与波阻抗的对数相乘
    # 对一个batch的波阻抗做一个转置才能与正演矩阵相乘，最后再转置回来变成原来的大小（两次转置）
    pre_impede = torch.transpose(pre_impede, 1, 0)
    pre_impede2 = pre_impede[:, :]*6538+3995.7
    syn_seismic = gg.mm(torch.log(pre_impede2))
    syn_seismic = torch.transpose(syn_seismic, 1, 0)
    #
    max_value = torch.max(syn_seismic)
    min_value = torch.min(syn_seismic)-0.0001
    value_len = max_value-min_value
    syn_seismic = (syn_seismic-min_value)/value_len
    ret = syn_seismic
    return ret


def syn_seismic_fun2(pre_impede, wavelet):

    # 先用波阻抗序列求反射系数序列
    pre_impede2 = pre_impede[:, :]*6538+3995.7
    trace_number = pre_impede.shape[0]
    point_number = pre_impede.shape[1]
    reflect = np.zeros((trace_number, point_number))
    for i in range(0, trace_number):
        for j in range(1, point_number):
            reflect[i, j] = (pre_impede2[i, j]-pre_impede2[i, j-1])/(pre_impede2[i, j]+pre_impede2[i, j-1])
    # 再卷积，取目标长度就行
    syn_seismic = np.zeros((trace_number, point_number))
    for i in range(0, trace_number):
        temp_syn_seismic = scipy.signal.convolve(reflect[i, :], wavelet)
        lensyn_seis = temp_syn_seismic.shape[0]
        syn_seismic[i, :] = temp_syn_seismic[round((lensyn_seis - point_number)/2):round((lensyn_seis - point_number)/2 + point_number)]

    syn_seismic = torch.from_numpy(syn_seismic)
    max_value = torch.max(syn_seismic)
    min_value = torch.min(syn_seismic)-0.0001
    value_len = max_value-min_value
    syn_seismic = (syn_seismic-min_value)/value_len
    ret = syn_seismic
    return ret
# def my_mse_loss_fun(pre_impede, lable):
#     # 计算损失函数
#     ret = torch.mean((pre_impede - lable) ** 2)
#
#     return ret


def my_mse_loss_fun(seismic, pre_impede, ip_m0, wavelet):
    # 利用子波和地震数据长度构建矩阵G
    gg = g_generate(wavelet, seismic.size(1))

    # 合成地震记录syn_seismic  要与波阻抗的对数相乘
    # 对一个batch的波阻抗做一个转置才能与正演矩阵相乘，最后再转置回来变成原来的大小（两次转置）
    pre_impede = torch.transpose(pre_impede, 1, 0)
    pre_impede2 = pre_impede[:, :]*6538+3995.7
    syn_seismic = gg.mm(torch.log(pre_impede2))
    syn_seismic = torch.transpose(syn_seismic, 1, 0)
    pre_impede = torch.transpose(pre_impede, 1, 0)
    # 计算损失函数
    ret = torch.mean(((seismic - syn_seismic) ** 2) + ((pre_impede - ip_m0) ** 2))

    return ret


def my_mse_loss_fun2(output, train_lable, wavelet):
    syn_seismic = syn_seismic_fun2(output, wavelet)
    syn_seismic = syn_seismic.float()
    # 计算损失函数
    ret = torch.mean(((train_lable - syn_seismic) ** 2))

    return ret




