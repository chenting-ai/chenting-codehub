import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


class MyDataset(data.Dataset):
    def __init__(self, parameter_data, parameter_target):
        self.data = np.array(parameter_data)
        self.target = np.array(parameter_target)

    def __getitem__(self, index):  # 返回的是tensor
        d, target = self.data[:, :], self.target[index, :]
        d = torch.from_numpy(d)
        target = torch.from_numpy(target)
        return d, target

    def __len__(self):
        return self.target.shape[0]


class MyDataset2(data.Dataset):
    def __init__(self, parameter_data, parameter_target):
        self.data = np.array(parameter_data)
        self.target = np.array(parameter_target)

    def __getitem__(self, index):  # 返回的是tensor
        d, target = self.data[index, :], self.target[index, :]
        d = torch.from_numpy(d)
        target = torch.from_numpy(target)
        return d, target

    def __len__(self):
        return self.target.shape[0]


class TripleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        '''
        双重卷积
        :param in_ch: 输入通道
        :param out_ch: 输出通道
        '''
        super(TripleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class QuadrupleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        '''
        双重卷积
        :param in_ch: 输入通道
        :param out_ch: 输出通道
        '''
        super(QuadrupleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class QuadrupleConv_U(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(QuadrupleConv_U, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 2*out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(2*out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*out_ch, 3*out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(3*out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*out_ch, 2*out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(2*out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class TripleConv_U(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TripleConv_U, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 2*out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(2*out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*out_ch, 2*out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(2*out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class QuadrupleConv1d_U(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(QuadrupleConv1d_U, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 2*out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(2*out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(2*out_ch, 3*out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(3*out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(3*out_ch, 2*out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(2*out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(2*out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class TripleConv1d_U(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TripleConv1d_U, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 2*out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(2*out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(2*out_ch, 2*out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(2*out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(2*out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class TripleConv1d(nn.Module):
    def __init__(self, in_ch, out_ch):
        '''
        双重卷积
        :param in_ch: 输入通道
        :param out_ch: 输出通道
        '''
        super(TripleConv1d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class QuadrupleConv1d(nn.Module):
    def __init__(self, in_ch, out_ch):
        '''
        双重卷积
        :param in_ch: 输入通道
        :param out_ch: 输出通道
        '''
        super(QuadrupleConv1d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class CNN_SMI_1d_Res(nn.Module):
    # 卷积神经网络，丢掉之前的LSTM层
    def __init__(self):
        super(CNN_SMI_1d_Res, self).__init__()

        self.conv11 = QuadrupleConv1d_U(2, 1)
        self.conv12 = TripleConv1d_U(2, 1)
        self.conv13 = QuadrupleConv1d_U(2, 1)
        self.conv14 = TripleConv1d_U(2, 1)
        self.conv15 = QuadrupleConv1d_U(2, 1)
        self.conv16 = TripleConv1d_U(1, 1)
        self.conv17 = QuadrupleConv1d_U(5, 1)
        self.conv18 = QuadrupleConv1d_U(6, 1)

        self.conv21 = TripleConv1d_U(8, 1)
        self.conv22 = QuadrupleConv1d_U(8, 1)
        self.conv23 = TripleConv1d_U(8, 1)
        self.conv24 = QuadrupleConv1d_U(8, 1)
        self.conv25 = TripleConv1d_U(8, 1)
        self.conv26 = QuadrupleConv1d_U(8, 1)
        self.conv27 = TripleConv1d_U(8, 1)
        self.conv28 = QuadrupleConv1d_U(8, 1)

        self.conv31 = QuadrupleConv1d_U(8, 1)
        self.conv32 = TripleConv1d_U(8, 1)
        self.conv33 = QuadrupleConv1d_U(8, 1)
        self.conv34 = TripleConv1d_U(8, 1)
        self.conv35 = QuadrupleConv1d_U(8, 1)
        self.conv36 = TripleConv1d_U(8, 1)
        self.conv37 = QuadrupleConv1d_U(8, 1)
        self.conv38 = TripleConv1d_U(8, 1)

        self.conv41 = TripleConv1d_U(8, 1)
        self.conv42 = QuadrupleConv1d_U(8, 1)
        self.conv43 = TripleConv1d_U(8, 1)
        self.conv44 = QuadrupleConv1d_U(8, 1)
        self.conv45 = TripleConv1d_U(8, 1)
        self.conv46 = QuadrupleConv1d_U(8, 1)
        self.conv47 = TripleConv1d_U(8, 1)
        self.conv48 = QuadrupleConv1d_U(8, 1)

        self.conv51 = QuadrupleConv1d_U(8, 1)
        self.conv52 = TripleConv1d_U(8, 1)
        self.conv53 = QuadrupleConv1d_U(8, 1)
        self.conv54 = TripleConv1d_U(8, 1)
        self.conv55 = QuadrupleConv1d_U(8, 1)
        self.conv56 = TripleConv1d_U(8, 1)
        self.conv57 = QuadrupleConv1d_U(8, 1)
        self.conv58 = TripleConv1d_U(8, 1)

        self.conv61 = TripleConv1d_U(8, 1)
        self.conv62 = QuadrupleConv1d_U(8, 1)
        self.conv63 = TripleConv1d_U(8, 1)
        self.conv64 = QuadrupleConv1d_U(8, 1)
        self.conv65 = TripleConv1d_U(8, 1)
        self.conv66 = QuadrupleConv1d_U(8, 1)
        self.conv67 = TripleConv1d_U(8, 1)
        self.conv68 = QuadrupleConv1d_U(8, 1)

        self.conv71 = QuadrupleConv1d_U(8, 1)
        self.conv72 = TripleConv1d_U(8, 1)
        self.conv73 = QuadrupleConv1d_U(8, 1)
        self.conv74 = TripleConv1d_U(8, 1)
        self.conv75 = QuadrupleConv1d_U(8, 1)
        self.conv76 = TripleConv1d_U(8, 1)
        self.conv77 = QuadrupleConv1d_U(8, 1)
        self.conv78 = TripleConv1d_U(8, 1)

        self.conv81 = TripleConv1d_U(8, 1)
        self.conv82 = QuadrupleConv1d_U(8, 1)
        self.conv83 = TripleConv1d_U(8, 1)
        self.conv84 = QuadrupleConv1d_U(8, 1)
        self.conv85 = TripleConv1d_U(8, 1)
        self.conv86 = QuadrupleConv1d_U(8, 1)
        self.conv87 = TripleConv1d_U(8, 1)
        self.conv88 = QuadrupleConv1d_U(8, 1)

        self.conv91 = QuadrupleConv1d_U(8, 1)
        self.conv92 = TripleConv1d_U(8, 1)
        self.conv93 = QuadrupleConv1d_U(8, 1)
        self.conv94 = TripleConv1d_U(8, 1)
        self.conv95 = QuadrupleConv1d_U(8, 1)
        self.conv96 = TripleConv1d_U(8, 1)
        self.conv97 = QuadrupleConv1d_U(8, 1)
        self.conv98 = TripleConv1d_U(8, 1)

        self.conv101 = TripleConv1d_U(8, 1)
        self.conv102 = QuadrupleConv1d_U(8, 1)
        self.conv103 = TripleConv1d_U(8, 1)
        self.conv104 = QuadrupleConv1d_U(8, 1)
        self.conv105 = TripleConv1d_U(8, 1)
        self.conv106 = QuadrupleConv1d_U(8, 1)
        self.conv107 = TripleConv1d_U(8, 1)
        self.conv108 = QuadrupleConv1d_U(8, 1)

        self.conv111 = QuadrupleConv1d_U(8, 1)
        self.conv112 = TripleConv1d_U(8, 1)
        self.conv113 = QuadrupleConv1d_U(8, 1)
        self.conv114 = TripleConv1d_U(8, 1)
        self.conv115 = QuadrupleConv1d_U(8, 1)
        self.conv116 = TripleConv1d_U(8, 1)
        self.conv117 = QuadrupleConv1d_U(8, 1)
        self.conv118 = TripleConv1d_U(8, 1)

        self.conv121 = TripleConv1d_U(8, 1)
        self.conv122 = QuadrupleConv1d_U(8, 1)
        self.conv123 = TripleConv1d_U(8, 1)
        self.conv124 = QuadrupleConv1d_U(8, 1)
        self.conv125 = TripleConv1d_U(8, 1)
        self.conv126 = QuadrupleConv1d_U(8, 1)
        self.conv127 = TripleConv1d_U(8, 1)
        self.conv128 = QuadrupleConv1d_U(8, 1)

        self.conv131 = QuadrupleConv1d_U(8, 1)
        self.conv132 = TripleConv1d_U(8, 1)
        self.conv133 = QuadrupleConv1d_U(8, 1)
        self.conv134 = TripleConv1d_U(8, 1)
        self.conv135 = QuadrupleConv1d_U(8, 1)
        self.conv136 = TripleConv1d_U(8, 1)
        self.conv137 = QuadrupleConv1d_U(8, 1)
        self.conv138 = TripleConv1d_U(8, 1)

        self.conv141 = TripleConv1d_U(8, 1)
        self.conv142 = QuadrupleConv1d_U(8, 1)
        self.conv143 = TripleConv1d_U(8, 1)
        self.conv144 = QuadrupleConv1d_U(8, 1)
        self.conv145 = TripleConv1d_U(8, 1)
        self.conv146 = QuadrupleConv1d_U(8, 1)
        self.conv147 = TripleConv1d_U(8, 1)
        self.conv148 = QuadrupleConv1d_U(8, 1)

        self.fc151 = nn.Linear(60, 60)
        self.fc152 = nn.Linear(60, 60)
        self.fc153 = nn.Linear(60, 60)
        self.fc154 = nn.Linear(60, 60)
        self.fc155 = nn.Linear(60, 60)
        self.fc156 = nn.Linear(60, 60)
        self.fc157 = nn.Linear(60, 60)
        self.fc158 = nn.Linear(60, 60)
        self.fc159 = nn.Linear(480, 240)

        self.fc161 = nn.Linear(60, 60)
        self.fc162 = nn.Linear(60, 60)
        self.fc163 = nn.Linear(60, 60)
        self.fc164 = nn.Linear(60, 60)
        self.fc165 = nn.Linear(60, 60)
        self.fc166 = nn.Linear(60, 60)
        self.fc167 = nn.Linear(60, 60)
        self.fc168 = nn.Linear(60, 60)
        self.fc169 = nn.Linear(480, 240)
        self.fc1610 = nn.Linear(240, 120)

        self.fc171 = nn.Linear(480, 60)
        self.fc172 = nn.Linear(240, 60)
        self.fc173 = nn.Linear(120, 60)

        self.fc18 = nn.Linear(180, 60)

        self._init_weight()

    # 权重初始化
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_well, seismic):

        # 0720的设计一个深度复杂的卷积网络
        # 输入层
        seismic_in = seismic.view(1, 1, -1)
        layer_in_1_1 = train_well[:, 0, :]
        layer_in_1_1 = torch.cat([layer_in_1_1.view(1, 1, -1), seismic_in], dim=1)
        # layer_in_1_1 = layer_in_1_1.view(layer_in_1_1.size(0), layer_in_1_1.size(1), -1, 10)

        layer_in_1_2 = train_well[:, 1, :]
        layer_in_1_2 = torch.cat([layer_in_1_2.view(1, 1, -1), seismic_in], dim=1)
        # layer_in_1_2 = layer_in_1_2.view(layer_in_1_2.size(0), layer_in_1_2.size(1), -1, 10)

        layer_in_1_3 = train_well[:, 2, :]
        layer_in_1_3 = torch.cat([layer_in_1_3.view(1, 1, -1), seismic_in], dim=1)
        # layer_in_1_3 = layer_in_1_3.view(layer_in_1_3.size(0), layer_in_1_3.size(1), -1, 10)

        layer_in_1_4 = train_well[:, 3, :]
        layer_in_1_4 = torch.cat([layer_in_1_4.view(1, 1, -1), seismic_in], dim=1)
        # layer_in_1_4 = layer_in_1_4.view(layer_in_1_4.size(0), layer_in_1_4.size(1), -1, 10)

        layer_in_1_5 = train_well[:, 4, :]
        layer_in_1_5 = torch.cat([layer_in_1_5.view(1, 1, -1), seismic_in], dim=1)
        # layer_in_1_5 = layer_in_1_5.view(layer_in_1_5.size(0), layer_in_1_5.size(1), -1, 10)

        layer_in_1_6 = seismic.view(1, 1, -1)
        # layer_in_1_6 = layer_in_1_6.view(seismic.size(0), seismic.size(0), -1, 10)

        layer_in_1_7 = train_well.view(1, 5, -1)
        # layer_in_1_7 = layer_in_1_7.view(train_well.size(0), train_well.size(1), -1, 10)

        layer_in_1_8 = torch.cat([layer_in_1_6, layer_in_1_7], dim=1)

        # 第一层 卷积层
        output_layer1_1 = self.conv11(layer_in_1_1)
        output_layer1 = output_layer1_1
        output_layer1_2 = self.conv12(layer_in_1_2)
        output_layer1 = torch.cat([output_layer1, output_layer1_2], dim=1)
        output_layer1_3 = self.conv13(layer_in_1_3)
        output_layer1 = torch.cat([output_layer1, output_layer1_3], dim=1)
        output_layer1_4 = self.conv14(layer_in_1_4)
        output_layer1 = torch.cat([output_layer1, output_layer1_4], dim=1)
        output_layer1_5 = self.conv15(layer_in_1_5)
        output_layer1 = torch.cat([output_layer1, output_layer1_5], dim=1)
        output_layer1_6 = self.conv16(layer_in_1_6)
        output_layer1 = torch.cat([output_layer1, output_layer1_6], dim=1)
        output_layer1_7 = self.conv17(layer_in_1_7)
        output_layer1 = torch.cat([output_layer1, output_layer1_7], dim=1)
        output_layer1_8 = self.conv18(layer_in_1_8)
        output_layer1 = torch.cat([output_layer1, output_layer1_8], dim=1)

        # 第二层 卷积层
        output_layer2_1 = self.conv21(output_layer1)+output_layer1_1
        output_layer2 = output_layer2_1
        output_layer2_2 = self.conv22(output_layer1)+output_layer1_2
        output_layer2 = torch.cat([output_layer2, output_layer2_2], dim=1)
        output_layer2_3 = self.conv23(output_layer1)+output_layer1_3
        output_layer2 = torch.cat([output_layer2, output_layer2_3], dim=1)
        output_layer2_4 = self.conv24(output_layer1)+output_layer1_4
        output_layer2 = torch.cat([output_layer2, output_layer2_4], dim=1)
        output_layer2_5 = self.conv25(output_layer1)+output_layer1_5
        output_layer2 = torch.cat([output_layer2, output_layer2_5], dim=1)
        output_layer2_6 = self.conv26(output_layer1)+output_layer1_6
        output_layer2 = torch.cat([output_layer2, output_layer2_6], dim=1)
        output_layer2_7 = self.conv27(output_layer1)+output_layer1_7
        output_layer2 = torch.cat([output_layer2, output_layer2_7], dim=1)
        output_layer2_8 = self.conv28(output_layer1)+output_layer1_8
        output_layer2 = torch.cat([output_layer2, output_layer2_8], dim=1)

        # 第三层 卷积层
        output_layer3_1 = self.conv31(output_layer2)+output_layer2_1
        output_layer3 = output_layer3_1
        output_layer3_2 = self.conv32(output_layer2)+output_layer2_2
        output_layer3 = torch.cat([output_layer3, output_layer3_2], dim=1)
        output_layer3_3 = self.conv33(output_layer2)+output_layer2_3
        output_layer3 = torch.cat([output_layer3, output_layer3_3], dim=1)
        output_layer3_4 = self.conv34(output_layer2)+output_layer2_4
        output_layer3 = torch.cat([output_layer3, output_layer3_4], dim=1)
        output_layer3_5 = self.conv35(output_layer2)+output_layer2_5
        output_layer3 = torch.cat([output_layer3, output_layer3_5], dim=1)
        output_layer3_6 = self.conv36(output_layer2)+output_layer2_6
        output_layer3 = torch.cat([output_layer3, output_layer3_6], dim=1)
        output_layer3_7 = self.conv37(output_layer2)+output_layer2_7
        output_layer3 = torch.cat([output_layer3, output_layer3_7], dim=1)
        output_layer3_8 = self.conv38(output_layer2)+output_layer2_8
        output_layer3 = torch.cat([output_layer3, output_layer3_8], dim=1)

        # 第四层 卷积层
        output_layer4_1 = self.conv41(output_layer3)+output_layer3_1
        output_layer4 = output_layer4_1
        output_layer4_2 = self.conv42(output_layer3)+output_layer3_2
        output_layer4 = torch.cat([output_layer4, output_layer4_2], dim=1)
        output_layer4_3 = self.conv43(output_layer3)+output_layer3_3
        output_layer4 = torch.cat([output_layer4, output_layer4_3], dim=1)
        output_layer4_4 = self.conv44(output_layer3)+output_layer3_4
        output_layer4 = torch.cat([output_layer4, output_layer4_4], dim=1)
        output_layer4_5 = self.conv45(output_layer3)+output_layer3_5
        output_layer4 = torch.cat([output_layer4, output_layer4_5], dim=1)
        output_layer4_6 = self.conv46(output_layer3)+output_layer3_6
        output_layer4 = torch.cat([output_layer4, output_layer4_6], dim=1)
        output_layer4_7 = self.conv47(output_layer3)+output_layer3_7
        output_layer4 = torch.cat([output_layer4, output_layer4_7], dim=1)
        output_layer4_8 = self.conv48(output_layer3)+output_layer3_8
        output_layer4 = torch.cat([output_layer4, output_layer4_8], dim=1)

        # 第五层 卷积层
        output_layer5_1 = self.conv51(output_layer4)+output_layer4_1
        output_layer5 = output_layer5_1
        output_layer5_2 = self.conv52(output_layer4)+output_layer4_2
        output_layer5 = torch.cat([output_layer5, output_layer5_2], dim=1)
        output_layer5_3 = self.conv53(output_layer4)+output_layer4_3
        output_layer5 = torch.cat([output_layer5, output_layer5_3], dim=1)
        output_layer5_4 = self.conv54(output_layer4)+output_layer4_4
        output_layer5 = torch.cat([output_layer5, output_layer5_4], dim=1)
        output_layer5_5 = self.conv55(output_layer4)+output_layer4_5
        output_layer5 = torch.cat([output_layer5, output_layer5_5], dim=1)
        output_layer5_6 = self.conv56(output_layer4)+output_layer4_6
        output_layer5 = torch.cat([output_layer5, output_layer5_6], dim=1)
        output_layer5_7 = self.conv57(output_layer4)+output_layer4_7
        output_layer5 = torch.cat([output_layer5, output_layer5_7], dim=1)
        output_layer5_8 = self.conv58(output_layer4)+output_layer4_8
        output_layer5 = torch.cat([output_layer5, output_layer5_8], dim=1)

        # 第六层 卷积层
        output_layer6_1 = self.conv61(output_layer5)+output_layer5_1
        output_layer6 = output_layer6_1
        output_layer6_2 = self.conv62(output_layer5)+output_layer5_2
        output_layer6 = torch.cat([output_layer6, output_layer6_2], dim=1)
        output_layer6_3 = self.conv63(output_layer5)+output_layer5_3
        output_layer6 = torch.cat([output_layer6, output_layer6_3], dim=1)
        output_layer6_4 = self.conv64(output_layer5)+output_layer5_4
        output_layer6 = torch.cat([output_layer6, output_layer6_4], dim=1)
        output_layer6_5 = self.conv65(output_layer5)+output_layer5_5
        output_layer6 = torch.cat([output_layer6, output_layer6_5], dim=1)
        output_layer6_6 = self.conv66(output_layer5)+output_layer5_6
        output_layer6 = torch.cat([output_layer6, output_layer6_6], dim=1)
        output_layer6_7 = self.conv67(output_layer5)+output_layer5_7
        output_layer6 = torch.cat([output_layer6, output_layer6_7], dim=1)
        output_layer6_8 = self.conv68(output_layer5)+output_layer5_8
        output_layer6 = torch.cat([output_layer6, output_layer6_8], dim=1)

        # 第七层 卷积层
        output_layer7_1 = self.conv71(output_layer6)+output_layer6_1
        output_layer7 = output_layer7_1
        output_layer7_2 = self.conv72(output_layer6)+output_layer6_2
        output_layer7 = torch.cat([output_layer7, output_layer7_2], dim=1)
        output_layer7_3 = self.conv73(output_layer6)+output_layer6_3
        output_layer7 = torch.cat([output_layer7, output_layer7_3], dim=1)
        output_layer7_4 = self.conv74(output_layer6)+output_layer6_4
        output_layer7 = torch.cat([output_layer7, output_layer7_4], dim=1)
        output_layer7_5 = self.conv75(output_layer6)+output_layer6_5
        output_layer7 = torch.cat([output_layer7, output_layer7_5], dim=1)
        output_layer7_6 = self.conv76(output_layer6)+output_layer6_6
        output_layer7 = torch.cat([output_layer7, output_layer7_6], dim=1)
        output_layer7_7 = self.conv77(output_layer6)+output_layer6_7
        output_layer7 = torch.cat([output_layer7, output_layer7_7], dim=1)
        output_layer7_8 = self.conv78(output_layer6)+output_layer6_8
        output_layer7 = torch.cat([output_layer7, output_layer7_8], dim=1)

        # 第八层 卷积层
        output_layer8_1 = self.conv81(output_layer7)+output_layer7_1
        output_layer8 = output_layer8_1
        output_layer8_2 = self.conv82(output_layer7)+output_layer7_2
        output_layer8 = torch.cat([output_layer8, output_layer8_2], dim=1)
        output_layer8_3 = self.conv83(output_layer7)+output_layer7_3
        output_layer8 = torch.cat([output_layer8, output_layer8_3], dim=1)
        output_layer8_4 = self.conv84(output_layer7)+output_layer7_4
        output_layer8 = torch.cat([output_layer8, output_layer8_4], dim=1)
        output_layer8_5 = self.conv85(output_layer7)+output_layer7_5
        output_layer8 = torch.cat([output_layer8, output_layer8_5], dim=1)
        output_layer8_6 = self.conv86(output_layer7)+output_layer7_6
        output_layer8 = torch.cat([output_layer8, output_layer8_6], dim=1)
        output_layer8_7 = self.conv87(output_layer7)+output_layer7_7
        output_layer8 = torch.cat([output_layer8, output_layer8_7], dim=1)
        output_layer8_8 = self.conv88(output_layer7)+output_layer7_8
        output_layer8 = torch.cat([output_layer8, output_layer8_8], dim=1)

        # 第九层 卷积层
        output_layer9_1 = self.conv91(output_layer8)+output_layer8_1
        output_layer9 = output_layer9_1
        output_layer9_2 = self.conv92(output_layer8)+output_layer8_2
        output_layer9 = torch.cat([output_layer9, output_layer9_2], dim=1)
        output_layer9_3 = self.conv93(output_layer8)+output_layer8_3
        output_layer9 = torch.cat([output_layer9, output_layer9_3], dim=1)
        output_layer9_4 = self.conv94(output_layer8)+output_layer8_4
        output_layer9 = torch.cat([output_layer9, output_layer9_4], dim=1)
        output_layer9_5 = self.conv95(output_layer8)+output_layer8_5
        output_layer9 = torch.cat([output_layer9, output_layer9_5], dim=1)
        output_layer9_6 = self.conv96(output_layer8)+output_layer8_6
        output_layer9 = torch.cat([output_layer9, output_layer9_6], dim=1)
        output_layer9_7 = self.conv97(output_layer8)+output_layer8_7
        output_layer9 = torch.cat([output_layer9, output_layer9_7], dim=1)
        output_layer9_8 = self.conv98(output_layer8)+output_layer8_8
        output_layer9 = torch.cat([output_layer9, output_layer9_8], dim=1)

        # 第十层 卷积层
        output_layer10_1 = self.conv101(output_layer9)+output_layer9_1
        output_layer10 = output_layer10_1
        output_layer10_2 = self.conv102(output_layer9)+output_layer9_2
        output_layer10 = torch.cat([output_layer10, output_layer10_2], dim=1)
        output_layer10_3 = self.conv103(output_layer9)+output_layer9_3
        output_layer10 = torch.cat([output_layer10, output_layer10_3], dim=1)
        output_layer10_4 = self.conv104(output_layer9)+output_layer9_4
        output_layer10 = torch.cat([output_layer10, output_layer10_4], dim=1)
        output_layer10_5 = self.conv105(output_layer9)+output_layer9_5
        output_layer10 = torch.cat([output_layer10, output_layer10_5], dim=1)
        output_layer10_6 = self.conv106(output_layer9)+output_layer9_6
        output_layer10 = torch.cat([output_layer10, output_layer10_6], dim=1)
        output_layer10_7 = self.conv107(output_layer9)+output_layer9_7
        output_layer10 = torch.cat([output_layer10, output_layer10_7], dim=1)
        output_layer10_8 = self.conv108(output_layer9)+output_layer9_8
        output_layer10 = torch.cat([output_layer10, output_layer10_8], dim=1)

        # 第十一层 卷积层
        output_layer11_1 = self.conv111(output_layer10)+output_layer10_1
        output_layer11 = output_layer11_1
        output_layer11_2 = self.conv112(output_layer10)+output_layer10_2
        output_layer11 = torch.cat([output_layer11, output_layer11_2], dim=1)
        output_layer11_3 = self.conv113(output_layer10)+output_layer10_3
        output_layer11 = torch.cat([output_layer11, output_layer11_3], dim=1)
        output_layer11_4 = self.conv114(output_layer10)+output_layer10_4
        output_layer11 = torch.cat([output_layer11, output_layer11_4], dim=1)
        output_layer11_5 = self.conv115(output_layer10)+output_layer10_5
        output_layer11 = torch.cat([output_layer11, output_layer11_5], dim=1)
        output_layer11_6 = self.conv116(output_layer10)+output_layer10_6
        output_layer11 = torch.cat([output_layer11, output_layer11_6], dim=1)
        output_layer11_7 = self.conv117(output_layer10)+output_layer10_7
        output_layer11 = torch.cat([output_layer11, output_layer11_7], dim=1)
        output_layer11_8 = self.conv118(output_layer10)+output_layer10_8
        output_layer11 = torch.cat([output_layer11, output_layer11_8], dim=1)

        # 第十二层 卷积层
        output_layer12_1 = self.conv121(output_layer11)+output_layer11_1
        output_layer12 = output_layer12_1
        output_layer12_2 = self.conv122(output_layer11)+output_layer11_2
        output_layer12 = torch.cat([output_layer12, output_layer12_2], dim=1)
        output_layer12_3 = self.conv123(output_layer11)+output_layer11_3
        output_layer12 = torch.cat([output_layer12, output_layer12_3], dim=1)
        output_layer12_4 = self.conv124(output_layer11)+output_layer11_4
        output_layer12 = torch.cat([output_layer12, output_layer12_4], dim=1)
        output_layer12_5 = self.conv125(output_layer11)+output_layer11_5
        output_layer12 = torch.cat([output_layer12, output_layer12_5], dim=1)
        output_layer12_6 = self.conv126(output_layer11)+output_layer11_6
        output_layer12 = torch.cat([output_layer12, output_layer12_6], dim=1)
        output_layer12_7 = self.conv127(output_layer11)+output_layer11_7
        output_layer12 = torch.cat([output_layer12, output_layer12_7], dim=1)
        output_layer12_8 = self.conv128(output_layer11)+output_layer11_8
        output_layer12 = torch.cat([output_layer12, output_layer12_8], dim=1)

        # 第十三层 卷积层
        output_layer13_1 = self.conv131(output_layer12)+output_layer12_1
        output_layer13 = output_layer13_1
        output_layer13_2 = self.conv132(output_layer12)+output_layer12_2
        output_layer13 = torch.cat([output_layer13, output_layer13_2], dim=1)
        output_layer13_3 = self.conv133(output_layer12)+output_layer12_3
        output_layer13 = torch.cat([output_layer13, output_layer13_3], dim=1)
        output_layer13_4 = self.conv134(output_layer12)+output_layer12_4
        output_layer13 = torch.cat([output_layer13, output_layer13_4], dim=1)
        output_layer13_5 = self.conv135(output_layer12)+output_layer12_5
        output_layer13 = torch.cat([output_layer13, output_layer13_5], dim=1)
        output_layer13_6 = self.conv136(output_layer12)+output_layer12_6
        output_layer13 = torch.cat([output_layer13, output_layer13_6], dim=1)
        output_layer13_7 = self.conv137(output_layer12)+output_layer12_7
        output_layer13 = torch.cat([output_layer13, output_layer13_7], dim=1)
        output_layer13_8 = self.conv138(output_layer12)+output_layer12_8
        output_layer13 = torch.cat([output_layer13, output_layer13_8], dim=1)

        # 第十四层 卷积层
        output_layer14_1 = self.conv141(output_layer13)+output_layer13_1
        output_layer14 = output_layer14_1
        output_layer14_2 = self.conv142(output_layer13)+output_layer13_2
        output_layer14 = torch.cat([output_layer14, output_layer14_2], dim=1)
        output_layer14_3 = self.conv143(output_layer13)+output_layer13_3
        output_layer14 = torch.cat([output_layer14, output_layer14_3], dim=1)
        output_layer14_4 = self.conv144(output_layer13)+output_layer13_4
        output_layer14 = torch.cat([output_layer14, output_layer14_4], dim=1)
        output_layer14_5 = self.conv145(output_layer13)+output_layer13_5
        output_layer14 = torch.cat([output_layer14, output_layer14_5], dim=1)
        output_layer14_6 = self.conv146(output_layer13)+output_layer13_6
        output_layer14 = torch.cat([output_layer14, output_layer14_6], dim=1)
        output_layer14_7 = self.conv147(output_layer13)+output_layer13_7
        output_layer14 = torch.cat([output_layer14, output_layer14_7], dim=1)
        output_layer14_8 = self.conv148(output_layer13)+output_layer13_8
        output_layer14 = torch.cat([output_layer14, output_layer14_8], dim=1)

        # 第十五层 线性层 特征映射
        output_layer15_1 = output_layer14_1.view(1, -1)
        output_layer15_1 = self.fc151(output_layer15_1)+output_layer15_1
        output_layer15 = output_layer15_1

        output_layer15_2 = output_layer14_2.view(1, -1)
        output_layer15_2 = self.fc152(output_layer15_2)+output_layer15_2
        output_layer15 = torch.cat([output_layer15, output_layer15_2], dim=1)

        output_layer15_3 = output_layer14_3.view(1, -1)
        output_layer15_3 = self.fc153(output_layer15_3)+output_layer15_3
        output_layer15 = torch.cat([output_layer15, output_layer15_3], dim=1)

        output_layer15_4 = output_layer14_4.view(1, -1)
        output_layer15_4 = self.fc154(output_layer15_4)+output_layer15_4
        output_layer15 = torch.cat([output_layer15, output_layer15_4], dim=1)

        output_layer15_5 = output_layer14_5.view(1, -1)
        output_layer15_5 = self.fc155(output_layer15_5)+output_layer15_5
        output_layer15 = torch.cat([output_layer15, output_layer15_5], dim=1)

        output_layer15_6 = output_layer14_6.view(1, -1)
        output_layer15_6 = self.fc156(output_layer15_6)+output_layer15_6
        output_layer15 = torch.cat([output_layer15, output_layer15_6], dim=1)

        output_layer15_7 = output_layer14_7.view(1, -1)
        output_layer15_7 = self.fc157(output_layer15_7)+output_layer15_7
        output_layer15 = torch.cat([output_layer15, output_layer15_7], dim=1)

        output_layer15_8 = output_layer14_8.view(1, -1)
        output_layer15_8 = self.fc158(output_layer15_8)+output_layer15_8
        output_layer15 = torch.cat([output_layer15, output_layer15_8], dim=1)

        output_layer15_9 = output_layer14.view(1, -1)
        output_layer15_9 = self.fc159(output_layer15_9)

        # 第十六层 线性层 特征映射
        output_layer16_1 = self.fc161(output_layer15_1)+output_layer15_1
        output_layer16 = output_layer16_1

        output_layer16_2 = self.fc162(output_layer15_2)+output_layer15_2
        output_layer16 = torch.cat([output_layer16, output_layer16_2], dim=1)

        output_layer16_3 = self.fc163(output_layer15_3)+output_layer15_3
        output_layer16 = torch.cat([output_layer16, output_layer16_3], dim=1)

        output_layer16_4 = self.fc164(output_layer15_4)+output_layer15_4
        output_layer16 = torch.cat([output_layer16, output_layer16_4], dim=1)

        output_layer16_5 = self.fc165(output_layer15_5)+output_layer15_5
        output_layer16 = torch.cat([output_layer16, output_layer16_5], dim=1)

        output_layer16_6 = self.fc166(output_layer15_6)+output_layer15_6
        output_layer16 = torch.cat([output_layer16, output_layer16_6], dim=1)

        output_layer16_7 = self.fc167(output_layer15_7)+output_layer15_7
        output_layer16 = torch.cat([output_layer16, output_layer16_7], dim=1)

        output_layer16_8 = self.fc168(output_layer15_8)+output_layer15_8
        output_layer16 = torch.cat([output_layer16, output_layer16_8], dim=1)

        output_layer16_9 = self.fc169(output_layer15)+output_layer15_9

        output_layer16_10 = self.fc1610(output_layer15_9)

        # 第十七层 线性层
        output_layer17_1 = output_layer16.view(1, -1)
        output_layer17_1 = self.fc171(output_layer17_1)
        output_layer17_2 = output_layer16_9.view(1, -1)
        output_layer17_2 = self.fc172(output_layer17_2)
        output_layer17 = torch.cat([output_layer17_1, output_layer17_2], dim=1)
        output_layer17_3 = output_layer16_10.view(1, -1)
        output_layer17_3 = self.fc173(output_layer17_3)
        output_layer17 = torch.cat([output_layer17, output_layer17_3], dim=1)

        # 第十八层 线性层
        output_layer18 = output_layer17.view(1, -1)
        output_layer18 = self.fc18(output_layer18)

        # 第十四层 输出层
        output = output_layer18
        return output


class Deep_Net_2_M1d_Res(nn.Module):
    def __init__(self, BATCH_LEN):
        super(Deep_Net_2_M1d_Res, self).__init__()

        self.conv11 = TripleConv1d(2, 1)
        self.conv12 = QuadrupleConv1d(2, 1)
        self.conv13 = TripleConv1d(2, 1)
        self.conv14 = QuadrupleConv1d(2, 1)
        self.conv15 = TripleConv1d(2, 1)

        self.conv21 = QuadrupleConv1d(5, 5)
        self.conv22 = TripleConv1d(5, 1)

        self.conv30 = QuadrupleConv1d(10, 2)
        self.conv31 = TripleConv1d(2, 1)
        self.conv32 = QuadrupleConv1d(2, 1)
        self.conv33 = TripleConv1d(2, 1)
        self.conv34 = QuadrupleConv1d(2, 1)
        self.conv35 = TripleConv1d(2, 1)

        self.lstm40 = nn.LSTM(np.int(BATCH_LEN*2), BATCH_LEN, 3)
        self.lstm41 = nn.LSTM(BATCH_LEN, np.int(BATCH_LEN/5), 3)
        self.lstm42 = nn.LSTM(BATCH_LEN, np.int(BATCH_LEN/5), 3)
        self.lstm43 = nn.LSTM(BATCH_LEN, np.int(BATCH_LEN/5), 3)
        self.lstm44 = nn.LSTM(BATCH_LEN, np.int(BATCH_LEN/5), 3)
        self.lstm45 = nn.LSTM(BATCH_LEN, np.int(BATCH_LEN/5), 3)

        self.lstm50 = nn.LSTM(BATCH_LEN*2, BATCH_LEN, 3)
        self.lstm51 = nn.LSTM(BATCH_LEN+np.int(BATCH_LEN/5), np.int(BATCH_LEN/5), 3)
        self.lstm52 = nn.LSTM(BATCH_LEN+np.int(BATCH_LEN/5), np.int(BATCH_LEN/5), 3)
        self.lstm53 = nn.LSTM(BATCH_LEN+np.int(BATCH_LEN/5), np.int(BATCH_LEN/5), 3)
        self.lstm54 = nn.LSTM(BATCH_LEN+np.int(BATCH_LEN/5), np.int(BATCH_LEN/5), 3)
        self.lstm55 = nn.LSTM(BATCH_LEN+np.int(BATCH_LEN/5), np.int(BATCH_LEN/5), 3)

        self.fc60 = nn.Linear(BATCH_LEN, BATCH_LEN)
        self.fc61 = nn.Linear(np.int(BATCH_LEN/5), np.int(BATCH_LEN/5))
        self.fc62 = nn.Linear(np.int(BATCH_LEN/5), np.int(BATCH_LEN/5))
        self.fc63 = nn.Linear(np.int(BATCH_LEN/5), np.int(BATCH_LEN/5))
        self.fc64 = nn.Linear(np.int(BATCH_LEN/5), np.int(BATCH_LEN/5))
        self.fc65 = nn.Linear(np.int(BATCH_LEN/5), np.int(BATCH_LEN/5))

        self.lstm7 = nn.LSTM(BATCH_LEN*2, BATCH_LEN, 3)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]

        self.fc8 = nn.Linear(BATCH_LEN, BATCH_LEN)  # 线性拟合，输入[x,60]  输出[x,60]
        # self.dropout = nn.Dropout(p=0.5)
        self._init_weight()

    # 权重初始化
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_well, seismic, BATCH_LEN):

        # 0706的设计一个深度复杂的网络
        # 第一层 卷积层 提取地震数据与井数据的共同特征
        seismic_in = seismic.view(1, 1, -1)
        layer_in_1_1 = train_well[:, 0, :]
        layer_in_1_1 = torch.cat([layer_in_1_1.view(1, 1, -1), seismic_in], dim=1)
        # layer_in_1_1 = layer_in_1_1.view(layer_in_1_1.size(0), layer_in_1_1.size(1), -1, 10)
        output_layer1_1 = self.conv11(layer_in_1_1)+train_well[:, 0, :]
        # output_layer1_1 = self.dropout(output_layer1_1)
        output_layer1 = output_layer1_1

        layer_in_1_2 = train_well[:, 1, :]
        layer_in_1_2 = torch.cat([layer_in_1_2.view(1, 1, -1), seismic_in], dim=1)
        # layer_in_1_2 = layer_in_1_2.view(layer_in_1_2.size(0), layer_in_1_2.size(1), -1, 10)
        output_layer1_2 = self.conv12(layer_in_1_2)+train_well[:, 1, :]
        # output_layer1_2 = self.dropout(output_layer1_2)
        output_layer1 = torch.cat([output_layer1, output_layer1_2], dim=1)

        layer_in_1_3 = train_well[:, 2, :]
        layer_in_1_3 = torch.cat([layer_in_1_3.view(1, 1, -1), seismic_in], dim=1)
        # layer_in_1_3 = layer_in_1_3.view(layer_in_1_3.size(0), layer_in_1_3.size(1), -1, 10)
        output_layer1_3 = self.conv13(layer_in_1_3)+train_well[:, 2, :]
        # output_layer1_3 = self.dropout(output_layer1_3)
        output_layer1 = torch.cat([output_layer1, output_layer1_3], dim=1)

        layer_in_1_4 = train_well[:, 3, :]
        layer_in_1_4 = torch.cat([layer_in_1_4.view(1, 1, -1), seismic_in], dim=1)
        # layer_in_1_4 = layer_in_1_4.view(layer_in_1_4.size(0), layer_in_1_4.size(1), -1, 10)
        output_layer1_4 = self.conv14(layer_in_1_4)+train_well[:, 3, :]
        # output_layer1_4 = self.dropout(output_layer1_4)
        output_layer1 = torch.cat([output_layer1, output_layer1_4], dim=1)

        layer_in_1_5 = train_well[:, 4, :]
        layer_in_1_5 = torch.cat([layer_in_1_5.view(1, 1, -1), seismic_in], dim=1)
        # layer_in_1_5 = layer_in_1_5.view(layer_in_1_5.size(0), layer_in_1_5.size(1), -1, 10)
        output_layer1_5 = self.conv15(layer_in_1_5)+train_well[:, 4, :]
        # output_layer1_5 = self.dropout(output_layer1_5)
        output_layer1 = torch.cat([output_layer1, output_layer1_5], dim=1)

        # 第二层 卷积层 提取井数据的特征
        layer_in_2_1 = train_well
        output_layer2_1 = self.conv21(layer_in_2_1)+train_well
        # output_layer2_1 = self.dropout(output_layer2_1)

        layer_in_2_2 = train_well
        output_layer2_2 = self.conv22(layer_in_2_2)
        # output_layer2_2 = self.dropout(output_layer2_2)

        # 第三层 卷积层 提取井数据的特征
        layer_in_3_0 = torch.cat([output_layer1, output_layer2_1], dim=1)
        output_layer3_0 = self.conv30(layer_in_3_0)
        # output_layer3_0 = self.dropout(output_layer3_0)

        layer_in_3_1 = torch.cat([output_layer1_1, output_layer2_2], dim=1)
        output_layer3_1 = self.conv31(layer_in_3_1)+output_layer1_1
        # output_layer3_1 = self.dropout(output_layer3_1)

        layer_in_3_2 = torch.cat([output_layer1_2, output_layer2_2], dim=1)
        output_layer3_2 = self.conv32(layer_in_3_2)+output_layer1_2
        # output_layer3_2 = self.dropout(output_layer3_2)

        layer_in_3_3 = torch.cat([output_layer1_3, output_layer2_2], dim=1)
        output_layer3_3 = self.conv33(layer_in_3_3)+output_layer1_3
        # output_layer3_3 = self.dropout(output_layer3_3)

        layer_in_3_4 = torch.cat([output_layer1_4, output_layer2_2], dim=1)
        output_layer3_4 = self.conv34(layer_in_3_4)+output_layer1_4
        # output_layer3_4 = self.dropout(output_layer3_4)

        layer_in_3_5 = torch.cat([output_layer1_5, output_layer2_2], dim=1)
        output_layer3_5 = self.conv35(layer_in_3_5)+output_layer1_5
        # output_layer3_5 = self.dropout(output_layer3_5)

        # 第四层 LSTM 提取井数据的特征
        layer_in_4_0 = output_layer3_0.view(1, output_layer3_0.size(0), -1)
        output_layer4_0, temp4_0 = self.lstm40(layer_in_4_0)

        layer_in_4_1 = output_layer3_1.view(1, output_layer3_1.size(0), -1)
        output_layer4_1, temp4_1 = self.lstm41(layer_in_4_1)
        output_layer4 = output_layer4_1

        layer_in_4_2 = output_layer3_2.view(1, output_layer3_2.size(0), -1)
        output_layer4_2, temp4_2 = self.lstm42(layer_in_4_2)
        output_layer4 = torch.cat([output_layer4, output_layer4_2], dim=2)

        layer_in_4_3 = output_layer3_3.view(1, output_layer3_3.size(0), -1)
        output_layer4_3, temp4_3 = self.lstm43(layer_in_4_3)
        output_layer4 = torch.cat([output_layer4, output_layer4_3], dim=2)

        layer_in_4_4 = output_layer3_4.view(1, output_layer3_4.size(0), -1)
        output_layer4_4, temp4_4 = self.lstm44(layer_in_4_4)
        output_layer4 = torch.cat([output_layer4, output_layer4_4], dim=2)

        layer_in_4_5 = output_layer3_5.view(1, output_layer3_5.size(0), -1)
        output_layer4_5, temp4_5 = self.lstm42(layer_in_4_5)
        output_layer4 = torch.cat([output_layer4, output_layer4_5], dim=2)

        # 第五层 LSTM 特征映射
        layer_in_5_0 = torch.cat([output_layer4_0, output_layer4], dim=2)
        output_layer5_0, temp5_0 = self.lstm50(layer_in_5_0)

        layer_in_5_1 = torch.cat([output_layer4_0, output_layer4_1], dim=2)
        output_layer5_1, temp5_1 = self.lstm51(layer_in_5_1)
        output_layer5_1 = output_layer5_1+output_layer4_1

        layer_in_5_2 = torch.cat([output_layer4_0, output_layer4_2], dim=2)
        output_layer5_2, temp5_2 = self.lstm52(layer_in_5_2)
        output_layer5_2 = output_layer5_2+output_layer4_2

        layer_in_5_3 = torch.cat([output_layer4_0, output_layer4_3], dim=2)
        output_layer5_3, temp5_3 = self.lstm53(layer_in_5_3)
        output_layer5_3 = output_layer5_3+output_layer4_3

        layer_in_5_4 = torch.cat([output_layer4_0, output_layer4_4], dim=2)
        output_layer5_4, temp5_4 = self.lstm54(layer_in_5_4)
        output_layer5_4 = output_layer5_4+output_layer4_4

        layer_in_5_5 = torch.cat([output_layer4_0, output_layer4_5], dim=2)
        output_layer5_5, temp5_5 = self.lstm55(layer_in_5_5)
        output_layer5_5 = output_layer5_5+output_layer4_5

        # 第六层 线性层 特征映射
        output_layer6_0 = output_layer5_0.view(1, -1)
        output_layer6_0 = self.fc60(output_layer6_0)

        output_layer6_1 = output_layer5_1.view(1, -1)
        output_layer6_1 = self.fc61(output_layer6_1)+output_layer6_1
        output_layer6 = output_layer6_1

        output_layer6_2 = output_layer5_2.view(1, -1)
        output_layer6_2 = self.fc62(output_layer6_2)+output_layer6_2
        output_layer6 = torch.cat([output_layer6, output_layer6_2], dim=1)

        output_layer6_3 = output_layer5_3.view(1, -1)
        output_layer6_3 = self.fc63(output_layer6_3)+output_layer6_3
        output_layer6 = torch.cat([output_layer6, output_layer6_3], dim=1)

        output_layer6_4 = output_layer5_4.view(1, -1)
        output_layer6_4 = self.fc64(output_layer6_4)+output_layer6_4
        output_layer6 = torch.cat([output_layer6, output_layer6_4], dim=1)

        output_layer6_5 = output_layer5_5.view(1, -1)
        output_layer6_5 = self.fc65(output_layer6_5)+output_layer6_5
        output_layer6 = torch.cat([output_layer6, output_layer6_5], dim=1)

        # 第七层 LSTM 特征映射
        layer_in_7 = torch.cat([output_layer6_0, output_layer6], dim=1)
        layer_in_7 = layer_in_7.view(1, 1, -1)
        output_layer7, temp7 = self.lstm7(layer_in_7)

        # 第八层 线性层 特征映射
        layer_in_8 = output_layer7.view(1, -1)
        output8 = self.fc8(layer_in_8)
        output = output8
        return output


class BSsequential_net(nn.Module):
    def __init__(self, BATCH_LEN, num_well):
        super(BSsequential_net, self).__init__()

        self.conv1 = TripleConv1d(6, 8)
        self.conv2 = QuadrupleConv1d(8, 10)
        self.conv3 = TripleConv1d(10, 12)
        self.fc30 = nn.Linear(12 * BATCH_LEN, 12 * BATCH_LEN)
        self.conv4 = QuadrupleConv1d(12, 14)
        self.conv5 = TripleConv1d(14, 16)
        self.conv6 = QuadrupleConv1d(16, 18)
        self.fc60 = nn.Linear(18 * BATCH_LEN, 18 * BATCH_LEN)
        self.conv7 = TripleConv1d(18, 20)
        self.conv8 = QuadrupleConv1d(20, 18)
        self.conv9 = TripleConv1d(18, 16)
        self.fc90 = nn.Linear(16 * BATCH_LEN, 16 * BATCH_LEN)
        self.conv10 = QuadrupleConv1d(16, 14)
        self.conv11 = TripleConv1d(14, 12)
        self.conv12 = QuadrupleConv1d(12, 10)
        self.fc120 = nn.Linear(10 * BATCH_LEN, 10 * BATCH_LEN)
        self.conv13 = TripleConv1d(10, 8)
        self.conv14 = QuadrupleConv1d(8, 6)
        self.conv15 = TripleConv1d(6, 4)
        self.fc150 = nn.Linear(4 * BATCH_LEN, 4 * BATCH_LEN)
        self.conv16 = QuadrupleConv1d(4, 2)
        self.conv17 = TripleConv1d(2, 1)
        self.fc170 = nn.Linear(BATCH_LEN, BATCH_LEN)  # 线性拟合，输入[x,60]  输出[x,60]
        # self.dropout = nn.Dropout(p=0.5)
        self._init_weight()

    # 权重初始化
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_well, seismic):

        # 0727的按厍师兄的想法，设计一个顺序的残差N型网
        input1 = torch.cat([train_well, torch.unsqueeze(seismic, 1)], dim=1)
        BATCH_SIZE = input1.size(0)
        BATCH_LEN = input1.size(2)

        output1 = self.conv1(input1)
        output2 = self.conv2(output1)
        output3 = self.conv3(output2)
        input3_0 = output3.view(BATCH_SIZE, -1)
        output3_0 = self.fc30(input3_0)
        output3_0 = output3_0 + input3_0
        output3_0 = output3_0.view(BATCH_SIZE, -1, BATCH_LEN)

        output4 = self.conv4(output3_0)
        output5 = self.conv5(output4)
        output6 = self.conv6(output5)
        input6_0 = output6.view(BATCH_SIZE, -1)
        output6_0 = self.fc60(input6_0)
        output6_0 = output6_0 + input6_0
        output6_0 = output6_0.view(BATCH_SIZE, -1, BATCH_LEN)

        output7 = self.conv7(output6_0)
        output8 = self.conv8(output7) + output6
        output9 = self.conv9(output8) + output5
        input9_0 = output9.view(BATCH_SIZE, -1)
        output9_0 = self.fc90(input9_0)
        output9_0 = output9_0 + input9_0
        output9_0 = output9_0.view(BATCH_SIZE, -1, BATCH_LEN)

        output10 = self.conv10(output9_0) + output4
        output11 = self.conv11(output10) + output3
        output12 = self.conv12(output11) + output2
        input12_0 = output12.view(BATCH_SIZE, -1)
        output12_0 = self.fc120(input12_0)
        output12_0 = output12_0 + input12_0
        output12_0 = output12_0.view(BATCH_SIZE, -1, BATCH_LEN)

        output13 = self.conv13(output12_0) + output1
        output14 = self.conv14(output13) + input1
        output15 = self.conv15(output14)
        input15_0 = output15.view(BATCH_SIZE, -1)
        output15_0 = self.fc150(input15_0)
        output15_0 = output15_0 + input15_0
        output15_0 = output15_0.view(BATCH_SIZE, -1, BATCH_LEN)

        output16 = self.conv16(output15_0)
        output17 = self.conv17(output16)
        input17_0 = output17.view(BATCH_SIZE, -1)
        output18 = self.fc170(input17_0)
        output = output18

        return output


class BSsequential_net_hu(nn.Module):
    def __init__(self, BATCH_LEN, num_well):
        super(BSsequential_net_hu, self).__init__()

        self.conv1_1 = TripleConv1d(num_well + 1, num_well + 2)
        self.conv1_2 = QuadrupleConv1d(BATCH_LEN, BATCH_LEN + 20)
        self.fc1_3 = nn.Linear((num_well + 2) * (BATCH_LEN + 20), (num_well + 2) * (BATCH_LEN + 20))

        self.conv2_1 = TripleConv1d(num_well + 2, num_well + 3)
        self.conv2_2 = QuadrupleConv1d(BATCH_LEN + 20, BATCH_LEN + 40)
        self.fc2_3 = nn.Linear((num_well + 3) * (BATCH_LEN + 40), (num_well + 3) * (BATCH_LEN + 40))

        self.conv3_1 = TripleConv1d(num_well + 3, num_well + 4)
        self.conv3_2 = QuadrupleConv1d(BATCH_LEN + 40, BATCH_LEN + 60)
        self.fc3_3 = nn.Linear((num_well + 4) * (BATCH_LEN + 60), (num_well + 4) * (BATCH_LEN + 60))

        self.conv4_1 = TripleConv1d(num_well + 4, num_well + 5)
        self.conv4_2 = QuadrupleConv1d(BATCH_LEN + 60, BATCH_LEN + 80)
        self.fc4_3 = nn.Linear((num_well + 5) * (BATCH_LEN + 80), (num_well + 5) * (BATCH_LEN + 80))

        self.conv5_1 = TripleConv1d(num_well + 5, num_well + 6)
        self.conv5_2 = QuadrupleConv1d(BATCH_LEN + 80, BATCH_LEN + 100)
        self.fc5_3 = nn.Linear((num_well + 6) * (BATCH_LEN + 100), (num_well + 6) * (BATCH_LEN + 100))

        self.conv6_1 = QuadrupleConv1d(BATCH_LEN + 100, BATCH_LEN + 80)
        self.conv6_2 = TripleConv1d(num_well + 6, num_well + 5)
        self.fc6_3 = nn.Linear((num_well + 5) * (BATCH_LEN + 80), (num_well + 5) * (BATCH_LEN + 80))

        self.conv7_1 = QuadrupleConv1d(BATCH_LEN + 80, BATCH_LEN + 60)
        self.conv7_2 = TripleConv1d(num_well + 5, num_well + 4)
        self.fc7_3 = nn.Linear((num_well + 4) * (BATCH_LEN + 60), (num_well + 4) * (BATCH_LEN + 60))

        self.conv8_1 = QuadrupleConv1d(BATCH_LEN + 60, BATCH_LEN + 40)
        self.conv8_2 = TripleConv1d(num_well + 4, num_well + 3)
        self.fc8_3 = nn.Linear((num_well + 3) * (BATCH_LEN + 40), (num_well + 3) * (BATCH_LEN + 40))

        self.conv9_1 = QuadrupleConv1d(BATCH_LEN + 40, BATCH_LEN + 20)
        self.conv9_2 = TripleConv1d(num_well + 3, num_well + 2)
        self.fc9_3 = nn.Linear((num_well + 2) * (BATCH_LEN + 20), (num_well + 2) * (BATCH_LEN + 20))

        self.conv10_1 = QuadrupleConv1d(BATCH_LEN + 20, BATCH_LEN)
        self.conv10_2 = TripleConv1d(num_well + 2, num_well + 1)
        self.fc10_3 = nn.Linear((num_well + 1) * BATCH_LEN, (num_well + 1) * BATCH_LEN)

        self.conv11_1 = TripleConv1d(num_well + 1, num_well-1)
        self.fc11_3 = nn.Linear((num_well-1) * BATCH_LEN, (num_well-1) * BATCH_LEN)

        self.conv12_1 = TripleConv1d(num_well - 1, num_well - 2)
        self.fc12_3 = nn.Linear((num_well-2) * BATCH_LEN, (num_well-2) * BATCH_LEN)

        self.conv13_1 = TripleConv1d(num_well - 2, 2)
        self.fc13_3 = nn.Linear(2*BATCH_LEN, BATCH_LEN)

        self.fc14 = nn.Linear(BATCH_LEN, BATCH_LEN)

        self._init_weight()

    # 权重初始化
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_well, seismic,):

        # 0727的按厍师兄的想法，设计一个顺序的残差N型网
        input1 = torch.cat([train_well, torch.unsqueeze(seismic, 1)], dim=1)    # 6*100
        output1_1 = self.conv1_1(input1)    # 7*100
        output1_2 = torch.transpose(output1_1, dim0=1, dim1=2)  # 100*7
        output1_3 = self.conv1_2(output1_2)  # 120*7
        a, b, c = output1_3.shape
        output1_4 = output1_3.view(a, -1)  # 1*840
        output1_5 = self.fc1_3(output1_4) + output1_4  # 1*840

        input2 = output1_5.view(a, c, b)    # 7*120
        output2_1 = self.conv2_1(input2)    # 8*120
        output2_2 = torch.transpose(output2_1, dim0=1, dim1=2)  # 120*8
        output2_3 = self.conv2_2(output2_2)  # 140*8
        a, b, c = output2_3.shape
        output2_4 = output2_3.view(a, -1)  # 1*1120
        output2_5 = self.fc2_3(output2_4) + output2_4  # 1*1120

        input3 = output2_5.view(a, c, b)  # 8*140
        output3_1 = self.conv3_1(input3)  # 9*140
        output3_2 = torch.transpose(output3_1, dim0=1, dim1=2)  # 140*9
        output3_3 = self.conv3_2(output3_2)  # 160*9
        a, b, c = output3_3.shape
        output3_4 = output3_3.view(a, -1)  # 1*1440
        output3_5 = self.fc3_3(output3_4) + output3_4  # 1*1440

        input4 = output3_5.view(a, c, b)  # 9*160
        output4_1 = self.conv4_1(input4)  # 10*160
        output4_2 = torch.transpose(output4_1, dim0=1, dim1=2)  # 160*10
        output4_3 = self.conv4_2(output4_2)  # 180*10
        a, b, c = output4_3.shape
        output4_4 = output4_3.view(a, -1)  # 1*1800
        output4_5 = self.fc4_3(output4_4) + output4_4  # 1*1800

        input5 = output4_5.view(a, c, b)  # 10*180
        output5_1 = self.conv5_1(input5)  # 11*180
        output5_2 = torch.transpose(output5_1, dim0=1, dim1=2)  # 180*11
        output5_3 = self.conv5_2(output5_2)  # 200*11
        a, b, c = output5_3.shape
        output5_4 = output5_3.view(a, -1)  # 1*2200
        output5_5 = self.fc5_3(output5_4) + output5_4  # 1*2200

        input6 = output5_5.view(a, b, c) + output5_3  # 200*11
        output6 = self.conv6_1(input6)  # 180*11
        output6 = torch.transpose(output6, dim0=1, dim1=2) + output5_1  # 11*180
        output6 = self.conv6_2(output6) + input5  # 10*180
        a, b, c = output6.shape
        output6 = output6.view(a, -1)  # 1*1800
        output6 = self.fc6_3(output6) + output6  # 1*1800

        input7 = output6.view(a, c, b) + output4_3  # 180*10
        output7 = self.conv7_1(input7)  # 160*10
        output7 = torch.transpose(output7, dim0=1, dim1=2) + output4_1  # 10*160
        output7 = self.conv7_2(output7) + input4  # 9*160
        a, b, c = output7.shape
        output7 = output7.view(a, -1)  # 1*1440
        output7 = self.fc7_3(output7) + output7  # 1*1440

        input8 = output7.view(a, c, b) + output3_3  # 160*9
        output8 = self.conv8_1(input8)  # 140*9
        output8 = torch.transpose(output8, dim0=1, dim1=2) + output3_1  # 9*140
        output8 = self.conv8_2(output8) + input3  # 8*140
        a, b, c = output8.shape
        output8 = output8.view(a, -1)  # 1*1120
        output8 = self.fc8_3(output8) + output8  # 1*1120

        input9 = output8.view(a, c, b) + output2_3  # 140*8
        output9 = self.conv9_1(input9)  # 120*8
        output9 = torch.transpose(output9, dim0=1, dim1=2) + output2_1  # 8*120
        output9 = self.conv9_2(output9) + input2  # 7*120
        a, b, c = output9.shape
        output9 = output9.view(a, -1)  # 1*840
        output9 = self.fc9_3(output9) + output9  # 1*840

        input10 = output9.view(a, c, b) + output1_3  # 120*7
        output10 = self.conv10_1(input10)  # 100*7
        output10 = torch.transpose(output10, dim0=1, dim1=2) + output1_1  # 7*100
        output10 = self.conv10_2(output10) + input1  # 6*100
        a, b, c = output10.shape
        output10 = output10.view(a, -1)  # 1*600
        output10 = self.fc10_3(output10) + output10  # 1*600

        input11 = output10.view(a, b, c)
        output11 = self.conv11_1(input11)
        a, b, c = output11.shape
        output11 = output11.view(a, -1)
        output11 = self.fc11_3(output11) + output11

        input12 = output11.view(a, b, c)
        output12 = self.conv12_1(input12)
        a, b, c = output12.shape
        output12 = output12.view(a, -1)
        output12 = self.fc12_3(output12) + output12

        input13 = output12.view(a, b, c)
        output13 = self.conv13_1(input13)
        output13 = output13.view(a, -1)
        output13 = self.fc13_3(output13)

        output14 = self.fc14(output13)

        output = output14

        return output


class BSsequential_net_lstm(nn.Module):
    def __init__(self):
        super(BSsequential_net_lstm, self).__init__()

        self.conv1 = TripleConv1d(6, 8)
        self.conv2 = QuadrupleConv1d(8, 10)
        self.conv3 = TripleConv1d(10, 12)
        self.lstm30 = nn.LSTM(12 * 30, 12 * 30, 1)
        self.conv4 = QuadrupleConv1d(12, 14)
        self.conv5 = TripleConv1d(14, 16)
        self.conv6 = QuadrupleConv1d(16, 18)
        self.lstm60 = nn.LSTM(18 * 30, 18 * 30, 1)
        self.conv7 = TripleConv1d(18, 20)
        self.conv8 = QuadrupleConv1d(20, 18)
        self.conv9 = TripleConv1d(18, 16)
        self.lstm90 = nn.LSTM(16 * 30, 16 * 30, 1)
        self.conv10 = QuadrupleConv1d(16, 14)
        self.conv11 = TripleConv1d(14, 12)
        self.conv12 = QuadrupleConv1d(12, 10)
        self.lstm120 = nn.LSTM(10 * 30, 10 * 30, 1)
        self.conv13 = TripleConv1d(10, 8)
        self.conv14 = QuadrupleConv1d(8, 6)
        self.conv15 = TripleConv1d(6, 4)
        self.lstm150 = nn.LSTM(4 * 30, 4 * 30, 1)
        self.conv16 = QuadrupleConv1d(4, 2)
        self.conv17 = TripleConv1d(2, 1)
        self.fc170 = nn.Linear(30, 30)  # 线性拟合，输入[x,60]  输出[x,60]
        # self.dropout = nn.Dropout(p=0.5)
        self._init_weight()

    # 权重初始化
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_well, seismic):

        # 0727的按厍师兄的想法，设计一个顺序的残差N型网
        input1 = torch.cat([train_well, seismic.view(1, 1, -1)], dim=1)

        output1 = self.conv1(input1)
        output2 = self.conv2(output1)
        output3 = self.conv3(output2)
        input3_0 = output3.view(1, 1, -1)
        output3_0, temp3_0 = self.lstm30(input3_0)
        output3_0 = output3_0 + input3_0
        output3_0 = output3_0.view(1, -1, 30)

        output4 = self.conv4(output3_0)
        output5 = self.conv5(output4)
        output6 = self.conv6(output5)
        input6_0 = output6.view(1, 1, -1)
        output6_0, temp6_0 = self.lstm60(input6_0)
        output6_0 = output6_0 + input6_0
        output6_0 = output6_0.view(1, -1, 30)

        output7 = self.conv7(output6_0)
        output8 = self.conv8(output7) + output6
        output9 = self.conv9(output8) + output5
        input9_0 = output9.view(1, 1, -1)
        output9_0, temp9_0 = self.lstm90(input9_0)
        output9_0 = output9_0 + input9_0
        output9_0 = output9_0.view(1, -1, 30)

        output10 = self.conv10(output9_0) + output4
        output11 = self.conv11(output10) + output3
        output12 = self.conv12(output11) + output2
        input12_0 = output12.view(1, 1, -1)
        output12_0, temp12_0 = self.lstm120(input12_0)
        output12_0 = output12_0 + input12_0
        output12_0 = output12_0.view(1, -1, 30)

        output13 = self.conv13(output12_0) + output1
        output14 = self.conv14(output13) + input1
        output15 = self.conv15(output14)
        input15_0 = output15.view(1, 1, -1)
        output15_0, temp15_0 = self.lstm150(input15_0)
        output15_0 = output15_0 + input15_0
        output15_0 = output15_0.view(1, -1, 30)

        output16 = self.conv16(output15_0)
        output17 = self.conv17(output16)
        input17_0 = output17.view(1, -1)
        output18 = self.fc170(input17_0)
        output = output18

        return output


class BSsequential_net_seismic(nn.Module):
    def __init__(self, BATCH_LEN):
        super(BSsequential_net_seismic, self).__init__()

        self.conv1 = TripleConv1d_U(1, 2)
        self.conv2 = QuadrupleConv1d_U(2, 4)
        self.conv3 = TripleConv1d_U(4, 6)
        self.fc30 = nn.Linear(6 * BATCH_LEN, 6 * BATCH_LEN)
        self.conv4 = QuadrupleConv1d_U(6, 8)
        self.conv5 = TripleConv1d_U(8, 10)
        self.conv6 = QuadrupleConv1d_U(10, 12)
        self.fc60 = nn.Linear(12 * BATCH_LEN, 12 * BATCH_LEN)
        self.conv7 = TripleConv1d_U(12, 10)
        self.conv8 = QuadrupleConv1d_U(10, 8)
        self.conv9 = TripleConv1d_U(8, 6)
        self.fc90 = nn.Linear(6 * BATCH_LEN, 6 * BATCH_LEN)
        self.conv10 = QuadrupleConv1d_U(6, 4)
        self.conv11 = TripleConv1d_U(4, 2)
        self.conv12 = QuadrupleConv1d_U(2, 1)
        self.fc120 = nn.Linear(1 * BATCH_LEN, 1 * BATCH_LEN)
        # self.dropout = nn.Dropout(p=0.5)
        self._init_weight()

    # 权重初始化
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, seismic, BATCH_LEN):

        # 0727的按厍师兄的想法，设计一个顺序的残差N型网
        input1 = seismic.view(1, 1, -1)

        output1 = self.conv1(input1)
        output2 = self.conv2(output1)
        output3 = self.conv3(output2)
        input3_0 = output3.view(1, -1)
        output3_0 = self.fc30(input3_0)
        output3_0 = output3_0 + input3_0
        output3_0 = output3_0.view(1, -1, BATCH_LEN)

        output4 = self.conv4(output3_0)
        output5 = self.conv5(output4)
        output6 = self.conv6(output5)
        input6_0 = output6.view(1, -1)
        output6_0 = self.fc60(input6_0)
        output6_0 = output6_0 + input6_0
        output6_0 = output6_0.view(1, -1, BATCH_LEN)

        output7 = self.conv7(output6_0) + output5
        output8 = self.conv8(output7) + output4
        output9 = self.conv9(output8) + output3
        input9_0 = output9.view(1, -1)
        output9_0 = self.fc90(input9_0)
        output9_0 = output9_0 + input9_0
        output9_0 = output9_0.view(1, -1, BATCH_LEN)

        output10 = self.conv10(output9_0) + output2
        output11 = self.conv11(output10) + output1
        output12 = self.conv12(output11) + input1
        input12_0 = output12.view(1, -1)
        output12_0 = self.fc120(input12_0)

        output = output12_0

        return output


class BSsequential_net_D(nn.Module):
    def __init__(self):
        super(BSsequential_net_D, self).__init__()

        self.conv1 = TripleConv1d_U(6, 6)
        self.conv2 = QuadrupleConv1d_U(6, 6)
        self.conv3 = TripleConv1d_U(6, 6)
        self.fc30 = nn.Linear(6 * 60, 6 * 60)
        self.conv4 = QuadrupleConv1d_U(6, 5)
        self.conv5 = TripleConv1d_U(5, 5)
        self.conv6 = QuadrupleConv1d_U(5, 5)
        self.fc60 = nn.Linear(5 * 60, 5 * 60)
        self.conv7 = TripleConv1d_U(5, 4)
        self.conv8 = QuadrupleConv1d_U(4, 4)
        self.conv9 = TripleConv1d_U(4, 4)
        self.fc90 = nn.Linear(4 * 60, 4 * 60)
        self.conv10 = QuadrupleConv1d_U(4, 3)
        self.conv11 = TripleConv1d_U(3, 3)
        self.conv12 = QuadrupleConv1d_U(3, 3)
        self.fc120 = nn.Linear(3 * 60, 3 * 60)
        self.conv13 = TripleConv1d_U(3, 2)
        self.conv14 = QuadrupleConv1d_U(2, 2)
        self.conv15 = TripleConv1d_U(2, 2)
        self.fc150 = nn.Linear(2 * 60, 2 * 60)
        self.conv16 = QuadrupleConv1d_U(2, 1)
        self.conv17 = TripleConv1d_U(1, 1)
        self.conv18 = QuadrupleConv1d_U(1, 1)
        self.fc180 = nn.Linear(60, 60)  # 线性拟合，输入[x,60]  输出[x,60]
        # self.dropout = nn.Dropout(p=0.5)
        self._init_weight()

    # 权重初始化
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_well, seismic):

        # 0727的按厍师兄的想法，设计一个顺序的残差N型网
        input1 = torch.cat([train_well, seismic.view(1, 1, -1)], dim=1)

        output1 = self.conv1(input1)
        output2 = self.conv2(output1) + output1
        output3 = self.conv3(output2) + output1 + output2
        input3_0 = output3.view(1, -1)
        output3_0 = self.fc30(input3_0)
        output3_0 = output3_0.view(1, -1, 60)

        output4 = self.conv4(output3_0)
        output5 = self.conv5(output4) + output4
        output6 = self.conv6(output5) + output4 + output5
        input6_0 = output6.view(1, -1)
        output6_0 = self.fc60(input6_0)
        output6_0 = output6_0.view(1, -1, 60)

        output7 = self.conv7(output6_0)
        output8 = self.conv8(output7) + output7
        output9 = self.conv9(output8) + output7 + output8
        input9_0 = output9.view(1, -1)
        output9_0 = self.fc90(input9_0)
        output9_0 = output9_0.view(1, -1, 60)

        output10 = self.conv10(output9_0)
        output11 = self.conv11(output10) + output10
        output12 = self.conv12(output11) + output10 + output11
        input12_0 = output12.view(1, -1)
        output12_0 = self.fc120(input12_0)
        output12_0 = output12_0.view(1, -1, 60)

        output13 = self.conv13(output12_0)
        output14 = self.conv14(output13) + output13
        output15 = self.conv15(output14) + output13 + output14
        input15_0 = output15.view(1, -1)
        output15_0 = self.fc150(input15_0)
        output15_0 = output15_0.view(1, -1, 60)

        output16 = self.conv16(output15_0)
        output17 = self.conv17(output16) + output16
        output18 = self.conv18(output17) + output16 + output17
        input18_0 = output18.view(1, -1)
        output18 = self.fc180(input18_0)
        output = output18

        return output

