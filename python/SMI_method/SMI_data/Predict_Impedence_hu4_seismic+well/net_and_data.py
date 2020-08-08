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
        d, target = self.data[index, :], self.target[index, :]
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
        d, target = self.data[:, :], self.target[index, :]
        d = torch.from_numpy(d)
        target = torch.from_numpy(target)
        return d, target

    def __len__(self):
        return self.target.shape[0]


class ConvNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 20)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.lstm = nn.LSTM(180, 60, 1)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]
        self.fc1 = nn.Linear(3485, 60)  # 线性拟合   输入[x,3485]  输出[x,60]
        self.fc2 = nn.Linear(60, 60)  # 线性拟合，输入[x,60]  输出[x,60]

    def forward(self, seismic, well_seismic, well):  # 输入x为[batch_size， 350]

        seismic_first_size = seismic.size(0)
        seismic_second_size = seismic.size(1)
        well_seismic_first_size = well_seismic.size(0)
        well_first_size = well.size(0)

        seismic = seismic.view(seismic_first_size, seismic_second_size)
        well_seismic = well_seismic.view(seismic_first_size, 1, well_seismic_first_size, -1)
        well = well.view(seismic_first_size, 1, well_first_size, -1)

        well_seismic = self.conv1(well_seismic)  #
        well_seismic = well_seismic.view(seismic_first_size, -1)
        well_seismic = self.fc1(well_seismic)

        well = self.conv1(well)  #
        well = well.view(seismic_first_size, -1)
        well = self.fc1(well)

        out = torch.cat((well_seismic, seismic, well), 1)  # 在 1 维(纵向)进行拼接
        out = out.view(seismic_first_size, 1, -1)
        x1, x2 = self.lstm(out)
        a, b, c = x1.shape
        x1 = x1.view(-1, c)  # x1[5, 1000]
        out = self.fc2(x1)  # out[5, 350]
        return out


class ConvNet1_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 4, 1)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv2 = nn.Conv2d(4, 2, 1)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv3 = nn.Conv2d(2, 1, 1)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.lstm = nn.LSTM(60, 60, 1)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]
        self.fc1 = nn.Linear(5, 4)  # 线性拟合   输入[x,3485]  输出[x,60]
        self.fc2 = nn.Linear(4, 2)  # 线性拟合，输入[x,60]  输出[x,60]
        self.fc3 = nn.Linear(2, 1)  # 线性拟合，输入[x,60]  输出[x,60]

    def forward(self, train_dt, tempval):  # 输入x为[batch_size， 350]

        # 0516网络设计
        # train_dt = train_dt.view(1, train_dt.size(1), train_dt.size(0), -1)
        # output = self.conv1(train_dt)
        # output = F.relu(output, inplace=True)
        # output = self.conv2(output)
        # output = output.view(1, 1, -1)
        # output, y = self.lstm(output)
        # out = output.view(1, -1)
        # output = self.fc1(output)  # out[5, 350]
        # output = F.relu(output, inplace=True)
        # out = self.fc2(output)

        # 0522的一个线性层，三个卷积层网络
        train_dt = train_dt.view(1, train_dt.size(1), train_dt.size(0), -1)
        output = self.conv1(train_dt)
        output = F.relu(output, inplace=True)
        output = output.view(output.size(1), -1)
        output = output.t()

        # train_dt = train_dt.view(train_dt.size(1), train_dt.size(2))
        # train_dt = train_dt.t()
        # output = self.fc1(train_dt)
        # output = F.relu(output, inplace=True)
        #
        # output = output.t()
        # output = output.view(1, output.size(0), 1, -1)
        # output = self.conv2(output)
        # output = F.relu(output, inplace=True)
        # output = self.conv3(output)
        # out = output.view(1, -1)

        output = self.fc2(output)
        output = F.relu(output, inplace=True)
        out = self.fc3(output)
        out = out.t()
        # 0521的一个线性层，三个卷积层网络
        # # train_dt = train_dt.view(train_dt.size(1), train_dt.size(0), -1)
        # # output = self.fc1(train_dt)
        # # output = F.relu(output, inplace=True)
        # # output = output.view(1, output.size(0), output.size(1), -1)
        #
        # output = train_dt.view(1, train_dt.size(1), train_dt.size(0), -1)
        # output = self.conv1(output)
        # output = F.relu(output, inplace=True)
        #
        # # output = output.view(output.size(1), output.size(0), -1)
        # # output = self.fc1(output)
        # # output = F.relu(output, inplace=True)
        # # output = output.view(1, output.size(0), output.size(1), -1)
        #
        # output = self.conv2(output)
        # output = F.relu(output, inplace=True)
        #
        # # output = output.view(output.size(1), output.size(0), -1)
        # # output = self.fc1(output)
        # # output = F.relu(output, inplace=True)
        # # output = output.view(1, output.size(0), output.size(1), -1)
        #
        # output = self.conv3(output)
        # output = output.view(1, -1)
        #
        # output = F.relu(output, inplace=True)
        # out = self.fc1(output)

        # 0518的三个卷积层网络
        # train_dt = train_dt.view(1, train_dt.size(1), train_dt.size(0), -1)
        # output = self.conv1(train_dt)
        # output = F.relu(output, inplace=True)
        # output = self.conv2(output)
        # output = F.relu(output, inplace=True)
        # output = self.conv3(output)
        # out = output.view(1, -1)
        # 0518的线性层加卷积层网络
        # train_dt = train_dt.view(train_dt.size(1), -1)
        # output = self.fc1(train_dt)
        # output = F.relu(output, inplace=True)
        # output = output.view(1, output.size(0), output.size(1), -1)
        # output = self.conv1(output)
        # # output = F.relu(output, inplace=True)
        # out = output.view(output.size(1), -1)
        # # out = self.fc1(output)
        # 0516的LSTM层加卷积层加线性层网络
        # train_dt = train_dt.view(train_dt.size(1), train_dt.size(0), -1)
        # output, y = self.lstm(train_dt)
        # output = F.relu(output, inplace=True)
        # output = output.view(1, output.size(0), output.size(1), -1)
        # output = self.conv1(output)
        # output = F.relu(output, inplace=True)
        # output = output.view(output.size(1), -1)
        # out = self.fc1(output)
        # 0515前的两个卷积层网络
        # train_dt = train_dt.view(1, train_dt.size(1), train_dt.size(0), -1)
        # output = self.conv1(train_dt)
        # output = F.relu(output, inplace=True)
        # output = self.conv2(output)
        # out = output.view(1, -1)
        return out


class ConvNet1_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.LSTM = nn.LSTM(104, 104, 2)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,180]  输出[x,60]
        self.LSTM2 = nn.LSTM(60, 60, 2)  # 线性拟合   输入[x,3485]  输出[x,60]
        self.fc1 = nn.Linear(60, 60)  # 线性拟合，输入[x,60]  输出[x,60]

    def forward(self, seismic, well_seismic, well):  # 输入x为[batch_size， 350]

        seismic_first_size = seismic.size(0)

        output1 = seismic.mm(well_seismic.t())   # 输入[1, 60],[104, 60]  输出[1, 104]
        output1 = output1.view(1, seismic_first_size, -1)
        output1, x2 = self.LSTM(output1)

        output1 = output1.view(seismic_first_size, -1)
        output2 = output1.mm(well)
        output2 = output2.view(1, seismic_first_size, -1)
        output2, x3 = self.LSTM2(output2)

        output2 = output2.view(seismic_first_size, -1)
        out = self.fc1(output2)
        return out


class ConvNet1_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 2, 3)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.LSTM = nn.LSTM(60, 60, 2)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,180]  输出[x,60]
        self.fc1 = nn.Linear(30, 60)  # 线性拟合，输入[x,60]  输出[x,60]
        self.fc2 = nn.Linear(60, 60)  # 线性拟合，输入[x,60]  输出[x,60]
        self.fc3 = nn.Linear(2, 2)  # 线性拟合，输入[x,60]  输出[x,60]
        self.fc4 = nn.Linear(1, 1)  # 线性拟合，输入[x,60]  输出[x,60]

    def forward(self, train_dt, tempval):  # 输入x为[batch_size， 350]

        train_dt = train_dt.view(1, train_dt.size(1), 5, -1)
        output = self.conv1(train_dt)
        output = F.relu(output, inplace=True)
        output = output.view(output.size(0), -1)

        output = output.t()
        # output = self.fc3(output)
        # output = F.relu(output, inplace=True)
        # output = output.t()
        output = self.fc4(output)

        out = output.t()
        # train_dt = train_dt.view(train_dt.size(1), -1)
        # output = tempval.mm(train_dt)
        #
        # output = output.view(1, output.size(0), -1)
        # output, x2 = self.LSTM(output)
        # output = F.relu(output, inplace=True)
        # output = output.view(output.size(0), -1)
        #
        # out = self.fc1(output)
        # output = F.relu(output, inplace=True)
        # out = self.fc2(output)

        # seismic_first_size = seismic.size(0)

        # output1 = seismic.mm(well_seismic.t())   # 输入[1, 60],[104, 60]  输出[1, 104]
        # output1 = output1.mm(well)
        # output1 = output1.view(1, seismic_first_size, -1)
        # output1, x2 = self.LSTM(output1)
        # out = output1.view(seismic_first_size, -1)
        # output1 = F.relu(out, inplace=True)
        # out = self.fc4(out)
        # out = output1.view(seismic_first_size, -1)
        # train_dt = train_dt.view(1, train_dt.size(0), train_dt.size(1), -1)
        # out1 = self.conv1(train_dt)
        # out1 = F.relu(out1, inplace=True)
        # out1 = out1.view(out1.size(0), -1)
        # out1 = self.fc1(out1)
        # out1 = F.relu(out1, inplace=True)
        #
        # out2 = self.fc2(tempval)
        # out2 = F.relu(out2, inplace=True)
        #
        # out = torch.cat((out1, out2), 0)
        # out = out.view(1, train_dt.size(0), out.size(0), -1)
        # out = self.conv2(out)
        # out = F.relu(out, inplace=True)
        # out = out.view(1, out.size(1), -1)
        # out, x2 = self.LSTM(out)
        # out = out.view(out.size(1), -1)
        # out = F.relu(out, inplace=True)
        # out = self.fc3(out)

        return out


class ConvNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 20)  # 10, 24x24# （输入channels,输出channels,kernel_size 5*5）
        self.conv2 = nn.Conv2d(1, 1, 20)  # 128, 10x10 # （输入channels,输出channels,kernel_size 3*3）
        self.lstm = nn.LSTM(60, 60, 1)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果
        self.fc1 = nn.Linear(60, 60)  # 线性拟合，接收数据的维度为500，输出数据的维度为350

    def forward(self, x):  # 输入x为[batch_size， 350]
        first_size = x.size(0)
        # second_size = x.size(1)
        # x = x.view(first_size, 1, second_size, -1)  # x[5, 1, 14, 25]
        #
        # out = self.conv1(x)  # out[5, 10, 10, 21]
        # out = F.relu(out)  # out[5, 10, 10, 21]
        #
        # out = self.conv2(out)  # out[5, 20, 8, 19]
        # out = F.relu(out)  # out[5, 20, 8, 19]
        #
        # first_size1 = x.size(0)
        # out = out.view(first_size1, -1)  # out[5, 20*8*19=3040]

        # out = out.view(1, -1, out.shape[1])  # out[1 , 5, 20*8*19=3040]
        x = x.view(first_size, 1, -1)  # x[5, 1, 14, 25]
        x1, x2 = self.lstm(x)  # x1[1 , 5, 1000]

        a, b, c = x1.shape
        x1 = x1.view(-1, c)  # x1[5, 1000]
        out = self.fc1(x1)  # out[5, 350]
        return out


class ConvNet2_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 20)  # 10, 24x24# （输入channels,输出channels,kernel_size 5*5）
        self.conv2 = nn.Conv2d(1, 1, 20)  # 128, 10x10 # （输入channels,输出channels,kernel_size 3*3）
        self.lstm = nn.LSTM(60, 60, 1)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果
        self.fc1 = nn.Linear(60, 60)  # 线性拟合，接收数据的维度为500，输出数据的维度为350
        self.fc2 = nn.Linear(60, 60)  # 线性拟合，接收数据的维度为500，输出数据的维度为350
        self.fc3 = nn.Linear(60, 60)  # 线性拟合，接收数据的维度为500，输出数据的维度为350

    def forward(self, x):  # 输入x为[batch_size， 350]
        # first_size = x.size(0)
        # second_size = x.size(1)
        # x = x.view(first_size, 1, second_size, -1)  # x[5, 1, 14, 25]
        #
        # out = self.conv1(x)  # out[5, 10, 10, 21]
        # out = F.relu(out)  # out[5, 10, 10, 21]
        #
        # out = self.conv2(out)  # out[5, 20, 8, 19]
        # out = F.relu(out)  # out[5, 20, 8, 19]
        #
        # first_size1 = x.size(0)
        # out = out.view(first_size1, -1)  # out[5, 20*8*19=3040]

        # out = out.view(1, -1, out.shape[1])  # out[1 , 5, 20*8*19=3040]
        # x = x.view(first_size, 1, -1)  # x[5, 1, 14, 25]
        # x1, x2 = self.lstm(x)  # x1[1 , 5, 1000]

        # a, b, c = x1.shape
        # x1 = x1.view(-1, c)  # x1[5, 1000]
        out = self.fc1(x)
        out = F.relu(out)  # out[5, 20, 8, 19]
        out = self.fc2(out)  # out[5, 350]
        # out = F.relu(out)  # out[5, 20, 8, 19]
        # out = self.fc3(out)  # out[5, 350]
        return out


class ConvNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(350, 500)  # （输入channels,输出channels,kernel_size 5*5）
        self.fc2 = nn.Linear(500, 350)  # （输入channels,输出channels,kernel_size 5*5）

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out, inplace=True)
        out = self.fc2(out)
        return out


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()  # 面向对象中的继承
        self.lstm = nn.LSTM(350, 500, 2)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果
        self.fc1 = nn.Linear(500, 350)  # 线性拟合，接收数据的维度为500，输出数据的维度为350

    def forward(self, x):
        x = x.view(1, -1, x.shape[1])
        x1, x2 = self.lstm(x)
        a, b, c = x1.shape
        out = self.fc1(x1.view(-1, c))  # 因为线性层输入的是个二维数据，所以此处应该将lstm输出的三维数据x1调整成二维数据，最后的特征维度不能变
        return out


class LSTM_1(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv11 = nn.Conv2d(2, 1, 3)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv12 = nn.Conv2d(2, 1, 3)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv13 = nn.Conv2d(2, 1, 3)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv14 = nn.Conv2d(2, 1, 3)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv15 = nn.Conv2d(2, 1, 3)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]

        self.conv21 = nn.Conv2d(5, 5, 3)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv22 = nn.Conv2d(5, 1, 3)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]

        self.conv30 = nn.Conv2d(10, 5, 3)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv31 = nn.Conv2d(2, 1, 3)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv32 = nn.Conv2d(2, 1, 3)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv33 = nn.Conv2d(2, 1, 3)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv34 = nn.Conv2d(2, 1, 3)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv35 = nn.Conv2d(2, 1, 3)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]

        self.lstm40 = nn.LSTM(60, 60, 2)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]
        self.lstm41 = nn.LSTM(12, 12, 2)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]
        self.lstm42 = nn.LSTM(12, 12, 2)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]
        self.lstm43 = nn.LSTM(12, 12, 2)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]
        self.lstm44 = nn.LSTM(12, 12, 2)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]
        self.lstm45 = nn.LSTM(12, 12, 2)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]

        self.lstm5 = nn.LSTM(120, 60, 2)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]

        self.fc6 = nn.Linear(60, 60)  # 线性拟合，输入[x,60]  输出[x,60]
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, train_well, seismic):

        # 0702的设计一个复杂的网络
        # 第一层 卷积层 提取地震数据与井数据的共同特征
        seismic_in = seismic.view(1, 1, -1)
        layer_in_1_1 = train_well[:, 0, :]
        layer_in_1_1 = torch.cat([layer_in_1_1.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_1 = layer_in_1_1.view(layer_in_1_1.size(0), layer_in_1_1.size(1), -1, 10)
        output_layer1_1 = F.relu(self.conv11(layer_in_1_1))
        output_layer1_1 = self.dropout(output_layer1_1)
        output_layer1 = output_layer1_1

        layer_in_1_2 = train_well[:, 1, :]
        layer_in_1_2 = torch.cat([layer_in_1_2.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_2 = layer_in_1_2.view(layer_in_1_2.size(0), layer_in_1_2.size(1), -1, 10)
        output_layer1_2 = F.relu(self.conv12(layer_in_1_2))
        output_layer1_2 = self.dropout(output_layer1_2)
        output_layer1 = torch.cat([output_layer1, output_layer1_2], dim=1)

        layer_in_1_3 = train_well[:, 2, :]
        layer_in_1_3 = torch.cat([layer_in_1_3.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_3 = layer_in_1_3.view(layer_in_1_3.size(0), layer_in_1_3.size(1), -1, 10)
        output_layer1_3 = F.relu(self.conv13(layer_in_1_3))
        output_layer1_3 = self.dropout(output_layer1_3)
        output_layer1 = torch.cat([output_layer1, output_layer1_3], dim=1)

        layer_in_1_4 = train_well[:, 3, :]
        layer_in_1_4 = torch.cat([layer_in_1_4.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_4 = layer_in_1_4.view(layer_in_1_4.size(0), layer_in_1_4.size(1), -1, 10)
        output_layer1_4 = F.relu(self.conv14(layer_in_1_4))
        output_layer1_4 = self.dropout(output_layer1_4)
        output_layer1 = torch.cat([output_layer1, output_layer1_4], dim=1)

        layer_in_1_5 = train_well[:, 4, :]
        layer_in_1_5 = torch.cat([layer_in_1_5.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_5 = layer_in_1_5.view(layer_in_1_5.size(0), layer_in_1_5.size(1), -1, 10)
        output_layer1_5 = F.relu(self.conv15(layer_in_1_5))
        output_layer1_5 = self.dropout(output_layer1_5)
        output_layer1 = torch.cat([output_layer1, output_layer1_5], dim=1)

        # 第二层 卷积层 提取井数据的特征
        layer_in_2_1 = train_well.view(train_well.size(0), train_well.size(1), -1, 10)
        output_layer2_1 = F.relu(self.conv21(layer_in_2_1))
        output_layer2_1 = self.dropout(output_layer2_1)

        layer_in_2_2 = train_well.view(train_well.size(0), train_well.size(1), -1, 10)
        output_layer2_2 = F.relu(self.conv22(layer_in_2_2))
        output_layer2_2 = self.dropout(output_layer2_2)

        # 第三层 卷积层 提取井数据的特征
        layer_in_3_0 = torch.cat([output_layer1, output_layer2_1], dim=1)
        output_layer3_0 = F.relu(self.conv30(layer_in_3_0))
        output_layer3_0 = self.dropout(output_layer3_0)

        layer_in_3_1 = torch.cat([output_layer1_1, output_layer2_2], dim=1)
        output_layer3_1 = F.relu(self.conv31(layer_in_3_1))
        output_layer3_1 = self.dropout(output_layer3_1)

        layer_in_3_2 = torch.cat([output_layer1_2, output_layer2_2], dim=1)
        output_layer3_2 = F.relu(self.conv32(layer_in_3_2))
        output_layer3_2 = self.dropout(output_layer3_2)

        layer_in_3_3 = torch.cat([output_layer1_3, output_layer2_2], dim=1)
        output_layer3_3 = F.relu(self.conv33(layer_in_3_3))
        output_layer3_3 = self.dropout(output_layer3_3)

        layer_in_3_4 = torch.cat([output_layer1_4, output_layer2_2], dim=1)
        output_layer3_4 = F.relu(self.conv34(layer_in_3_4))
        output_layer3_4 = self.dropout(output_layer3_4)

        layer_in_3_5 = torch.cat([output_layer1_5, output_layer2_2], dim=1)
        output_layer3_5 = F.relu(self.conv35(layer_in_3_5))
        output_layer3_5 = self.dropout(output_layer3_5)

        # 第四层 LSTM 提取井数据的特征
        layer_in_4_0 = output_layer3_0.view(1, output_layer3_0.size(0), -1)
        output_layer4_0, temp4_0 = self.lstm40(layer_in_4_0)
        output_layer4_0 = self.dropout(output_layer4_0)

        layer_in_4_1 = output_layer3_1.view(1, output_layer3_1.size(0), -1)
        output_layer4_1, temp4_1 = self.lstm41(layer_in_4_1)
        output_layer4_1 = self.dropout(output_layer4_1)
        output_layer4 = output_layer4_1

        layer_in_4_2 = output_layer3_2.view(1, output_layer3_2.size(0), -1)
        output_layer4_2, temp4_2 = self.lstm42(layer_in_4_2)
        output_layer4_2 = self.dropout(output_layer4_2)
        output_layer4 = torch.cat([output_layer4, output_layer4_2], dim=2)

        layer_in_4_3 = output_layer3_3.view(1, output_layer3_3.size(0), -1)
        output_layer4_3, temp4_3 = self.lstm43(layer_in_4_3)
        output_layer4_3 = self.dropout(output_layer4_3)
        output_layer4 = torch.cat([output_layer4, output_layer4_3], dim=2)

        layer_in_4_4 = output_layer3_4.view(1, output_layer3_4.size(0), -1)
        output_layer4_4, temp4_4 = self.lstm44(layer_in_4_4)
        output_layer4_4 = self.dropout(output_layer4_4)
        output_layer4 = torch.cat([output_layer4, output_layer4_4], dim=2)

        layer_in_4_5 = output_layer3_5.view(1, output_layer3_5.size(0), -1)
        output_layer4_5, temp4_5 = self.lstm42(layer_in_4_5)
        output_layer4_5 = self.dropout(output_layer4_5)
        output_layer4 = torch.cat([output_layer4, output_layer4_5], dim=2)

        # 第五层 LSTM 提取井数据的特征
        layer_in_5 = torch.cat([output_layer4_0, output_layer4], dim=2)
        output_layer5, temp5 = self.lstm5(layer_in_5)
        output_layer5 = self.dropout(output_layer5)

        # 第六层 线性层 提取井数据的特征
        layer_in_6 = output_layer5.view(1, -1)
        output6 = self.fc6(layer_in_6)
        output = output6

        return output


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


class PentaConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        '''
        双重卷积
        :param in_ch: 输入通道
        :param out_ch: 输出通道
        '''
        super(PentaConv, self).__init__()
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
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Deep_Net_1(nn.Module):
    def __init__(self):
        super(Deep_Net_1, self).__init__()

        self.conv11 = TripleConv(2, 1)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv12 = TripleConv(2, 1)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv13 = TripleConv(2, 1)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv14 = TripleConv(2, 1)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv15 = TripleConv(2, 1)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]

        self.conv21 = TripleConv(5, 5)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv22 = TripleConv(5, 1)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]

        self.conv30 = TripleConv(10, 2)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv31 = TripleConv(2, 1)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv32 = TripleConv(2, 1)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv33 = TripleConv(2, 1)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv34 = TripleConv(2, 1)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]
        self.conv35 = TripleConv(2, 1)  # 10, 20x20# （输入channels,输出channels,kernel_size 20*20）输入[x,y,z,w]  输出[x,y,z-19,w-19]

        self.lstm40 = nn.LSTM(120, 60, 2)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]
        self.lstm41 = nn.LSTM(60, 12, 2)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]
        self.lstm42 = nn.LSTM(60, 12, 2)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]
        self.lstm43 = nn.LSTM(60, 12, 2)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]
        self.lstm44 = nn.LSTM(60, 12, 2)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]
        self.lstm45 = nn.LSTM(60, 12, 2)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]

        self.lstm5 = nn.LSTM(120, 60, 2)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]

        self.fc6 = nn.Linear(60, 60)  # 线性拟合，输入[x,60]  输出[x,60]
        self.dropout = nn.Dropout(p=0.5)
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

        # 0706的设计一个深度复杂的网络
        # 第一层 卷积层 提取地震数据与井数据的共同特征
        seismic_in = seismic.view(1, 1, -1)
        layer_in_1_1 = train_well[:, 0, :]
        layer_in_1_1 = torch.cat([layer_in_1_1.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_1 = layer_in_1_1.view(layer_in_1_1.size(0), layer_in_1_1.size(1), -1, 10)
        output_layer1_1 = self.conv11(layer_in_1_1)
        output_layer1_1 = self.dropout(output_layer1_1)
        output_layer1 = output_layer1_1

        layer_in_1_2 = train_well[:, 1, :]
        layer_in_1_2 = torch.cat([layer_in_1_2.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_2 = layer_in_1_2.view(layer_in_1_2.size(0), layer_in_1_2.size(1), -1, 10)
        output_layer1_2 = self.conv12(layer_in_1_2)
        output_layer1_2 = self.dropout(output_layer1_2)
        output_layer1 = torch.cat([output_layer1, output_layer1_2], dim=1)

        layer_in_1_3 = train_well[:, 2, :]
        layer_in_1_3 = torch.cat([layer_in_1_3.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_3 = layer_in_1_3.view(layer_in_1_3.size(0), layer_in_1_3.size(1), -1, 10)
        output_layer1_3 = self.conv13(layer_in_1_3)
        output_layer1_3 = self.dropout(output_layer1_3)
        output_layer1 = torch.cat([output_layer1, output_layer1_3], dim=1)

        layer_in_1_4 = train_well[:, 3, :]
        layer_in_1_4 = torch.cat([layer_in_1_4.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_4 = layer_in_1_4.view(layer_in_1_4.size(0), layer_in_1_4.size(1), -1, 10)
        output_layer1_4 = self.conv14(layer_in_1_4)
        output_layer1_4 = self.dropout(output_layer1_4)
        output_layer1 = torch.cat([output_layer1, output_layer1_4], dim=1)

        layer_in_1_5 = train_well[:, 4, :]
        layer_in_1_5 = torch.cat([layer_in_1_5.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_5 = layer_in_1_5.view(layer_in_1_5.size(0), layer_in_1_5.size(1), -1, 10)
        output_layer1_5 = self.conv15(layer_in_1_5)
        output_layer1_5 = self.dropout(output_layer1_5)
        output_layer1 = torch.cat([output_layer1, output_layer1_5], dim=1)

        # 第二层 卷积层 提取井数据的特征
        layer_in_2_1 = train_well.view(train_well.size(0), train_well.size(1), -1, 10)
        output_layer2_1 = self.conv21(layer_in_2_1)
        output_layer2_1 = self.dropout(output_layer2_1)

        layer_in_2_2 = train_well.view(train_well.size(0), train_well.size(1), -1, 10)
        output_layer2_2 = self.conv22(layer_in_2_2)
        output_layer2_2 = self.dropout(output_layer2_2)

        # 第三层 卷积层 提取井数据的特征
        layer_in_3_0 = torch.cat([output_layer1, output_layer2_1], dim=1)
        output_layer3_0 = self.conv30(layer_in_3_0)
        output_layer3_0 = self.dropout(output_layer3_0)

        layer_in_3_1 = torch.cat([output_layer1_1, output_layer2_2], dim=1)
        output_layer3_1 = self.conv31(layer_in_3_1)
        output_layer3_1 = self.dropout(output_layer3_1)

        layer_in_3_2 = torch.cat([output_layer1_2, output_layer2_2], dim=1)
        output_layer3_2 = self.conv32(layer_in_3_2)
        output_layer3_2 = self.dropout(output_layer3_2)

        layer_in_3_3 = torch.cat([output_layer1_3, output_layer2_2], dim=1)
        output_layer3_3 = self.conv33(layer_in_3_3)
        output_layer3_3 = self.dropout(output_layer3_3)

        layer_in_3_4 = torch.cat([output_layer1_4, output_layer2_2], dim=1)
        output_layer3_4 = self.conv34(layer_in_3_4)
        output_layer3_4 = self.dropout(output_layer3_4)

        layer_in_3_5 = torch.cat([output_layer1_5, output_layer2_2], dim=1)
        output_layer3_5 = self.conv35(layer_in_3_5)
        output_layer3_5 = self.dropout(output_layer3_5)

        # 第四层 LSTM 提取井数据的特征
        layer_in_4_0 = output_layer3_0.view(1, output_layer3_0.size(0), -1)
        output_layer4_0, temp4_0 = self.lstm40(layer_in_4_0)
        output_layer4_0 = self.dropout(output_layer4_0)

        layer_in_4_1 = output_layer3_1.view(1, output_layer3_1.size(0), -1)
        output_layer4_1, temp4_1 = self.lstm41(layer_in_4_1)
        output_layer4_1 = self.dropout(output_layer4_1)
        output_layer4 = output_layer4_1

        layer_in_4_2 = output_layer3_2.view(1, output_layer3_2.size(0), -1)
        output_layer4_2, temp4_2 = self.lstm42(layer_in_4_2)
        output_layer4_2 = self.dropout(output_layer4_2)
        output_layer4 = torch.cat([output_layer4, output_layer4_2], dim=2)

        layer_in_4_3 = output_layer3_3.view(1, output_layer3_3.size(0), -1)
        output_layer4_3, temp4_3 = self.lstm43(layer_in_4_3)
        output_layer4_3 = self.dropout(output_layer4_3)
        output_layer4 = torch.cat([output_layer4, output_layer4_3], dim=2)

        layer_in_4_4 = output_layer3_4.view(1, output_layer3_4.size(0), -1)
        output_layer4_4, temp4_4 = self.lstm44(layer_in_4_4)
        output_layer4_4 = self.dropout(output_layer4_4)
        output_layer4 = torch.cat([output_layer4, output_layer4_4], dim=2)

        layer_in_4_5 = output_layer3_5.view(1, output_layer3_5.size(0), -1)
        output_layer4_5, temp4_5 = self.lstm42(layer_in_4_5)
        output_layer4_5 = self.dropout(output_layer4_5)
        output_layer4 = torch.cat([output_layer4, output_layer4_5], dim=2)

        # 第五层 LSTM 提取井数据的特征
        layer_in_5 = torch.cat([output_layer4_0, output_layer4], dim=2)
        output_layer5, temp5 = self.lstm5(layer_in_5)
        output_layer5 = self.dropout(output_layer5)

        # 第六层 线性层 提取井数据的特征
        layer_in_6 = output_layer5.view(1, -1)
        output6 = self.fc6(layer_in_6)
        output = output6

        return output


class Deep_Net_2(nn.Module):
    def __init__(self):
        super(Deep_Net_2, self).__init__()

        self.conv11 = QuadrupleConv(2, 1)
        self.conv12 = QuadrupleConv(2, 1)
        self.conv13 = QuadrupleConv(2, 1)
        self.conv14 = QuadrupleConv(2, 1)
        self.conv15 = QuadrupleConv(2, 1)

        self.conv21 = QuadrupleConv(5, 5)
        self.conv22 = QuadrupleConv(5, 1)

        self.conv30 = QuadrupleConv(10, 2)
        self.conv31 = QuadrupleConv(2, 1)
        self.conv32 = QuadrupleConv(2, 1)
        self.conv33 = QuadrupleConv(2, 1)
        self.conv34 = QuadrupleConv(2, 1)
        self.conv35 = QuadrupleConv(2, 1)

        self.lstm40 = nn.LSTM(120, 60, 3)
        self.lstm41 = nn.LSTM(60, 12, 3)
        self.lstm42 = nn.LSTM(60, 12, 3)
        self.lstm43 = nn.LSTM(60, 12, 3)
        self.lstm44 = nn.LSTM(60, 12, 3)
        self.lstm45 = nn.LSTM(60, 12, 3)

        self.lstm50 = nn.LSTM(120, 60, 3)
        self.lstm51 = nn.LSTM(72, 12, 3)
        self.lstm52 = nn.LSTM(72, 12, 3)
        self.lstm53 = nn.LSTM(72, 12, 3)
        self.lstm54 = nn.LSTM(72, 12, 3)
        self.lstm55 = nn.LSTM(72, 12, 3)

        self.fc60 = nn.Linear(60, 60)
        self.fc61 = nn.Linear(12, 12)
        self.fc62 = nn.Linear(12, 12)
        self.fc63 = nn.Linear(12, 12)
        self.fc64 = nn.Linear(12, 12)
        self.fc65 = nn.Linear(12, 12)

        self.lstm7 = nn.LSTM(120, 60, 3)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]

        self.fc8 = nn.Linear(60, 60)  # 线性拟合，输入[x,60]  输出[x,60]
        self.dropout = nn.Dropout(p=0.5)
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

        # 0706的设计一个深度复杂的网络
        # 第一层 卷积层 提取地震数据与井数据的共同特征
        seismic_in = seismic.view(1, 1, -1)
        layer_in_1_1 = train_well[:, 0, :]
        layer_in_1_1 = torch.cat([layer_in_1_1.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_1 = layer_in_1_1.view(layer_in_1_1.size(0), layer_in_1_1.size(1), -1, 10)
        output_layer1_1 = self.conv11(layer_in_1_1)
        output_layer1_1 = self.dropout(output_layer1_1)
        output_layer1 = output_layer1_1

        layer_in_1_2 = train_well[:, 1, :]
        layer_in_1_2 = torch.cat([layer_in_1_2.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_2 = layer_in_1_2.view(layer_in_1_2.size(0), layer_in_1_2.size(1), -1, 10)
        output_layer1_2 = self.conv12(layer_in_1_2)
        output_layer1_2 = self.dropout(output_layer1_2)
        output_layer1 = torch.cat([output_layer1, output_layer1_2], dim=1)

        layer_in_1_3 = train_well[:, 2, :]
        layer_in_1_3 = torch.cat([layer_in_1_3.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_3 = layer_in_1_3.view(layer_in_1_3.size(0), layer_in_1_3.size(1), -1, 10)
        output_layer1_3 = self.conv13(layer_in_1_3)
        output_layer1_3 = self.dropout(output_layer1_3)
        output_layer1 = torch.cat([output_layer1, output_layer1_3], dim=1)

        layer_in_1_4 = train_well[:, 3, :]
        layer_in_1_4 = torch.cat([layer_in_1_4.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_4 = layer_in_1_4.view(layer_in_1_4.size(0), layer_in_1_4.size(1), -1, 10)
        output_layer1_4 = self.conv14(layer_in_1_4)
        output_layer1_4 = self.dropout(output_layer1_4)
        output_layer1 = torch.cat([output_layer1, output_layer1_4], dim=1)

        layer_in_1_5 = train_well[:, 4, :]
        layer_in_1_5 = torch.cat([layer_in_1_5.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_5 = layer_in_1_5.view(layer_in_1_5.size(0), layer_in_1_5.size(1), -1, 10)
        output_layer1_5 = self.conv15(layer_in_1_5)
        output_layer1_5 = self.dropout(output_layer1_5)
        output_layer1 = torch.cat([output_layer1, output_layer1_5], dim=1)

        # 第二层 卷积层 提取井数据的特征
        layer_in_2_1 = train_well.view(train_well.size(0), train_well.size(1), -1, 10)
        output_layer2_1 = self.conv21(layer_in_2_1)
        output_layer2_1 = self.dropout(output_layer2_1)

        layer_in_2_2 = train_well.view(train_well.size(0), train_well.size(1), -1, 10)
        output_layer2_2 = self.conv22(layer_in_2_2)
        output_layer2_2 = self.dropout(output_layer2_2)

        # 第三层 卷积层 提取井数据的特征
        layer_in_3_0 = torch.cat([output_layer1, output_layer2_1], dim=1)
        output_layer3_0 = self.conv30(layer_in_3_0)
        output_layer3_0 = self.dropout(output_layer3_0)

        layer_in_3_1 = torch.cat([output_layer1_1, output_layer2_2], dim=1)
        output_layer3_1 = self.conv31(layer_in_3_1)
        output_layer3_1 = self.dropout(output_layer3_1)

        layer_in_3_2 = torch.cat([output_layer1_2, output_layer2_2], dim=1)
        output_layer3_2 = self.conv32(layer_in_3_2)
        output_layer3_2 = self.dropout(output_layer3_2)

        layer_in_3_3 = torch.cat([output_layer1_3, output_layer2_2], dim=1)
        output_layer3_3 = self.conv33(layer_in_3_3)
        output_layer3_3 = self.dropout(output_layer3_3)

        layer_in_3_4 = torch.cat([output_layer1_4, output_layer2_2], dim=1)
        output_layer3_4 = self.conv34(layer_in_3_4)
        output_layer3_4 = self.dropout(output_layer3_4)

        layer_in_3_5 = torch.cat([output_layer1_5, output_layer2_2], dim=1)
        output_layer3_5 = self.conv35(layer_in_3_5)
        output_layer3_5 = self.dropout(output_layer3_5)

        # 第四层 LSTM 提取井数据的特征
        layer_in_4_0 = output_layer3_0.view(1, output_layer3_0.size(0), -1)
        output_layer4_0, temp4_0 = self.lstm40(layer_in_4_0)
        output_layer4_0 = self.dropout(output_layer4_0)

        layer_in_4_1 = output_layer3_1.view(1, output_layer3_1.size(0), -1)
        output_layer4_1, temp4_1 = self.lstm41(layer_in_4_1)
        output_layer4_1 = self.dropout(output_layer4_1)
        output_layer4 = output_layer4_1

        layer_in_4_2 = output_layer3_2.view(1, output_layer3_2.size(0), -1)
        output_layer4_2, temp4_2 = self.lstm42(layer_in_4_2)
        output_layer4_2 = self.dropout(output_layer4_2)
        output_layer4 = torch.cat([output_layer4, output_layer4_2], dim=2)

        layer_in_4_3 = output_layer3_3.view(1, output_layer3_3.size(0), -1)
        output_layer4_3, temp4_3 = self.lstm43(layer_in_4_3)
        output_layer4_3 = self.dropout(output_layer4_3)
        output_layer4 = torch.cat([output_layer4, output_layer4_3], dim=2)

        layer_in_4_4 = output_layer3_4.view(1, output_layer3_4.size(0), -1)
        output_layer4_4, temp4_4 = self.lstm44(layer_in_4_4)
        output_layer4_4 = self.dropout(output_layer4_4)
        output_layer4 = torch.cat([output_layer4, output_layer4_4], dim=2)

        layer_in_4_5 = output_layer3_5.view(1, output_layer3_5.size(0), -1)
        output_layer4_5, temp4_5 = self.lstm42(layer_in_4_5)
        output_layer4_5 = self.dropout(output_layer4_5)
        output_layer4 = torch.cat([output_layer4, output_layer4_5], dim=2)

        # 第五层 LSTM 特征映射
        layer_in_5_0 = torch.cat([output_layer4_0, output_layer4], dim=2)
        output_layer5_0, temp5_0 = self.lstm50(layer_in_5_0)
        output_layer5_0 = self.dropout(output_layer5_0)

        layer_in_5_1 = torch.cat([output_layer4_0, output_layer4_1], dim=2)
        output_layer5_1, temp5_1 = self.lstm51(layer_in_5_1)
        output_layer5_1 = self.dropout(output_layer5_1)

        layer_in_5_2 = torch.cat([output_layer4_0, output_layer4_2], dim=2)
        output_layer5_2, temp5_2 = self.lstm52(layer_in_5_2)
        output_layer5_2 = self.dropout(output_layer5_2)

        layer_in_5_3 = torch.cat([output_layer4_0, output_layer4_3], dim=2)
        output_layer5_3, temp5_3 = self.lstm53(layer_in_5_3)
        output_layer5_3 = self.dropout(output_layer5_3)

        layer_in_5_4 = torch.cat([output_layer4_0, output_layer4_4], dim=2)
        output_layer5_4, temp5_4 = self.lstm54(layer_in_5_4)
        output_layer5_4 = self.dropout(output_layer5_4)

        layer_in_5_5 = torch.cat([output_layer4_0, output_layer4_5], dim=2)
        output_layer5_5, temp5_5 = self.lstm55(layer_in_5_5)
        output_layer5_5 = self.dropout(output_layer5_5)

        # 第六层 线性层 特征映射
        output_layer6_0 = output_layer5_0.view(1, -1)
        output_layer6_0 = self.fc60(output_layer6_0)

        output_layer6_1 = output_layer5_1.view(1, -1)
        output_layer6_1 = self.fc61(output_layer6_1)
        output_layer6 = output_layer6_1

        output_layer6_2 = output_layer5_2.view(1, -1)
        output_layer6_2 = self.fc62(output_layer6_2)
        output_layer6 = torch.cat([output_layer6, output_layer6_2], dim=1)

        output_layer6_3 = output_layer5_3.view(1, -1)
        output_layer6_3 = self.fc63(output_layer6_3)
        output_layer6 = torch.cat([output_layer6, output_layer6_3], dim=1)

        output_layer6_4 = output_layer5_4.view(1, -1)
        output_layer6_4 = self.fc64(output_layer6_4)
        output_layer6 = torch.cat([output_layer6, output_layer6_4], dim=1)

        output_layer6_5 = output_layer5_5.view(1, -1)
        output_layer6_5 = self.fc65(output_layer6_5)
        output_layer6 = torch.cat([output_layer6, output_layer6_5], dim=1)

        # 第七层 LSTM 特征映射
        layer_in_7 = torch.cat([output_layer6_0, output_layer6], dim=1)
        layer_in_7 = layer_in_7.view(1, 1, -1)
        output_layer7, temp7 = self.lstm7(layer_in_7)
        output_layer7 = self.dropout(output_layer7)

        # 第八层 线性层 特征映射
        layer_in_8 = output_layer7.view(1, -1)
        output8 = self.fc8(layer_in_8)
        output = output8

        return output


class Deep_Net_2_M1(nn.Module):
    def __init__(self):
        super(Deep_Net_2_M1, self).__init__()

        self.conv11 = TripleConv(2, 1)
        self.conv12 = QuadrupleConv(2, 1)
        self.conv13 = TripleConv(2, 1)
        self.conv14 = QuadrupleConv(2, 1)
        self.conv15 = TripleConv(2, 1)

        self.conv21 = QuadrupleConv(5, 5)
        self.conv22 = TripleConv(5, 1)

        self.conv30 = QuadrupleConv(10, 2)
        self.conv31 = TripleConv(2, 1)
        self.conv32 = QuadrupleConv(2, 1)
        self.conv33 = TripleConv(2, 1)
        self.conv34 = QuadrupleConv(2, 1)
        self.conv35 = TripleConv(2, 1)

        self.lstm40 = nn.LSTM(120, 60, 3)
        self.lstm41 = nn.LSTM(60, 12, 3)
        self.lstm42 = nn.LSTM(60, 12, 3)
        self.lstm43 = nn.LSTM(60, 12, 3)
        self.lstm44 = nn.LSTM(60, 12, 3)
        self.lstm45 = nn.LSTM(60, 12, 3)

        self.lstm50 = nn.LSTM(120, 60, 3)
        self.lstm51 = nn.LSTM(72, 12, 3)
        self.lstm52 = nn.LSTM(72, 12, 3)
        self.lstm53 = nn.LSTM(72, 12, 3)
        self.lstm54 = nn.LSTM(72, 12, 3)
        self.lstm55 = nn.LSTM(72, 12, 3)

        self.fc60 = nn.Linear(60, 60)
        self.fc61 = nn.Linear(12, 12)
        self.fc62 = nn.Linear(12, 12)
        self.fc63 = nn.Linear(12, 12)
        self.fc64 = nn.Linear(12, 12)
        self.fc65 = nn.Linear(12, 12)

        self.lstm7 = nn.LSTM(120, 60, 3)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]

        self.fc8 = nn.Linear(60, 60)  # 线性拟合，输入[x,60]  输出[x,60]
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

        # 0706的设计一个深度复杂的网络
        # 第一层 卷积层 提取地震数据与井数据的共同特征
        seismic_in = seismic.view(1, 1, -1)
        layer_in_1_1 = train_well[:, 0, :]
        layer_in_1_1 = torch.cat([layer_in_1_1.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_1 = layer_in_1_1.view(layer_in_1_1.size(0), layer_in_1_1.size(1), -1, 10)
        output_layer1_1 = self.conv11(layer_in_1_1)
        # output_layer1_1 = self.dropout(output_layer1_1)
        output_layer1 = output_layer1_1

        layer_in_1_2 = train_well[:, 1, :]
        layer_in_1_2 = torch.cat([layer_in_1_2.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_2 = layer_in_1_2.view(layer_in_1_2.size(0), layer_in_1_2.size(1), -1, 10)
        output_layer1_2 = self.conv12(layer_in_1_2)
        # output_layer1_2 = self.dropout(output_layer1_2)
        output_layer1 = torch.cat([output_layer1, output_layer1_2], dim=1)

        layer_in_1_3 = train_well[:, 2, :]
        layer_in_1_3 = torch.cat([layer_in_1_3.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_3 = layer_in_1_3.view(layer_in_1_3.size(0), layer_in_1_3.size(1), -1, 10)
        output_layer1_3 = self.conv13(layer_in_1_3)
        # output_layer1_3 = self.dropout(output_layer1_3)
        output_layer1 = torch.cat([output_layer1, output_layer1_3], dim=1)

        layer_in_1_4 = train_well[:, 3, :]
        layer_in_1_4 = torch.cat([layer_in_1_4.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_4 = layer_in_1_4.view(layer_in_1_4.size(0), layer_in_1_4.size(1), -1, 10)
        output_layer1_4 = self.conv14(layer_in_1_4)
        # output_layer1_4 = self.dropout(output_layer1_4)
        output_layer1 = torch.cat([output_layer1, output_layer1_4], dim=1)

        layer_in_1_5 = train_well[:, 4, :]
        layer_in_1_5 = torch.cat([layer_in_1_5.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_5 = layer_in_1_5.view(layer_in_1_5.size(0), layer_in_1_5.size(1), -1, 10)
        output_layer1_5 = self.conv15(layer_in_1_5)
        # output_layer1_5 = self.dropout(output_layer1_5)
        output_layer1 = torch.cat([output_layer1, output_layer1_5], dim=1)

        # 第二层 卷积层 提取井数据的特征
        layer_in_2_1 = train_well.view(train_well.size(0), train_well.size(1), -1, 10)
        output_layer2_1 = self.conv21(layer_in_2_1)
        # output_layer2_1 = self.dropout(output_layer2_1)

        layer_in_2_2 = train_well.view(train_well.size(0), train_well.size(1), -1, 10)
        output_layer2_2 = self.conv22(layer_in_2_2)
        # output_layer2_2 = self.dropout(output_layer2_2)

        # 第三层 卷积层 提取井数据的特征
        layer_in_3_0 = torch.cat([output_layer1, output_layer2_1], dim=1)
        output_layer3_0 = self.conv30(layer_in_3_0)
        # output_layer3_0 = self.dropout(output_layer3_0)

        layer_in_3_1 = torch.cat([output_layer1_1, output_layer2_2], dim=1)
        output_layer3_1 = self.conv31(layer_in_3_1)
        # output_layer3_1 = self.dropout(output_layer3_1)

        layer_in_3_2 = torch.cat([output_layer1_2, output_layer2_2], dim=1)
        output_layer3_2 = self.conv32(layer_in_3_2)
        # output_layer3_2 = self.dropout(output_layer3_2)

        layer_in_3_3 = torch.cat([output_layer1_3, output_layer2_2], dim=1)
        output_layer3_3 = self.conv33(layer_in_3_3)
        # output_layer3_3 = self.dropout(output_layer3_3)

        layer_in_3_4 = torch.cat([output_layer1_4, output_layer2_2], dim=1)
        output_layer3_4 = self.conv34(layer_in_3_4)
        # output_layer3_4 = self.dropout(output_layer3_4)

        layer_in_3_5 = torch.cat([output_layer1_5, output_layer2_2], dim=1)
        output_layer3_5 = self.conv35(layer_in_3_5)
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

        layer_in_5_2 = torch.cat([output_layer4_0, output_layer4_2], dim=2)
        output_layer5_2, temp5_2 = self.lstm52(layer_in_5_2)

        layer_in_5_3 = torch.cat([output_layer4_0, output_layer4_3], dim=2)
        output_layer5_3, temp5_3 = self.lstm53(layer_in_5_3)

        layer_in_5_4 = torch.cat([output_layer4_0, output_layer4_4], dim=2)
        output_layer5_4, temp5_4 = self.lstm54(layer_in_5_4)

        layer_in_5_5 = torch.cat([output_layer4_0, output_layer4_5], dim=2)
        output_layer5_5, temp5_5 = self.lstm55(layer_in_5_5)

        # 第六层 线性层 特征映射
        output_layer6_0 = output_layer5_0.view(1, -1)
        output_layer6_0 = self.fc60(output_layer6_0)

        output_layer6_1 = output_layer5_1.view(1, -1)
        output_layer6_1 = self.fc61(output_layer6_1)
        output_layer6 = output_layer6_1

        output_layer6_2 = output_layer5_2.view(1, -1)
        output_layer6_2 = self.fc62(output_layer6_2)
        output_layer6 = torch.cat([output_layer6, output_layer6_2], dim=1)

        output_layer6_3 = output_layer5_3.view(1, -1)
        output_layer6_3 = self.fc63(output_layer6_3)
        output_layer6 = torch.cat([output_layer6, output_layer6_3], dim=1)

        output_layer6_4 = output_layer5_4.view(1, -1)
        output_layer6_4 = self.fc64(output_layer6_4)
        output_layer6 = torch.cat([output_layer6, output_layer6_4], dim=1)

        output_layer6_5 = output_layer5_5.view(1, -1)
        output_layer6_5 = self.fc65(output_layer6_5)
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


class Deep_Net_3(nn.Module):
    def __init__(self):
        super(Deep_Net_3, self).__init__()

        self.conv11 = PentaConv(2, 1)
        self.conv12 = PentaConv(2, 1)
        self.conv13 = PentaConv(2, 1)
        self.conv14 = PentaConv(2, 1)
        self.conv15 = PentaConv(2, 1)

        self.conv21 = PentaConv(5, 5)
        self.conv22 = PentaConv(5, 1)

        self.conv30 = PentaConv(10, 2)
        self.conv31 = PentaConv(2, 1)
        self.conv32 = PentaConv(2, 1)
        self.conv33 = PentaConv(2, 1)
        self.conv34 = PentaConv(2, 1)
        self.conv35 = PentaConv(2, 1)

        self.lstm40 = nn.LSTM(120, 60, 5)
        self.lstm41 = nn.LSTM(60, 12, 5)
        self.lstm42 = nn.LSTM(60, 12, 5)
        self.lstm43 = nn.LSTM(60, 12, 5)
        self.lstm44 = nn.LSTM(60, 12, 5)
        self.lstm45 = nn.LSTM(60, 12, 5)

        self.lstm50 = nn.LSTM(120, 60, 5)
        self.lstm51 = nn.LSTM(72, 12, 5)
        self.lstm52 = nn.LSTM(72, 12, 5)
        self.lstm53 = nn.LSTM(72, 12, 5)
        self.lstm54 = nn.LSTM(72, 12, 5)
        self.lstm55 = nn.LSTM(72, 12, 5)

        self.fc60 = nn.Linear(60, 60)
        self.fc61 = nn.Linear(12, 12)
        self.fc62 = nn.Linear(12, 12)
        self.fc63 = nn.Linear(12, 12)
        self.fc64 = nn.Linear(12, 12)
        self.fc65 = nn.Linear(12, 12)

        self.lstm7 = nn.LSTM(120, 60, 5)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]

        self.fc8 = nn.Linear(60, 60)  # 线性拟合，输入[x,60]  输出[x,60]
        self.dropout = nn.Dropout(p=0.5)
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

        # 0717的设计一个深度深度复杂的网络
        # 第一层 卷积层 提取地震数据与井数据的共同特征
        seismic_in = seismic.view(1, 1, -1)
        layer_in_1_1 = train_well[:, 0, :]
        layer_in_1_1 = torch.cat([layer_in_1_1.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_1 = layer_in_1_1.view(layer_in_1_1.size(0), layer_in_1_1.size(1), -1, 10)
        output_layer1_1 = self.conv11(layer_in_1_1)
        output_layer1_1 = self.dropout(output_layer1_1)
        output_layer1 = output_layer1_1

        layer_in_1_2 = train_well[:, 1, :]
        layer_in_1_2 = torch.cat([layer_in_1_2.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_2 = layer_in_1_2.view(layer_in_1_2.size(0), layer_in_1_2.size(1), -1, 10)
        output_layer1_2 = self.conv12(layer_in_1_2)
        output_layer1_2 = self.dropout(output_layer1_2)
        output_layer1 = torch.cat([output_layer1, output_layer1_2], dim=1)

        layer_in_1_3 = train_well[:, 2, :]
        layer_in_1_3 = torch.cat([layer_in_1_3.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_3 = layer_in_1_3.view(layer_in_1_3.size(0), layer_in_1_3.size(1), -1, 10)
        output_layer1_3 = self.conv13(layer_in_1_3)
        output_layer1_3 = self.dropout(output_layer1_3)
        output_layer1 = torch.cat([output_layer1, output_layer1_3], dim=1)

        layer_in_1_4 = train_well[:, 3, :]
        layer_in_1_4 = torch.cat([layer_in_1_4.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_4 = layer_in_1_4.view(layer_in_1_4.size(0), layer_in_1_4.size(1), -1, 10)
        output_layer1_4 = self.conv14(layer_in_1_4)
        output_layer1_4 = self.dropout(output_layer1_4)
        output_layer1 = torch.cat([output_layer1, output_layer1_4], dim=1)

        layer_in_1_5 = train_well[:, 4, :]
        layer_in_1_5 = torch.cat([layer_in_1_5.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_5 = layer_in_1_5.view(layer_in_1_5.size(0), layer_in_1_5.size(1), -1, 10)
        output_layer1_5 = self.conv15(layer_in_1_5)
        output_layer1_5 = self.dropout(output_layer1_5)
        output_layer1 = torch.cat([output_layer1, output_layer1_5], dim=1)

        # 第二层 卷积层 提取井数据的特征
        layer_in_2_1 = train_well.view(train_well.size(0), train_well.size(1), -1, 10)
        output_layer2_1 = self.conv21(layer_in_2_1)
        output_layer2_1 = self.dropout(output_layer2_1)

        layer_in_2_2 = train_well.view(train_well.size(0), train_well.size(1), -1, 10)
        output_layer2_2 = self.conv22(layer_in_2_2)
        output_layer2_2 = self.dropout(output_layer2_2)

        # 第三层 卷积层 提取井数据的特征
        layer_in_3_0 = torch.cat([output_layer1, output_layer2_1], dim=1)
        output_layer3_0 = self.conv30(layer_in_3_0)
        output_layer3_0 = self.dropout(output_layer3_0)

        layer_in_3_1 = torch.cat([output_layer1_1, output_layer2_2], dim=1)
        output_layer3_1 = self.conv31(layer_in_3_1)
        output_layer3_1 = self.dropout(output_layer3_1)

        layer_in_3_2 = torch.cat([output_layer1_2, output_layer2_2], dim=1)
        output_layer3_2 = self.conv32(layer_in_3_2)
        output_layer3_2 = self.dropout(output_layer3_2)

        layer_in_3_3 = torch.cat([output_layer1_3, output_layer2_2], dim=1)
        output_layer3_3 = self.conv33(layer_in_3_3)
        output_layer3_3 = self.dropout(output_layer3_3)

        layer_in_3_4 = torch.cat([output_layer1_4, output_layer2_2], dim=1)
        output_layer3_4 = self.conv34(layer_in_3_4)
        output_layer3_4 = self.dropout(output_layer3_4)

        layer_in_3_5 = torch.cat([output_layer1_5, output_layer2_2], dim=1)
        output_layer3_5 = self.conv35(layer_in_3_5)
        output_layer3_5 = self.dropout(output_layer3_5)

        # 第四层 LSTM 提取井数据的特征
        layer_in_4_0 = output_layer3_0.view(1, output_layer3_0.size(0), -1)
        output_layer4_0, temp4_0 = self.lstm40(layer_in_4_0)
        output_layer4_0 = self.dropout(output_layer4_0)

        layer_in_4_1 = output_layer3_1.view(1, output_layer3_1.size(0), -1)
        output_layer4_1, temp4_1 = self.lstm41(layer_in_4_1)
        output_layer4_1 = self.dropout(output_layer4_1)
        output_layer4 = output_layer4_1

        layer_in_4_2 = output_layer3_2.view(1, output_layer3_2.size(0), -1)
        output_layer4_2, temp4_2 = self.lstm42(layer_in_4_2)
        output_layer4_2 = self.dropout(output_layer4_2)
        output_layer4 = torch.cat([output_layer4, output_layer4_2], dim=2)

        layer_in_4_3 = output_layer3_3.view(1, output_layer3_3.size(0), -1)
        output_layer4_3, temp4_3 = self.lstm43(layer_in_4_3)
        output_layer4_3 = self.dropout(output_layer4_3)
        output_layer4 = torch.cat([output_layer4, output_layer4_3], dim=2)

        layer_in_4_4 = output_layer3_4.view(1, output_layer3_4.size(0), -1)
        output_layer4_4, temp4_4 = self.lstm44(layer_in_4_4)
        output_layer4_4 = self.dropout(output_layer4_4)
        output_layer4 = torch.cat([output_layer4, output_layer4_4], dim=2)

        layer_in_4_5 = output_layer3_5.view(1, output_layer3_5.size(0), -1)
        output_layer4_5, temp4_5 = self.lstm42(layer_in_4_5)
        output_layer4_5 = self.dropout(output_layer4_5)
        output_layer4 = torch.cat([output_layer4, output_layer4_5], dim=2)

        # 第五层 LSTM 特征映射
        layer_in_5_0 = torch.cat([output_layer4_0, output_layer4], dim=2)
        output_layer5_0, temp5_0 = self.lstm50(layer_in_5_0)
        output_layer5_0 = self.dropout(output_layer5_0)

        layer_in_5_1 = torch.cat([output_layer4_0, output_layer4_1], dim=2)
        output_layer5_1, temp5_1 = self.lstm51(layer_in_5_1)
        output_layer5_1 = self.dropout(output_layer5_1)

        layer_in_5_2 = torch.cat([output_layer4_0, output_layer4_2], dim=2)
        output_layer5_2, temp5_2 = self.lstm52(layer_in_5_2)
        output_layer5_2 = self.dropout(output_layer5_2)

        layer_in_5_3 = torch.cat([output_layer4_0, output_layer4_3], dim=2)
        output_layer5_3, temp5_3 = self.lstm53(layer_in_5_3)
        output_layer5_3 = self.dropout(output_layer5_3)

        layer_in_5_4 = torch.cat([output_layer4_0, output_layer4_4], dim=2)
        output_layer5_4, temp5_4 = self.lstm54(layer_in_5_4)
        output_layer5_4 = self.dropout(output_layer5_4)

        layer_in_5_5 = torch.cat([output_layer4_0, output_layer4_5], dim=2)
        output_layer5_5, temp5_5 = self.lstm55(layer_in_5_5)
        output_layer5_5 = self.dropout(output_layer5_5)

        # 第六层 线性层 特征映射
        output_layer6_0 = output_layer5_0.view(1, -1)
        output_layer6_0 = self.fc60(output_layer6_0)

        output_layer6_1 = output_layer5_1.view(1, -1)
        output_layer6_1 = self.fc61(output_layer6_1)
        output_layer6 = output_layer6_1

        output_layer6_2 = output_layer5_2.view(1, -1)
        output_layer6_2 = self.fc62(output_layer6_2)
        output_layer6 = torch.cat([output_layer6, output_layer6_2], dim=1)

        output_layer6_3 = output_layer5_3.view(1, -1)
        output_layer6_3 = self.fc63(output_layer6_3)
        output_layer6 = torch.cat([output_layer6, output_layer6_3], dim=1)

        output_layer6_4 = output_layer5_4.view(1, -1)
        output_layer6_4 = self.fc64(output_layer6_4)
        output_layer6 = torch.cat([output_layer6, output_layer6_4], dim=1)

        output_layer6_5 = output_layer5_5.view(1, -1)
        output_layer6_5 = self.fc65(output_layer6_5)
        output_layer6 = torch.cat([output_layer6, output_layer6_5], dim=1)

        # 第七层 LSTM 特征映射
        layer_in_7 = torch.cat([output_layer6_0, output_layer6], dim=1)
        layer_in_7 = layer_in_7.view(1, 1, -1)
        output_layer7, temp7 = self.lstm7(layer_in_7)
        output_layer7 = self.dropout(output_layer7)

        # 第八层 线性层 特征映射
        layer_in_8 = output_layer7.view(1, -1)
        output8 = self.fc8(layer_in_8)
        output = output8

        return output


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


class CNN_SMI(nn.Module):
    # 卷积神经网络，丢掉之前的LSTM层
    def __init__(self):
        super(CNN_SMI, self).__init__()

        self.conv11 = QuadrupleConv_U(2, 1)
        self.conv12 = TripleConv_U(2, 1)
        self.conv13 = QuadrupleConv_U(2, 1)
        self.conv14 = TripleConv_U(2, 1)
        self.conv15 = QuadrupleConv_U(2, 1)
        self.conv16 = TripleConv_U(1, 1)
        self.conv17 = QuadrupleConv_U(5, 1)
        self.conv18 = QuadrupleConv_U(6, 1)

        self.conv21 = TripleConv_U(8, 1)
        self.conv22 = QuadrupleConv_U(8, 1)
        self.conv23 = TripleConv_U(8, 1)
        self.conv24 = QuadrupleConv_U(8, 1)
        self.conv25 = TripleConv_U(8, 1)
        self.conv26 = QuadrupleConv_U(8, 1)
        self.conv27 = TripleConv_U(8, 1)
        self.conv28 = QuadrupleConv_U(8, 1)

        self.conv31 = QuadrupleConv_U(8, 1)
        self.conv32 = TripleConv_U(8, 1)
        self.conv33 = QuadrupleConv_U(8, 1)
        self.conv34 = TripleConv_U(8, 1)
        self.conv35 = QuadrupleConv_U(8, 1)
        self.conv36 = TripleConv_U(8, 1)
        self.conv37 = QuadrupleConv_U(8, 1)
        self.conv38 = TripleConv_U(8, 1)

        self.conv41 = TripleConv_U(8, 1)
        self.conv42 = QuadrupleConv_U(8, 1)
        self.conv43 = TripleConv_U(8, 1)
        self.conv44 = QuadrupleConv_U(8, 1)
        self.conv45 = TripleConv_U(8, 1)
        self.conv46 = QuadrupleConv_U(8, 1)
        self.conv47 = TripleConv_U(8, 1)
        self.conv48 = QuadrupleConv_U(8, 1)

        self.conv51 = QuadrupleConv_U(8, 1)
        self.conv52 = TripleConv_U(8, 1)
        self.conv53 = QuadrupleConv_U(8, 1)
        self.conv54 = TripleConv_U(8, 1)
        self.conv55 = QuadrupleConv_U(8, 1)
        self.conv56 = TripleConv_U(8, 1)
        self.conv57 = QuadrupleConv_U(8, 1)
        self.conv58 = TripleConv_U(8, 1)

        self.conv61 = TripleConv_U(8, 1)
        self.conv62 = QuadrupleConv_U(8, 1)
        self.conv63 = TripleConv_U(8, 1)
        self.conv64 = QuadrupleConv_U(8, 1)
        self.conv65 = TripleConv_U(8, 1)
        self.conv66 = QuadrupleConv_U(8, 1)
        self.conv67 = TripleConv_U(8, 1)
        self.conv68 = QuadrupleConv_U(8, 1)

        self.conv71 = QuadrupleConv_U(8, 1)
        self.conv72 = TripleConv_U(8, 1)
        self.conv73 = QuadrupleConv_U(8, 1)
        self.conv74 = TripleConv_U(8, 1)
        self.conv75 = QuadrupleConv_U(8, 1)
        self.conv76 = TripleConv_U(8, 1)
        self.conv77 = QuadrupleConv_U(8, 1)
        self.conv78 = TripleConv_U(8, 1)

        self.conv81 = TripleConv_U(8, 1)
        self.conv82 = QuadrupleConv_U(8, 1)
        self.conv83 = TripleConv_U(8, 1)
        self.conv84 = QuadrupleConv_U(8, 1)
        self.conv85 = TripleConv_U(8, 1)
        self.conv86 = QuadrupleConv_U(8, 1)
        self.conv87 = TripleConv_U(8, 1)
        self.conv88 = QuadrupleConv_U(8, 1)

        self.conv91 = QuadrupleConv_U(8, 1)
        self.conv92 = TripleConv_U(8, 1)
        self.conv93 = QuadrupleConv_U(8, 1)
        self.conv94 = TripleConv_U(8, 1)
        self.conv95 = QuadrupleConv_U(8, 1)
        self.conv96 = TripleConv_U(8, 1)
        self.conv97 = QuadrupleConv_U(8, 1)
        self.conv98 = TripleConv_U(8, 1)

        self.conv101 = TripleConv_U(8, 1)
        self.conv102 = QuadrupleConv_U(8, 1)
        self.conv103 = TripleConv_U(8, 1)
        self.conv104 = QuadrupleConv_U(8, 1)
        self.conv105 = TripleConv_U(8, 1)
        self.conv106 = QuadrupleConv_U(8, 1)
        self.conv107 = TripleConv_U(8, 1)
        self.conv108 = QuadrupleConv_U(8, 1)

        self.conv111 = QuadrupleConv_U(8, 1)
        self.conv112 = TripleConv_U(8, 1)
        self.conv113 = QuadrupleConv_U(8, 1)
        self.conv114 = TripleConv_U(8, 1)
        self.conv115 = QuadrupleConv_U(8, 1)
        self.conv116 = TripleConv_U(8, 1)
        self.conv117 = QuadrupleConv_U(8, 1)
        self.conv118 = TripleConv_U(8, 1)

        self.conv121 = TripleConv_U(8, 1)
        self.conv122 = QuadrupleConv_U(8, 1)
        self.conv123 = TripleConv_U(8, 1)
        self.conv124 = QuadrupleConv_U(8, 1)
        self.conv125 = TripleConv_U(8, 1)
        self.conv126 = QuadrupleConv_U(8, 1)
        self.conv127 = TripleConv_U(8, 1)
        self.conv128 = QuadrupleConv_U(8, 1)

        self.conv131 = QuadrupleConv_U(8, 1)
        self.conv132 = TripleConv_U(8, 1)
        self.conv133 = QuadrupleConv_U(8, 1)
        self.conv134 = TripleConv_U(8, 1)
        self.conv135 = QuadrupleConv_U(8, 1)
        self.conv136 = TripleConv_U(8, 1)
        self.conv137 = QuadrupleConv_U(8, 1)
        self.conv138 = TripleConv_U(8, 1)

        self.conv141 = TripleConv_U(8, 1)
        self.conv142 = QuadrupleConv_U(8, 1)
        self.conv143 = TripleConv_U(8, 1)
        self.conv144 = QuadrupleConv_U(8, 1)
        self.conv145 = TripleConv_U(8, 1)
        self.conv146 = QuadrupleConv_U(8, 1)
        self.conv147 = TripleConv_U(8, 1)
        self.conv148 = QuadrupleConv_U(8, 1)

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
        layer_in_1_1 = layer_in_1_1.view(layer_in_1_1.size(0), layer_in_1_1.size(1), -1, 10)

        layer_in_1_2 = train_well[:, 1, :]
        layer_in_1_2 = torch.cat([layer_in_1_2.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_2 = layer_in_1_2.view(layer_in_1_2.size(0), layer_in_1_2.size(1), -1, 10)

        layer_in_1_3 = train_well[:, 2, :]
        layer_in_1_3 = torch.cat([layer_in_1_3.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_3 = layer_in_1_3.view(layer_in_1_3.size(0), layer_in_1_3.size(1), -1, 10)

        layer_in_1_4 = train_well[:, 3, :]
        layer_in_1_4 = torch.cat([layer_in_1_4.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_4 = layer_in_1_4.view(layer_in_1_4.size(0), layer_in_1_4.size(1), -1, 10)

        layer_in_1_5 = train_well[:, 4, :]
        layer_in_1_5 = torch.cat([layer_in_1_5.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_5 = layer_in_1_5.view(layer_in_1_5.size(0), layer_in_1_5.size(1), -1, 10)

        layer_in_1_6 = seismic.view(1, 1, -1)
        layer_in_1_6 = layer_in_1_6.view(seismic.size(0), seismic.size(0), -1, 10)

        layer_in_1_7 = train_well.view(1, 5, -1)
        layer_in_1_7 = layer_in_1_7.view(train_well.size(0), train_well.size(1), -1, 10)

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
        output_layer2_1 = self.conv21(output_layer1)
        output_layer2 = output_layer2_1
        output_layer2_2 = self.conv22(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_2], dim=1)
        output_layer2_3 = self.conv23(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_3], dim=1)
        output_layer2_4 = self.conv24(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_4], dim=1)
        output_layer2_5 = self.conv25(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_5], dim=1)
        output_layer2_6 = self.conv26(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_6], dim=1)
        output_layer2_7 = self.conv27(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_7], dim=1)
        output_layer2_8 = self.conv28(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_8], dim=1)

        # 第三层 卷积层
        output_layer3_1 = self.conv31(output_layer2)
        output_layer3 = output_layer3_1
        output_layer3_2 = self.conv32(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_2], dim=1)
        output_layer3_3 = self.conv33(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_3], dim=1)
        output_layer3_4 = self.conv34(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_4], dim=1)
        output_layer3_5 = self.conv35(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_5], dim=1)
        output_layer3_6 = self.conv36(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_6], dim=1)
        output_layer3_7 = self.conv37(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_7], dim=1)
        output_layer3_8 = self.conv38(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_8], dim=1)

        # 第四层 卷积层
        output_layer4_1 = self.conv41(output_layer3)
        output_layer4 = output_layer4_1
        output_layer4_2 = self.conv42(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_2], dim=1)
        output_layer4_3 = self.conv43(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_3], dim=1)
        output_layer4_4 = self.conv44(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_4], dim=1)
        output_layer4_5 = self.conv45(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_5], dim=1)
        output_layer4_6 = self.conv46(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_6], dim=1)
        output_layer4_7 = self.conv47(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_7], dim=1)
        output_layer4_8 = self.conv48(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_8], dim=1)

        # 第五层 卷积层
        output_layer5_1 = self.conv51(output_layer4)
        output_layer5 = output_layer5_1
        output_layer5_2 = self.conv52(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_2], dim=1)
        output_layer5_3 = self.conv53(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_3], dim=1)
        output_layer5_4 = self.conv54(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_4], dim=1)
        output_layer5_5 = self.conv55(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_5], dim=1)
        output_layer5_6 = self.conv56(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_6], dim=1)
        output_layer5_7 = self.conv57(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_7], dim=1)
        output_layer5_8 = self.conv58(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_8], dim=1)

        # 第六层 卷积层
        output_layer6_1 = self.conv61(output_layer5)
        output_layer6 = output_layer6_1
        output_layer6_2 = self.conv62(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_2], dim=1)
        output_layer6_3 = self.conv63(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_3], dim=1)
        output_layer6_4 = self.conv64(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_4], dim=1)
        output_layer6_5 = self.conv65(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_5], dim=1)
        output_layer6_6 = self.conv66(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_6], dim=1)
        output_layer6_7 = self.conv67(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_7], dim=1)
        output_layer6_8 = self.conv68(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_8], dim=1)

        # 第七层 卷积层
        output_layer7_1 = self.conv71(output_layer6)
        output_layer7 = output_layer7_1
        output_layer7_2 = self.conv72(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_2], dim=1)
        output_layer7_3 = self.conv73(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_3], dim=1)
        output_layer7_4 = self.conv74(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_4], dim=1)
        output_layer7_5 = self.conv75(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_5], dim=1)
        output_layer7_6 = self.conv76(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_6], dim=1)
        output_layer7_7 = self.conv77(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_7], dim=1)
        output_layer7_8 = self.conv78(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_8], dim=1)

        # 第八层 卷积层
        output_layer8_1 = self.conv81(output_layer7)
        output_layer8 = output_layer8_1
        output_layer8_2 = self.conv82(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_2], dim=1)
        output_layer8_3 = self.conv83(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_3], dim=1)
        output_layer8_4 = self.conv84(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_4], dim=1)
        output_layer8_5 = self.conv85(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_5], dim=1)
        output_layer8_6 = self.conv86(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_6], dim=1)
        output_layer8_7 = self.conv87(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_7], dim=1)
        output_layer8_8 = self.conv88(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_8], dim=1)

        # 第九层 卷积层
        output_layer9_1 = self.conv91(output_layer8)
        output_layer9 = output_layer9_1
        output_layer9_2 = self.conv92(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_2], dim=1)
        output_layer9_3 = self.conv93(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_3], dim=1)
        output_layer9_4 = self.conv94(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_4], dim=1)
        output_layer9_5 = self.conv95(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_5], dim=1)
        output_layer9_6 = self.conv96(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_6], dim=1)
        output_layer9_7 = self.conv97(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_7], dim=1)
        output_layer9_8 = self.conv98(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_8], dim=1)

        # 第十层 卷积层
        output_layer10_1 = self.conv101(output_layer9)
        output_layer10 = output_layer10_1
        output_layer10_2 = self.conv102(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_2], dim=1)
        output_layer10_3 = self.conv103(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_3], dim=1)
        output_layer10_4 = self.conv104(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_4], dim=1)
        output_layer10_5 = self.conv105(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_5], dim=1)
        output_layer10_6 = self.conv106(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_6], dim=1)
        output_layer10_7 = self.conv107(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_7], dim=1)
        output_layer10_8 = self.conv108(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_8], dim=1)

        # 第十一层 卷积层
        output_layer11_1 = self.conv111(output_layer10)
        output_layer11 = output_layer11_1
        output_layer11_2 = self.conv112(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_2], dim=1)
        output_layer11_3 = self.conv113(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_3], dim=1)
        output_layer11_4 = self.conv114(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_4], dim=1)
        output_layer11_5 = self.conv115(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_5], dim=1)
        output_layer11_6 = self.conv116(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_6], dim=1)
        output_layer11_7 = self.conv117(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_7], dim=1)
        output_layer11_8 = self.conv118(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_8], dim=1)

        # 第十二层 卷积层
        output_layer12_1 = self.conv121(output_layer11)
        output_layer12 = output_layer12_1
        output_layer12_2 = self.conv122(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_2], dim=1)
        output_layer12_3 = self.conv123(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_3], dim=1)
        output_layer12_4 = self.conv124(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_4], dim=1)
        output_layer12_5 = self.conv125(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_5], dim=1)
        output_layer12_6 = self.conv126(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_6], dim=1)
        output_layer12_7 = self.conv127(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_7], dim=1)
        output_layer12_8 = self.conv128(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_8], dim=1)

        # 第十三层 卷积层
        output_layer13_1 = self.conv131(output_layer12)
        output_layer13 = output_layer13_1
        output_layer13_2 = self.conv132(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_2], dim=1)
        output_layer13_3 = self.conv133(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_3], dim=1)
        output_layer13_4 = self.conv134(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_4], dim=1)
        output_layer13_5 = self.conv135(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_5], dim=1)
        output_layer13_6 = self.conv136(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_6], dim=1)
        output_layer13_7 = self.conv137(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_7], dim=1)
        output_layer13_8 = self.conv138(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_8], dim=1)

        # 第十四层 卷积层
        output_layer14_1 = self.conv141(output_layer13)
        output_layer14 = output_layer14_1
        output_layer14_2 = self.conv142(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_2], dim=1)
        output_layer14_3 = self.conv143(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_3], dim=1)
        output_layer14_4 = self.conv144(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_4], dim=1)
        output_layer14_5 = self.conv145(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_5], dim=1)
        output_layer14_6 = self.conv146(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_6], dim=1)
        output_layer14_7 = self.conv147(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_7], dim=1)
        output_layer14_8 = self.conv148(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_8], dim=1)

        # 第十五层 线性层 特征映射
        output_layer15_1 = output_layer14_1.view(1, -1)
        output_layer15_1 = self.fc151(output_layer15_1)
        output_layer15 = output_layer15_1

        output_layer15_2 = output_layer14_2.view(1, -1)
        output_layer15_2 = self.fc152(output_layer15_2)
        output_layer15 = torch.cat([output_layer15, output_layer15_2], dim=1)

        output_layer15_3 = output_layer14_3.view(1, -1)
        output_layer15_3 = self.fc153(output_layer15_3)
        output_layer15 = torch.cat([output_layer15, output_layer15_3], dim=1)

        output_layer15_4 = output_layer14_4.view(1, -1)
        output_layer15_4 = self.fc154(output_layer15_4)
        output_layer15 = torch.cat([output_layer15, output_layer15_4], dim=1)

        output_layer15_5 = output_layer14_5.view(1, -1)
        output_layer15_5 = self.fc155(output_layer15_5)
        output_layer15 = torch.cat([output_layer15, output_layer15_5], dim=1)

        output_layer15_6 = output_layer14_6.view(1, -1)
        output_layer15_6 = self.fc156(output_layer15_6)
        output_layer15 = torch.cat([output_layer15, output_layer15_6], dim=1)

        output_layer15_7 = output_layer14_7.view(1, -1)
        output_layer15_7 = self.fc157(output_layer15_7)
        output_layer15 = torch.cat([output_layer15, output_layer15_7], dim=1)

        output_layer15_8 = output_layer14_8.view(1, -1)
        output_layer15_8 = self.fc158(output_layer15_8)
        output_layer15 = torch.cat([output_layer15, output_layer15_8], dim=1)

        output_layer15_9 = output_layer14.view(1, -1)
        output_layer15_9 = self.fc159(output_layer15_9)

        # 第十六层 线性层 特征映射
        output_layer16_1 = output_layer15_1.view(1, -1)
        output_layer16_1 = self.fc161(output_layer16_1)
        output_layer16 = output_layer16_1

        output_layer16_2 = output_layer15_2.view(1, -1)
        output_layer16_2 = self.fc162(output_layer16_2)
        output_layer16 = torch.cat([output_layer16, output_layer16_2], dim=1)

        output_layer16_3 = output_layer15_3.view(1, -1)
        output_layer16_3 = self.fc163(output_layer16_3)
        output_layer16 = torch.cat([output_layer16, output_layer16_3], dim=1)

        output_layer16_4 = output_layer15_4.view(1, -1)
        output_layer16_4 = self.fc164(output_layer16_4)
        output_layer16 = torch.cat([output_layer16, output_layer16_4], dim=1)

        output_layer16_5 = output_layer15_5.view(1, -1)
        output_layer16_5 = self.fc165(output_layer16_5)
        output_layer16 = torch.cat([output_layer16, output_layer16_5], dim=1)

        output_layer16_6 = output_layer15_6.view(1, -1)
        output_layer16_6 = self.fc166(output_layer16_6)
        output_layer16 = torch.cat([output_layer16, output_layer16_6], dim=1)

        output_layer16_7 = output_layer15_7.view(1, -1)
        output_layer16_7 = self.fc167(output_layer16_7)
        output_layer16 = torch.cat([output_layer16, output_layer16_7], dim=1)

        output_layer16_8 = output_layer15_8.view(1, -1)
        output_layer16_8 = self.fc168(output_layer16_8)
        output_layer16 = torch.cat([output_layer16, output_layer16_8], dim=1)

        output_layer16_9 = output_layer15.view(1, -1)
        output_layer16_9 = self.fc169(output_layer16_9)

        output_layer16_10 = output_layer15_9.view(1, -1)
        output_layer16_10 = self.fc1610(output_layer16_10)

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


class CNN_SMI2(nn.Module):
    # 卷积神经网络，丢掉之前的LSTM层
    def __init__(self):
        super(CNN_SMI2, self).__init__()

        self.conv11 = QuadrupleConv_U(2, 1)
        self.conv12 = TripleConv_U(2, 1)
        self.conv13 = QuadrupleConv_U(2, 1)
        self.conv14 = TripleConv_U(2, 1)
        self.conv15 = QuadrupleConv_U(2, 1)
        self.conv16 = TripleConv_U(1, 1)
        self.conv17 = QuadrupleConv_U(5, 1)
        self.conv18 = QuadrupleConv_U(6, 1)

        self.conv21 = TripleConv_U(8, 1)
        self.conv22 = QuadrupleConv_U(8, 1)
        self.conv23 = TripleConv_U(8, 1)
        self.conv24 = QuadrupleConv_U(8, 1)
        self.conv25 = TripleConv_U(8, 1)
        self.conv26 = QuadrupleConv_U(8, 1)
        self.conv27 = TripleConv_U(8, 1)
        self.conv28 = QuadrupleConv_U(8, 1)

        self.conv31 = QuadrupleConv_U(8, 1)
        self.conv32 = TripleConv_U(8, 1)
        self.conv33 = QuadrupleConv_U(8, 1)
        self.conv34 = TripleConv_U(8, 1)
        self.conv35 = QuadrupleConv_U(8, 1)
        self.conv36 = TripleConv_U(8, 1)
        self.conv37 = QuadrupleConv_U(8, 1)
        self.conv38 = TripleConv_U(8, 1)

        self.conv41 = TripleConv_U(8, 1)
        self.conv42 = QuadrupleConv_U(8, 1)
        self.conv43 = TripleConv_U(8, 1)
        self.conv44 = QuadrupleConv_U(8, 1)
        self.conv45 = TripleConv_U(8, 1)
        self.conv46 = QuadrupleConv_U(8, 1)
        self.conv47 = TripleConv_U(8, 1)
        self.conv48 = QuadrupleConv_U(8, 1)

        self.conv51 = QuadrupleConv_U(8, 1)
        self.conv52 = TripleConv_U(8, 1)
        self.conv53 = QuadrupleConv_U(8, 1)
        self.conv54 = TripleConv_U(8, 1)
        self.conv55 = QuadrupleConv_U(8, 1)
        self.conv56 = TripleConv_U(8, 1)
        self.conv57 = QuadrupleConv_U(8, 1)
        self.conv58 = TripleConv_U(8, 1)

        self.conv61 = TripleConv_U(8, 1)
        self.conv62 = QuadrupleConv_U(8, 1)
        self.conv63 = TripleConv_U(8, 1)
        self.conv64 = QuadrupleConv_U(8, 1)
        self.conv65 = TripleConv_U(8, 1)
        self.conv66 = QuadrupleConv_U(8, 1)
        self.conv67 = TripleConv_U(8, 1)
        self.conv68 = QuadrupleConv_U(8, 1)

        self.conv71 = QuadrupleConv_U(8, 1)
        self.conv72 = TripleConv_U(8, 1)
        self.conv73 = QuadrupleConv_U(8, 1)
        self.conv74 = TripleConv_U(8, 1)
        self.conv75 = QuadrupleConv_U(8, 1)
        self.conv76 = TripleConv_U(8, 1)
        self.conv77 = QuadrupleConv_U(8, 1)
        self.conv78 = TripleConv_U(8, 1)

        self.conv81 = TripleConv_U(8, 1)
        self.conv82 = QuadrupleConv_U(8, 1)
        self.conv83 = TripleConv_U(8, 1)
        self.conv84 = QuadrupleConv_U(8, 1)
        self.conv85 = TripleConv_U(8, 1)
        self.conv86 = QuadrupleConv_U(8, 1)
        self.conv87 = TripleConv_U(8, 1)
        self.conv88 = QuadrupleConv_U(8, 1)

        self.conv91 = QuadrupleConv_U(8, 1)
        self.conv92 = TripleConv_U(8, 1)
        self.conv93 = QuadrupleConv_U(8, 1)
        self.conv94 = TripleConv_U(8, 1)
        self.conv95 = QuadrupleConv_U(8, 1)
        self.conv96 = TripleConv_U(8, 1)
        self.conv97 = QuadrupleConv_U(8, 1)
        self.conv98 = TripleConv_U(8, 1)

        self.conv101 = TripleConv_U(8, 1)
        self.conv102 = QuadrupleConv_U(8, 1)
        self.conv103 = TripleConv_U(8, 1)
        self.conv104 = QuadrupleConv_U(8, 1)
        self.conv105 = TripleConv_U(8, 1)
        self.conv106 = QuadrupleConv_U(8, 1)
        self.conv107 = TripleConv_U(8, 1)
        self.conv108 = QuadrupleConv_U(8, 1)

        self.conv111 = QuadrupleConv_U(8, 1)
        self.conv112 = TripleConv_U(8, 1)
        self.conv113 = QuadrupleConv_U(8, 1)
        self.conv114 = TripleConv_U(8, 1)
        self.conv115 = QuadrupleConv_U(8, 1)
        self.conv116 = TripleConv_U(8, 1)
        self.conv117 = QuadrupleConv_U(8, 1)
        self.conv118 = TripleConv_U(8, 1)

        self.conv121 = TripleConv_U(8, 1)
        self.conv122 = QuadrupleConv_U(8, 1)
        self.conv123 = TripleConv_U(8, 1)
        self.conv124 = QuadrupleConv_U(8, 1)
        self.conv125 = TripleConv_U(8, 1)
        self.conv126 = QuadrupleConv_U(8, 1)
        self.conv127 = TripleConv_U(8, 1)
        self.conv128 = QuadrupleConv_U(8, 1)

        self.conv131 = QuadrupleConv_U(8, 1)
        self.conv132 = TripleConv_U(8, 1)
        self.conv133 = QuadrupleConv_U(8, 1)
        self.conv134 = TripleConv_U(8, 1)
        self.conv135 = QuadrupleConv_U(8, 1)
        self.conv136 = TripleConv_U(8, 1)
        self.conv137 = QuadrupleConv_U(8, 1)
        self.conv138 = TripleConv_U(8, 1)

        self.conv141 = TripleConv_U(8, 1)
        self.conv142 = QuadrupleConv_U(8, 1)
        self.conv143 = TripleConv_U(8, 1)
        self.conv144 = QuadrupleConv_U(8, 1)
        self.conv145 = TripleConv_U(8, 1)
        self.conv146 = QuadrupleConv_U(8, 1)
        self.conv147 = TripleConv_U(8, 1)
        self.conv148 = QuadrupleConv_U(8, 1)

        self.lstm151 = nn.LSTM(60, 60, 3)
        self.lstm152 = nn.LSTM(60, 60, 3)
        self.lstm153 = nn.LSTM(60, 60, 3)
        self.lstm154 = nn.LSTM(60, 60, 3)
        self.lstm155 = nn.LSTM(60, 60, 3)
        self.lstm156 = nn.LSTM(60, 60, 3)
        self.lstm157 = nn.LSTM(60, 60, 3)
        self.lstm158 = nn.LSTM(60, 60, 3)
        self.lstm159 = nn.LSTM(480, 240, 3)

        self.lstm161 = nn.LSTM(60, 60, 3)
        self.lstm162 = nn.LSTM(60, 60, 3)
        self.lstm163 = nn.LSTM(60, 60, 3)
        self.lstm164 = nn.LSTM(60, 60, 3)
        self.lstm165 = nn.LSTM(60, 60, 3)
        self.lstm166 = nn.LSTM(60, 60, 3)
        self.lstm167 = nn.LSTM(60, 60, 3)
        self.lstm168 = nn.LSTM(60, 60, 3)
        self.lstm169 = nn.LSTM(480, 240, 3)
        self.lstm1610 = nn.LSTM(240, 120, 3)

        self.lstm171 = nn.LSTM(60, 60, 3)
        self.lstm172 = nn.LSTM(60, 60, 3)
        self.lstm173 = nn.LSTM(60, 60, 3)
        self.lstm174 = nn.LSTM(60, 60, 3)
        self.lstm175 = nn.LSTM(60, 60, 3)
        self.lstm176 = nn.LSTM(60, 60, 3)
        self.lstm177 = nn.LSTM(60, 60, 3)
        self.lstm178 = nn.LSTM(60, 60, 3)
        self.lstm179 = nn.LSTM(480, 240, 3)
        self.lstm1710 = nn.LSTM(240, 120, 3)
        self.lstm1711 = nn.LSTM(120, 60, 3)

        self.fc181 = nn.Linear(60, 60)
        self.fc182 = nn.Linear(60, 60)
        self.fc183 = nn.Linear(60, 60)
        self.fc184 = nn.Linear(60, 60)
        self.fc185 = nn.Linear(60, 60)
        self.fc186 = nn.Linear(60, 60)
        self.fc187 = nn.Linear(60, 60)
        self.fc188 = nn.Linear(60, 60)
        self.fc189 = nn.Linear(480, 240)
        self.fc1810 = nn.Linear(240, 120)
        self.fc1811 = nn.Linear(120, 60)
        self.fc1812 = nn.Linear(60, 60)

        self.fc191 = nn.Linear(60, 60)
        self.fc192 = nn.Linear(60, 60)
        self.fc193 = nn.Linear(60, 60)
        self.fc194 = nn.Linear(60, 60)
        self.fc195 = nn.Linear(60, 60)
        self.fc196 = nn.Linear(60, 60)
        self.fc197 = nn.Linear(60, 60)
        self.fc198 = nn.Linear(60, 60)
        self.fc199 = nn.Linear(480, 240)
        self.fc1910 = nn.Linear(240, 120)
        self.fc1911 = nn.Linear(120, 60)
        self.fc1912 = nn.Linear(60, 60)
        self.fc1913 = nn.Linear(60, 60)

        self.fc201 = nn.Linear(480, 120)
        self.fc202 = nn.Linear(240, 60)
        self.fc203 = nn.Linear(120, 60)
        self.fc204 = nn.Linear(60, 60)
        self.fc205 = nn.Linear(60, 60)
        self.fc206 = nn.Linear(60, 60)

        self.fc211 = nn.Linear(240, 120)
        self.fc212 = nn.Linear(180, 60)

        self.fc22 = nn.Linear(180, 60)

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
        layer_in_1_1 = layer_in_1_1.view(layer_in_1_1.size(0), layer_in_1_1.size(1), -1, 10)

        layer_in_1_2 = train_well[:, 1, :]
        layer_in_1_2 = torch.cat([layer_in_1_2.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_2 = layer_in_1_2.view(layer_in_1_2.size(0), layer_in_1_2.size(1), -1, 10)

        layer_in_1_3 = train_well[:, 2, :]
        layer_in_1_3 = torch.cat([layer_in_1_3.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_3 = layer_in_1_3.view(layer_in_1_3.size(0), layer_in_1_3.size(1), -1, 10)

        layer_in_1_4 = train_well[:, 3, :]
        layer_in_1_4 = torch.cat([layer_in_1_4.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_4 = layer_in_1_4.view(layer_in_1_4.size(0), layer_in_1_4.size(1), -1, 10)

        layer_in_1_5 = train_well[:, 4, :]
        layer_in_1_5 = torch.cat([layer_in_1_5.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_5 = layer_in_1_5.view(layer_in_1_5.size(0), layer_in_1_5.size(1), -1, 10)

        layer_in_1_6 = seismic.view(1, 1, -1)
        layer_in_1_6 = layer_in_1_6.view(seismic.size(0), seismic.size(0), -1, 10)

        layer_in_1_7 = train_well.view(1, 5, -1)
        layer_in_1_7 = layer_in_1_7.view(train_well.size(0), train_well.size(1), -1, 10)

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
        output_layer2_1 = self.conv21(output_layer1)
        output_layer2 = output_layer2_1
        output_layer2_2 = self.conv22(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_2], dim=1)
        output_layer2_3 = self.conv23(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_3], dim=1)
        output_layer2_4 = self.conv24(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_4], dim=1)
        output_layer2_5 = self.conv25(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_5], dim=1)
        output_layer2_6 = self.conv26(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_6], dim=1)
        output_layer2_7 = self.conv27(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_7], dim=1)
        output_layer2_8 = self.conv28(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_8], dim=1)

        # 第三层 卷积层
        output_layer3_1 = self.conv31(output_layer2)
        output_layer3 = output_layer3_1
        output_layer3_2 = self.conv32(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_2], dim=1)
        output_layer3_3 = self.conv33(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_3], dim=1)
        output_layer3_4 = self.conv34(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_4], dim=1)
        output_layer3_5 = self.conv35(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_5], dim=1)
        output_layer3_6 = self.conv36(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_6], dim=1)
        output_layer3_7 = self.conv37(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_7], dim=1)
        output_layer3_8 = self.conv38(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_8], dim=1)

        # 第四层 卷积层
        output_layer4_1 = self.conv41(output_layer3)
        output_layer4 = output_layer4_1
        output_layer4_2 = self.conv42(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_2], dim=1)
        output_layer4_3 = self.conv43(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_3], dim=1)
        output_layer4_4 = self.conv44(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_4], dim=1)
        output_layer4_5 = self.conv45(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_5], dim=1)
        output_layer4_6 = self.conv46(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_6], dim=1)
        output_layer4_7 = self.conv47(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_7], dim=1)
        output_layer4_8 = self.conv48(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_8], dim=1)

        # 第五层 卷积层
        output_layer5_1 = self.conv51(output_layer4)
        output_layer5 = output_layer5_1
        output_layer5_2 = self.conv52(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_2], dim=1)
        output_layer5_3 = self.conv53(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_3], dim=1)
        output_layer5_4 = self.conv54(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_4], dim=1)
        output_layer5_5 = self.conv55(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_5], dim=1)
        output_layer5_6 = self.conv56(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_6], dim=1)
        output_layer5_7 = self.conv57(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_7], dim=1)
        output_layer5_8 = self.conv58(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_8], dim=1)

        # 第六层 卷积层
        output_layer6_1 = self.conv61(output_layer5)
        output_layer6 = output_layer6_1
        output_layer6_2 = self.conv62(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_2], dim=1)
        output_layer6_3 = self.conv63(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_3], dim=1)
        output_layer6_4 = self.conv64(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_4], dim=1)
        output_layer6_5 = self.conv65(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_5], dim=1)
        output_layer6_6 = self.conv66(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_6], dim=1)
        output_layer6_7 = self.conv67(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_7], dim=1)
        output_layer6_8 = self.conv68(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_8], dim=1)

        # 第七层 卷积层
        output_layer7_1 = self.conv71(output_layer6)
        output_layer7 = output_layer7_1
        output_layer7_2 = self.conv72(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_2], dim=1)
        output_layer7_3 = self.conv73(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_3], dim=1)
        output_layer7_4 = self.conv74(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_4], dim=1)
        output_layer7_5 = self.conv75(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_5], dim=1)
        output_layer7_6 = self.conv76(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_6], dim=1)
        output_layer7_7 = self.conv77(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_7], dim=1)
        output_layer7_8 = self.conv78(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_8], dim=1)

        # 第八层 卷积层
        output_layer8_1 = self.conv81(output_layer7)
        output_layer8 = output_layer8_1
        output_layer8_2 = self.conv82(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_2], dim=1)
        output_layer8_3 = self.conv83(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_3], dim=1)
        output_layer8_4 = self.conv84(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_4], dim=1)
        output_layer8_5 = self.conv85(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_5], dim=1)
        output_layer8_6 = self.conv86(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_6], dim=1)
        output_layer8_7 = self.conv87(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_7], dim=1)
        output_layer8_8 = self.conv88(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_8], dim=1)

        # 第九层 卷积层
        output_layer9_1 = self.conv91(output_layer8)
        output_layer9 = output_layer9_1
        output_layer9_2 = self.conv92(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_2], dim=1)
        output_layer9_3 = self.conv93(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_3], dim=1)
        output_layer9_4 = self.conv94(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_4], dim=1)
        output_layer9_5 = self.conv95(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_5], dim=1)
        output_layer9_6 = self.conv96(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_6], dim=1)
        output_layer9_7 = self.conv97(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_7], dim=1)
        output_layer9_8 = self.conv98(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_8], dim=1)

        # 第十层 卷积层
        output_layer10_1 = self.conv101(output_layer9)
        output_layer10 = output_layer10_1
        output_layer10_2 = self.conv102(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_2], dim=1)
        output_layer10_3 = self.conv103(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_3], dim=1)
        output_layer10_4 = self.conv104(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_4], dim=1)
        output_layer10_5 = self.conv105(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_5], dim=1)
        output_layer10_6 = self.conv106(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_6], dim=1)
        output_layer10_7 = self.conv107(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_7], dim=1)
        output_layer10_8 = self.conv108(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_8], dim=1)

        # 第十一层 卷积层
        output_layer11_1 = self.conv111(output_layer10)
        output_layer11 = output_layer11_1
        output_layer11_2 = self.conv112(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_2], dim=1)
        output_layer11_3 = self.conv113(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_3], dim=1)
        output_layer11_4 = self.conv114(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_4], dim=1)
        output_layer11_5 = self.conv115(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_5], dim=1)
        output_layer11_6 = self.conv116(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_6], dim=1)
        output_layer11_7 = self.conv117(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_7], dim=1)
        output_layer11_8 = self.conv118(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_8], dim=1)

        # 第十二层 卷积层
        output_layer12_1 = self.conv121(output_layer11)
        output_layer12 = output_layer12_1
        output_layer12_2 = self.conv122(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_2], dim=1)
        output_layer12_3 = self.conv123(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_3], dim=1)
        output_layer12_4 = self.conv124(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_4], dim=1)
        output_layer12_5 = self.conv125(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_5], dim=1)
        output_layer12_6 = self.conv126(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_6], dim=1)
        output_layer12_7 = self.conv127(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_7], dim=1)
        output_layer12_8 = self.conv128(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_8], dim=1)

        # 第十三层 卷积层
        output_layer13_1 = self.conv131(output_layer12)
        output_layer13 = output_layer13_1
        output_layer13_2 = self.conv132(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_2], dim=1)
        output_layer13_3 = self.conv133(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_3], dim=1)
        output_layer13_4 = self.conv134(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_4], dim=1)
        output_layer13_5 = self.conv135(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_5], dim=1)
        output_layer13_6 = self.conv136(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_6], dim=1)
        output_layer13_7 = self.conv137(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_7], dim=1)
        output_layer13_8 = self.conv138(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_8], dim=1)

        # 第十四层 卷积层
        output_layer14_1 = self.conv141(output_layer13)
        output_layer14 = output_layer14_1
        output_layer14_2 = self.conv142(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_2], dim=1)
        output_layer14_3 = self.conv143(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_3], dim=1)
        output_layer14_4 = self.conv144(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_4], dim=1)
        output_layer14_5 = self.conv145(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_5], dim=1)
        output_layer14_6 = self.conv146(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_6], dim=1)
        output_layer14_7 = self.conv147(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_7], dim=1)
        output_layer14_8 = self.conv148(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_8], dim=1)

        # 第十五层 LSTM层 特征映射
        output_layer15_1 = output_layer14_1.view(1, 1, -1)
        output_layer15_1, temp15_1 = self.lstm151(output_layer15_1)
        output_layer15 = output_layer15_1

        output_layer15_2 = output_layer14_2.view(1, 1, -1)
        output_layer15_2, temp15_2 = self.lstm152(output_layer15_2)
        output_layer15 = torch.cat([output_layer15, output_layer15_2], dim=2)

        output_layer15_3 = output_layer14_3.view(1, 1, -1)
        output_layer15_3, temp15_3 = self.lstm153(output_layer15_3)
        output_layer15 = torch.cat([output_layer15, output_layer15_3], dim=2)

        output_layer15_4 = output_layer14_4.view(1, 1, -1)
        output_layer15_4, temp15_4 = self.lstm154(output_layer15_4)
        output_layer15 = torch.cat([output_layer15, output_layer15_4], dim=2)

        output_layer15_5 = output_layer14_5.view(1, 1, -1)
        output_layer15_5, temp15_5 = self.lstm155(output_layer15_5)
        output_layer15 = torch.cat([output_layer15, output_layer15_5], dim=2)

        output_layer15_6 = output_layer14_6.view(1, 1, -1)
        output_layer15_6, temp15_6 = self.lstm156(output_layer15_6)
        output_layer15 = torch.cat([output_layer15, output_layer15_6], dim=2)

        output_layer15_7 = output_layer14_7.view(1, 1, -1)
        output_layer15_7, temp15_7 = self.lstm157(output_layer15_7)
        output_layer15 = torch.cat([output_layer15, output_layer15_7], dim=2)

        output_layer15_8 = output_layer14_8.view(1, 1, -1)
        output_layer15_8, temp15_8 = self.lstm158(output_layer15_8)
        output_layer15 = torch.cat([output_layer15, output_layer15_8], dim=2)

        output_layer15_9 = output_layer14.view(1, 1, -1)
        output_layer15_9, temp15_9 = self.lstm159(output_layer15_9)

        # 第十六层 LSTM层 特征映射
        output_layer16_1, temp16_1 = self.lstm161(output_layer15_1)
        output_layer16 = output_layer16_1

        output_layer16_2, temp16_2 = self.lstm162(output_layer15_2)
        output_layer16 = torch.cat([output_layer16, output_layer16_2], dim=2)

        output_layer16_3, temp16_3 = self.lstm163(output_layer15_3)
        output_layer16 = torch.cat([output_layer16, output_layer16_3], dim=2)

        output_layer16_4, temp16_4 = self.lstm164(output_layer15_4)
        output_layer16 = torch.cat([output_layer16, output_layer16_4], dim=2)

        output_layer16_5, temp16_5 = self.lstm165(output_layer15_5)
        output_layer16 = torch.cat([output_layer16, output_layer16_5], dim=2)

        output_layer16_6, temp16_6 = self.lstm166(output_layer15_6)
        output_layer16 = torch.cat([output_layer16, output_layer16_6], dim=2)

        output_layer16_7, temp16_7 = self.lstm167(output_layer15_7)
        output_layer16 = torch.cat([output_layer16, output_layer16_7], dim=2)

        output_layer16_8, temp16_8 = self.lstm168(output_layer15_8)
        output_layer16 = torch.cat([output_layer16, output_layer16_8], dim=2)

        output_layer16_9, temp16_9 = self.lstm169(output_layer15)

        output_layer16_10, temp16_10 = self.lstm1610(output_layer15_9)

        # 第十七层 LSTM层 特征映射
        output_layer17_1, temp17_1 = self.lstm171(output_layer16_1)
        output_layer17 = output_layer17_1

        output_layer17_2, temp17_2 = self.lstm172(output_layer16_2)
        output_layer17 = torch.cat([output_layer17, output_layer17_2], dim=2)

        output_layer17_3, temp17_3 = self.lstm173(output_layer16_3)
        output_layer17 = torch.cat([output_layer17, output_layer17_3], dim=2)

        output_layer17_4, temp17_4 = self.lstm174(output_layer16_4)
        output_layer17 = torch.cat([output_layer17, output_layer17_4], dim=2)

        output_layer17_5, temp17_5 = self.lstm175(output_layer16_5)
        output_layer17 = torch.cat([output_layer17, output_layer17_5], dim=2)

        output_layer17_6, temp17_6 = self.lstm176(output_layer16_6)
        output_layer17 = torch.cat([output_layer17, output_layer17_6], dim=2)

        output_layer17_7, temp17_7 = self.lstm177(output_layer16_7)
        output_layer17 = torch.cat([output_layer17, output_layer17_7], dim=2)

        output_layer17_8, temp17_8 = self.lstm178(output_layer16_8)
        output_layer17 = torch.cat([output_layer17, output_layer17_8], dim=2)

        output_layer17_9, temp17_9 = self.lstm179(output_layer16)

        output_layer17_10, temp17_10 = self.lstm1710(output_layer16_9)

        output_layer17_11, temp17_11 = self.lstm1711(output_layer16_10)

        # 第十八层 线性层 特征映射
        output_layer18_1 = output_layer17_1.view(1, -1)
        output_layer18_1 = self.fc181(output_layer18_1)
        output_layer18 = output_layer18_1

        output_layer18_2 = output_layer17_2.view(1, -1)
        output_layer18_2 = self.fc182(output_layer18_2)
        output_layer18 = torch.cat([output_layer18, output_layer18_2], dim=1)

        output_layer18_3 = output_layer17_3.view(1, -1)
        output_layer18_3 = self.fc183(output_layer18_3)
        output_layer18 = torch.cat([output_layer18, output_layer18_3], dim=1)

        output_layer18_4 = output_layer17_4.view(1, -1)
        output_layer18_4 = self.fc184(output_layer18_4)
        output_layer18 = torch.cat([output_layer18, output_layer18_4], dim=1)

        output_layer18_5 = output_layer17_5.view(1, -1)
        output_layer18_5 = self.fc185(output_layer18_5)
        output_layer18 = torch.cat([output_layer18, output_layer18_5], dim=1)

        output_layer18_6 = output_layer17_6.view(1, -1)
        output_layer18_6 = self.fc186(output_layer18_6)
        output_layer18 = torch.cat([output_layer18, output_layer18_6], dim=1)

        output_layer18_7 = output_layer17_7.view(1, -1)
        output_layer18_7 = self.fc187(output_layer18_7)
        output_layer18 = torch.cat([output_layer18, output_layer18_7], dim=1)

        output_layer18_8 = output_layer17_8.view(1, -1)
        output_layer18_8 = self.fc188(output_layer18_8)
        output_layer18 = torch.cat([output_layer18, output_layer18_8], dim=1)

        output_layer18_9 = output_layer17.view(1, -1)
        output_layer18_9 = self.fc189(output_layer18_9)

        output_layer18_10 = output_layer17_9.view(1, -1)
        output_layer18_10 = self.fc1810(output_layer18_10)

        output_layer18_11 = output_layer17_10.view(1, -1)
        output_layer18_11 = self.fc1811(output_layer18_11)

        output_layer18_12 = output_layer17_11.view(1, -1)
        output_layer18_12 = self.fc1812(output_layer18_12)

        # 第十九层 线性层 特征映射
        output_layer19_1 = self.fc191(output_layer18_1)
        output_layer19 = output_layer19_1

        output_layer19_2 = self.fc192(output_layer18_2)
        output_layer19 = torch.cat([output_layer19, output_layer19_2], dim=1)

        output_layer19_3 = self.fc193(output_layer18_3)
        output_layer19 = torch.cat([output_layer19, output_layer19_3], dim=1)

        output_layer19_4 = self.fc194(output_layer18_4)
        output_layer19 = torch.cat([output_layer19, output_layer19_4], dim=1)

        output_layer19_5 = self.fc195(output_layer18_5)
        output_layer19 = torch.cat([output_layer19, output_layer19_5], dim=1)

        output_layer19_6 = self.fc196(output_layer18_6)
        output_layer19 = torch.cat([output_layer19, output_layer19_6], dim=1)

        output_layer19_7 = self.fc197(output_layer18_7)
        output_layer19 = torch.cat([output_layer19, output_layer19_7], dim=1)

        output_layer19_8 = self.fc198(output_layer18_8)
        output_layer19 = torch.cat([output_layer19, output_layer19_8], dim=1)

        output_layer19_9 = self.fc199(output_layer18)

        output_layer19_10 = self.fc1910(output_layer18_9)

        output_layer19_11 = self.fc1911(output_layer18_10)

        output_layer19_12 = self.fc1912(output_layer18_11)

        output_layer19_13 = self.fc1913(output_layer18_12)

        # 第二十层 线性层
        output_layer20_1 = self.fc201(output_layer19)
        output_layer20_2 = self.fc202(output_layer19_9)
        output_layer20_3 = self.fc203(output_layer19_10)
        output_layer20_4 = self.fc204(output_layer19_11)
        output_layer20_5 = self.fc205(output_layer19_12)
        output_layer20_6 = self.fc206(output_layer19_13)
        output_layer20_7 = torch.cat([output_layer20_1, output_layer20_2], dim=1)
        output_layer20_7 = torch.cat([output_layer20_7, output_layer20_3], dim=1)
        output_layer20_8 = torch.cat([output_layer20_4, output_layer20_5], dim=1)
        output_layer20_8 = torch.cat([output_layer20_8, output_layer20_6], dim=1)

        # 第二十一层 线性层
        output_layer21_1 = self.fc211(output_layer20_7)
        output_layer21_2 = self.fc212(output_layer20_8)
        output_layer21 = torch.cat([output_layer21_1, output_layer21_2], dim=1)

        # 第二十二层 线性层
        output_layer22 = self.fc212(output_layer21)

        # 输出层
        output = output_layer22
        return output


class Deep_CNN_SMI2(nn.Module):
    # 卷积神经网络，丢掉之前的LSTM层
    def __init__(self):
        super(Deep_CNN_SMI2, self).__init__()

        self.conv0101 = QuadrupleConv_U(2, 1)
        self.conv0102 = TripleConv_U(2, 1)
        self.conv0103 = QuadrupleConv_U(2, 1)
        self.conv0104 = TripleConv_U(2, 1)
        self.conv0105 = QuadrupleConv_U(2, 1)
        self.conv0106 = TripleConv_U(1, 1)
        self.conv0107 = QuadrupleConv_U(5, 1)
        self.conv0108 = TripleConv_U(6, 1)
        self.conv0109 = QuadrupleConv_U(1, 1)
        self.conv0110 = TripleConv_U(1, 1)
        self.conv0111 = QuadrupleConv_U(1, 1)
        self.conv0112 = TripleConv_U(1, 1)
        self.conv0113 = QuadrupleConv_U(1, 1)

        self.conv0201 = TripleConv_U(13, 1)
        self.conv0202 = QuadrupleConv_U(13, 1)
        self.conv0203 = TripleConv_U(13, 1)
        self.conv0204 = QuadrupleConv_U(13, 1)
        self.conv0205 = TripleConv_U(13, 1)
        self.conv0206 = QuadrupleConv_U(13, 1)
        self.conv0207 = TripleConv_U(13, 1)
        self.conv0208 = QuadrupleConv_U(13, 1)
        self.conv0209 = TripleConv_U(13, 1)
        self.conv0210 = QuadrupleConv_U(13, 1)
        self.conv0211 = TripleConv_U(13, 1)
        self.conv0212 = QuadrupleConv_U(13, 1)
        self.conv0213 = TripleConv_U(13, 1)

        self.conv0301 = QuadrupleConv_U(13, 1)
        self.conv0302 = TripleConv_U(13, 1)
        self.conv0303 = QuadrupleConv_U(13, 1)
        self.conv0304 = TripleConv_U(13, 1)
        self.conv0305 = QuadrupleConv_U(13, 1)
        self.conv0306 = TripleConv_U(13, 1)
        self.conv0307 = QuadrupleConv_U(13, 1)
        self.conv0308 = TripleConv_U(13, 1)
        self.conv0309 = QuadrupleConv_U(13, 1)
        self.conv0310 = TripleConv_U(13, 1)
        self.conv0311 = QuadrupleConv_U(13, 1)
        self.conv0312 = TripleConv_U(13, 1)
        self.conv0313 = QuadrupleConv_U(13, 1)

        self.conv0401 = TripleConv_U(13, 1)
        self.conv0402 = QuadrupleConv_U(13, 1)
        self.conv0403 = TripleConv_U(13, 1)
        self.conv0404 = QuadrupleConv_U(13, 1)
        self.conv0405 = TripleConv_U(13, 1)
        self.conv0406 = QuadrupleConv_U(13, 1)
        self.conv0407 = TripleConv_U(13, 1)
        self.conv0408 = QuadrupleConv_U(13, 1)
        self.conv0409 = TripleConv_U(13, 1)
        self.conv0410 = QuadrupleConv_U(13, 1)
        self.conv0411 = TripleConv_U(13, 1)
        self.conv0412 = QuadrupleConv_U(13, 1)
        self.conv0413 = TripleConv_U(13, 1)

        self.conv0501 = QuadrupleConv_U(13, 1)
        self.conv0502 = TripleConv_U(13, 1)
        self.conv0503 = QuadrupleConv_U(13, 1)
        self.conv0504 = TripleConv_U(13, 1)
        self.conv0505 = QuadrupleConv_U(13, 1)
        self.conv0506 = TripleConv_U(13, 1)
        self.conv0507 = QuadrupleConv_U(13, 1)
        self.conv0508 = TripleConv_U(13, 1)
        self.conv0509 = QuadrupleConv_U(13, 1)
        self.conv0510 = TripleConv_U(13, 1)
        self.conv0511 = QuadrupleConv_U(13, 1)
        self.conv0512 = TripleConv_U(13, 1)
        self.conv0513 = QuadrupleConv_U(13, 1)

        self.conv0601 = TripleConv_U(13, 1)
        self.conv0602 = QuadrupleConv_U(13, 1)
        self.conv0603 = TripleConv_U(13, 1)
        self.conv0604 = QuadrupleConv_U(13, 1)
        self.conv0605 = TripleConv_U(13, 1)
        self.conv0606 = QuadrupleConv_U(13, 1)
        self.conv0607 = TripleConv_U(13, 1)
        self.conv0608 = QuadrupleConv_U(13, 1)
        self.conv0609 = TripleConv_U(13, 1)
        self.conv0610 = QuadrupleConv_U(13, 1)
        self.conv0611 = TripleConv_U(13, 1)
        self.conv0612 = QuadrupleConv_U(13, 1)
        self.conv0613 = TripleConv_U(13, 1)

        self.conv0701 = QuadrupleConv_U(13, 1)
        self.conv0702 = TripleConv_U(13, 1)
        self.conv0703 = QuadrupleConv_U(13, 1)
        self.conv0704 = TripleConv_U(13, 1)
        self.conv0705 = QuadrupleConv_U(13, 1)
        self.conv0706 = TripleConv_U(13, 1)
        self.conv0707 = QuadrupleConv_U(13, 1)
        self.conv0708 = TripleConv_U(13, 1)
        self.conv0709 = QuadrupleConv_U(13, 1)
        self.conv0710 = TripleConv_U(13, 1)
        self.conv0711 = QuadrupleConv_U(13, 1)
        self.conv0712 = TripleConv_U(13, 1)
        self.conv0713 = QuadrupleConv_U(13, 1)

        self.conv0801 = TripleConv_U(13, 1)
        self.conv0802 = QuadrupleConv_U(13, 1)
        self.conv0803 = TripleConv_U(13, 1)
        self.conv0804 = QuadrupleConv_U(13, 1)
        self.conv0805 = TripleConv_U(13, 1)
        self.conv0806 = QuadrupleConv_U(13, 1)
        self.conv0807 = TripleConv_U(13, 1)
        self.conv0808 = QuadrupleConv_U(13, 1)
        self.conv0809 = TripleConv_U(13, 1)
        self.conv0810 = QuadrupleConv_U(13, 1)
        self.conv0811 = TripleConv_U(13, 1)
        self.conv0812 = QuadrupleConv_U(13, 1)
        self.conv0813 = TripleConv_U(13, 1)

        self.conv0901 = QuadrupleConv_U(13, 1)
        self.conv0902 = TripleConv_U(13, 1)
        self.conv0903 = QuadrupleConv_U(13, 1)
        self.conv0904 = TripleConv_U(13, 1)
        self.conv0905 = QuadrupleConv_U(13, 1)
        self.conv0906 = TripleConv_U(13, 1)
        self.conv0907 = QuadrupleConv_U(13, 1)
        self.conv0908 = TripleConv_U(13, 1)
        self.conv0909 = QuadrupleConv_U(13, 1)
        self.conv0910 = TripleConv_U(13, 1)
        self.conv0911 = QuadrupleConv_U(13, 1)
        self.conv0912 = TripleConv_U(13, 1)
        self.conv0913 = QuadrupleConv_U(13, 1)

        self.conv1001 = TripleConv_U(13, 1)
        self.conv1002 = QuadrupleConv_U(13, 1)
        self.conv1003 = TripleConv_U(13, 1)
        self.conv1004 = QuadrupleConv_U(13, 1)
        self.conv1005 = TripleConv_U(13, 1)
        self.conv1006 = QuadrupleConv_U(13, 1)
        self.conv1007 = TripleConv_U(13, 1)
        self.conv1008 = QuadrupleConv_U(13, 1)
        self.conv1009 = TripleConv_U(13, 1)
        self.conv1010 = QuadrupleConv_U(13, 1)
        self.conv1011 = TripleConv_U(13, 1)
        self.conv1012 = QuadrupleConv_U(13, 1)
        self.conv1013 = TripleConv_U(13, 1)

        self.conv1101 = QuadrupleConv_U(13, 1)
        self.conv1102 = TripleConv_U(13, 1)
        self.conv1103 = QuadrupleConv_U(13, 1)
        self.conv1104 = TripleConv_U(13, 1)
        self.conv1105 = QuadrupleConv_U(13, 1)
        self.conv1106 = TripleConv_U(13, 1)
        self.conv1107 = QuadrupleConv_U(13, 1)
        self.conv1108 = TripleConv_U(13, 1)
        self.conv1109 = QuadrupleConv_U(13, 1)
        self.conv1110 = TripleConv_U(13, 1)
        self.conv1111 = QuadrupleConv_U(13, 1)
        self.conv1112 = TripleConv_U(13, 1)
        self.conv1113 = QuadrupleConv_U(13, 1)

        self.conv1201 = TripleConv_U(13, 1)
        self.conv1202 = QuadrupleConv_U(13, 1)
        self.conv1203 = TripleConv_U(13, 1)
        self.conv1204 = QuadrupleConv_U(13, 1)
        self.conv1205 = TripleConv_U(13, 1)
        self.conv1206 = QuadrupleConv_U(13, 1)
        self.conv1207 = TripleConv_U(13, 1)
        self.conv1208 = QuadrupleConv_U(13, 1)
        self.conv1209 = TripleConv_U(13, 1)
        self.conv1210 = QuadrupleConv_U(13, 1)
        self.conv1211 = TripleConv_U(13, 1)
        self.conv1212 = QuadrupleConv_U(13, 1)
        self.conv1213 = TripleConv_U(13, 1)

        self.conv1301 = QuadrupleConv_U(13, 1)
        self.conv1302 = TripleConv_U(13, 1)
        self.conv1303 = QuadrupleConv_U(13, 1)
        self.conv1304 = TripleConv_U(13, 1)
        self.conv1305 = QuadrupleConv_U(13, 1)
        self.conv1306 = TripleConv_U(13, 1)
        self.conv1307 = QuadrupleConv_U(13, 1)
        self.conv1308 = TripleConv_U(13, 1)
        self.conv1309 = QuadrupleConv_U(13, 1)
        self.conv1310 = TripleConv_U(13, 1)
        self.conv1311 = QuadrupleConv_U(13, 1)
        self.conv1312 = TripleConv_U(13, 1)
        self.conv1313 = QuadrupleConv_U(13, 1)

        self.conv1401 = TripleConv_U(13, 1)
        self.conv1402 = QuadrupleConv_U(13, 1)
        self.conv1403 = TripleConv_U(13, 1)
        self.conv1404 = QuadrupleConv_U(13, 1)
        self.conv1405 = TripleConv_U(13, 1)
        self.conv1406 = QuadrupleConv_U(13, 1)
        self.conv1407 = TripleConv_U(13, 1)
        self.conv1408 = QuadrupleConv_U(13, 1)
        self.conv1409 = TripleConv_U(13, 1)
        self.conv1410 = QuadrupleConv_U(13, 1)
        self.conv1411 = TripleConv_U(13, 1)
        self.conv1412 = QuadrupleConv_U(13, 1)
        self.conv1413 = TripleConv_U(13, 1)

        self.conv1501 = QuadrupleConv_U(13, 1)
        self.conv1502 = TripleConv_U(13, 1)
        self.conv1503 = QuadrupleConv_U(13, 1)
        self.conv1504 = TripleConv_U(13, 1)
        self.conv1505 = QuadrupleConv_U(13, 1)
        self.conv1506 = TripleConv_U(13, 1)
        self.conv1507 = QuadrupleConv_U(13, 1)
        self.conv1508 = TripleConv_U(13, 1)
        self.conv1509 = QuadrupleConv_U(13, 1)
        self.conv1510 = TripleConv_U(13, 1)
        self.conv1511 = QuadrupleConv_U(13, 1)
        self.conv1512 = TripleConv_U(13, 1)
        self.conv1513 = QuadrupleConv_U(13, 1)

        self.conv1601 = TripleConv_U(13, 1)
        self.conv1602 = QuadrupleConv_U(13, 1)
        self.conv1603 = TripleConv_U(13, 1)
        self.conv1604 = QuadrupleConv_U(13, 1)
        self.conv1605 = TripleConv_U(13, 1)
        self.conv1606 = QuadrupleConv_U(13, 1)
        self.conv1607 = TripleConv_U(13, 1)
        self.conv1608 = QuadrupleConv_U(13, 1)
        self.conv1609 = TripleConv_U(13, 1)
        self.conv1610 = QuadrupleConv_U(13, 1)
        self.conv1611 = TripleConv_U(13, 1)
        self.conv1612 = QuadrupleConv_U(13, 1)
        self.conv1613 = TripleConv_U(13, 1)

        self.conv1701 = QuadrupleConv_U(13, 1)
        self.conv1702 = TripleConv_U(13, 1)
        self.conv1703 = QuadrupleConv_U(13, 1)
        self.conv1704 = TripleConv_U(13, 1)
        self.conv1705 = QuadrupleConv_U(13, 1)
        self.conv1706 = TripleConv_U(13, 1)
        self.conv1707 = QuadrupleConv_U(13, 1)
        self.conv1708 = TripleConv_U(13, 1)
        self.conv1709 = QuadrupleConv_U(13, 1)
        self.conv1710 = TripleConv_U(13, 1)
        self.conv1711 = QuadrupleConv_U(13, 1)
        self.conv1712 = TripleConv_U(13, 1)
        self.conv1713 = QuadrupleConv_U(13, 1)

        self.conv1801 = TripleConv_U(13, 1)
        self.conv1802 = QuadrupleConv_U(13, 1)
        self.conv1803 = TripleConv_U(13, 1)
        self.conv1804 = QuadrupleConv_U(13, 1)
        self.conv1805 = TripleConv_U(13, 1)
        self.conv1806 = QuadrupleConv_U(13, 1)
        self.conv1807 = TripleConv_U(13, 1)
        self.conv1808 = QuadrupleConv_U(13, 1)
        self.conv1809 = TripleConv_U(13, 1)
        self.conv1810 = QuadrupleConv_U(13, 1)
        self.conv1811 = TripleConv_U(13, 1)
        self.conv1812 = QuadrupleConv_U(13, 1)
        self.conv1813 = TripleConv_U(13, 1)

        self.conv1901 = QuadrupleConv_U(13, 1)
        self.conv1902 = TripleConv_U(13, 1)
        self.conv1903 = QuadrupleConv_U(13, 1)
        self.conv1904 = TripleConv_U(13, 1)
        self.conv1905 = QuadrupleConv_U(13, 1)
        self.conv1906 = TripleConv_U(13, 1)
        self.conv1907 = QuadrupleConv_U(13, 1)
        self.conv1908 = TripleConv_U(13, 1)
        self.conv1909 = QuadrupleConv_U(13, 1)
        self.conv1910 = TripleConv_U(13, 1)
        self.conv1911 = QuadrupleConv_U(13, 1)
        self.conv1912 = TripleConv_U(13, 1)
        self.conv1913 = QuadrupleConv_U(13, 1)

        self.conv2001 = TripleConv_U(13, 1)
        self.conv2002 = QuadrupleConv_U(13, 1)
        self.conv2003 = TripleConv_U(13, 1)
        self.conv2004 = QuadrupleConv_U(13, 1)
        self.conv2005 = TripleConv_U(13, 1)
        self.conv2006 = QuadrupleConv_U(13, 1)
        self.conv2007 = TripleConv_U(13, 1)
        self.conv2008 = QuadrupleConv_U(13, 1)
        self.conv2009 = TripleConv_U(13, 1)
        self.conv2010 = QuadrupleConv_U(13, 1)
        self.conv2011 = TripleConv_U(13, 1)
        self.conv2012 = QuadrupleConv_U(13, 1)
        self.conv2013 = TripleConv_U(13, 1)

        self.conv2101 = QuadrupleConv_U(13, 1)
        self.conv2102 = TripleConv_U(13, 1)
        self.conv2103 = QuadrupleConv_U(13, 1)
        self.conv2104 = TripleConv_U(13, 1)
        self.conv2105 = QuadrupleConv_U(13, 1)
        self.conv2106 = TripleConv_U(13, 1)
        self.conv2107 = QuadrupleConv_U(13, 1)
        self.conv2108 = TripleConv_U(13, 1)
        self.conv2109 = QuadrupleConv_U(13, 1)
        self.conv2110 = TripleConv_U(13, 1)
        self.conv2111 = QuadrupleConv_U(13, 1)
        self.conv2112 = TripleConv_U(13, 1)
        self.conv2113 = QuadrupleConv_U(13, 1)

        self.conv2201 = TripleConv_U(13, 1)
        self.conv2202 = QuadrupleConv_U(13, 1)
        self.conv2203 = TripleConv_U(13, 1)
        self.conv2204 = QuadrupleConv_U(13, 1)
        self.conv2205 = TripleConv_U(13, 1)
        self.conv2206 = QuadrupleConv_U(13, 1)
        self.conv2207 = TripleConv_U(13, 1)
        self.conv2208 = QuadrupleConv_U(13, 1)
        self.conv2209 = TripleConv_U(13, 1)
        self.conv2210 = QuadrupleConv_U(13, 1)
        self.conv2211 = TripleConv_U(13, 1)
        self.conv2212 = QuadrupleConv_U(13, 1)
        self.conv2213 = TripleConv_U(13, 1)

        self.conv2301 = QuadrupleConv_U(13, 1)
        self.conv2302 = TripleConv_U(13, 1)
        self.conv2303 = QuadrupleConv_U(13, 1)
        self.conv2304 = TripleConv_U(13, 1)
        self.conv2305 = QuadrupleConv_U(13, 1)
        self.conv2306 = TripleConv_U(13, 1)
        self.conv2307 = QuadrupleConv_U(13, 1)
        self.conv2308 = TripleConv_U(13, 1)
        self.conv2309 = QuadrupleConv_U(13, 1)
        self.conv2310 = TripleConv_U(13, 1)
        self.conv2311 = QuadrupleConv_U(13, 1)
        self.conv2312 = TripleConv_U(13, 1)
        self.conv2313 = QuadrupleConv_U(13, 1)

        self.conv2401 = TripleConv_U(13, 1)
        self.conv2402 = QuadrupleConv_U(13, 1)
        self.conv2403 = TripleConv_U(13, 1)
        self.conv2404 = QuadrupleConv_U(13, 1)
        self.conv2405 = TripleConv_U(13, 1)
        self.conv2406 = QuadrupleConv_U(13, 1)
        self.conv2407 = TripleConv_U(13, 1)
        self.conv2408 = QuadrupleConv_U(13, 1)
        self.conv2409 = TripleConv_U(13, 1)
        self.conv2410 = QuadrupleConv_U(13, 1)
        self.conv2411 = TripleConv_U(13, 1)
        self.conv2412 = QuadrupleConv_U(13, 1)
        self.conv2413 = TripleConv_U(13, 1)

        self.conv2501 = QuadrupleConv_U(13, 1)
        self.conv2502 = TripleConv_U(13, 1)
        self.conv2503 = QuadrupleConv_U(13, 1)
        self.conv2504 = TripleConv_U(13, 1)
        self.conv2505 = QuadrupleConv_U(13, 1)
        self.conv2506 = TripleConv_U(13, 1)
        self.conv2507 = QuadrupleConv_U(13, 1)
        self.conv2508 = TripleConv_U(13, 1)
        self.conv2509 = QuadrupleConv_U(13, 1)
        self.conv2510 = TripleConv_U(13, 1)
        self.conv2511 = QuadrupleConv_U(13, 1)
        self.conv2512 = TripleConv_U(13, 1)
        self.conv2513 = QuadrupleConv_U(13, 1)

        self.fc2601 = nn.Linear(60, 60)
        self.fc2602 = nn.Linear(60, 60)
        self.fc2603 = nn.Linear(60, 60)
        self.fc2604 = nn.Linear(60, 60)
        self.fc2605 = nn.Linear(60, 60)
        self.fc2606 = nn.Linear(60, 60)
        self.fc2607 = nn.Linear(60, 60)
        self.fc2608 = nn.Linear(60, 60)
        self.fc2609 = nn.Linear(60, 60)
        self.fc2610 = nn.Linear(60, 60)
        self.fc2611 = nn.Linear(60, 60)
        self.fc2612 = nn.Linear(60, 60)
        self.fc2613 = nn.Linear(60, 60)
        self.fc2614 = nn.Linear(780, 390)

        self.fc2701 = nn.Linear(60, 60)
        self.fc2702 = nn.Linear(60, 60)
        self.fc2703 = nn.Linear(60, 60)
        self.fc2704 = nn.Linear(60, 60)
        self.fc2705 = nn.Linear(60, 60)
        self.fc2706 = nn.Linear(60, 60)
        self.fc2707 = nn.Linear(60, 60)
        self.fc2708 = nn.Linear(60, 60)
        self.fc2709 = nn.Linear(60, 60)
        self.fc2710 = nn.Linear(60, 60)
        self.fc2711 = nn.Linear(60, 60)
        self.fc2712 = nn.Linear(60, 60)
        self.fc2713 = nn.Linear(60, 60)
        self.fc2714 = nn.Linear(780, 390)
        self.fc2715 = nn.Linear(390, 240)

        self.fc2801 = nn.Linear(60, 60)
        self.fc2802 = nn.Linear(60, 60)
        self.fc2803 = nn.Linear(60, 60)
        self.fc2804 = nn.Linear(60, 60)
        self.fc2805 = nn.Linear(60, 60)
        self.fc2806 = nn.Linear(60, 60)
        self.fc2807 = nn.Linear(60, 60)
        self.fc2808 = nn.Linear(60, 60)
        self.fc2809 = nn.Linear(60, 60)
        self.fc2810 = nn.Linear(60, 60)
        self.fc2811 = nn.Linear(60, 60)
        self.fc2812 = nn.Linear(60, 60)
        self.fc2813 = nn.Linear(60, 60)
        self.fc2814 = nn.Linear(780, 390)
        self.fc2815 = nn.Linear(390, 240)
        self.fc2816 = nn.Linear(240, 120)

        self.fc2901 = nn.Linear(60, 60)
        self.fc2902 = nn.Linear(60, 60)
        self.fc2903 = nn.Linear(60, 60)
        self.fc2904 = nn.Linear(60, 60)
        self.fc2905 = nn.Linear(60, 60)
        self.fc2906 = nn.Linear(60, 60)
        self.fc2907 = nn.Linear(60, 60)
        self.fc2908 = nn.Linear(60, 60)
        self.fc2909 = nn.Linear(60, 60)
        self.fc2910 = nn.Linear(60, 60)
        self.fc2911 = nn.Linear(60, 60)
        self.fc2912 = nn.Linear(60, 60)
        self.fc2913 = nn.Linear(60, 60)
        self.fc2914 = nn.Linear(780, 390)
        self.fc2915 = nn.Linear(390, 240)
        self.fc2916 = nn.Linear(240, 120)
        self.fc2917 = nn.Linear(120, 60)

        self.fc3001 = nn.Linear(60, 60)
        self.fc3002 = nn.Linear(60, 60)
        self.fc3003 = nn.Linear(60, 60)
        self.fc3004 = nn.Linear(60, 60)
        self.fc3005 = nn.Linear(60, 60)
        self.fc3006 = nn.Linear(60, 60)
        self.fc3007 = nn.Linear(60, 60)
        self.fc3008 = nn.Linear(60, 60)
        self.fc3009 = nn.Linear(60, 60)
        self.fc3010 = nn.Linear(60, 60)
        self.fc3011 = nn.Linear(60, 60)
        self.fc3012 = nn.Linear(60, 60)
        self.fc3013 = nn.Linear(60, 60)
        self.fc3014 = nn.Linear(780, 390)
        self.fc3015 = nn.Linear(390, 240)
        self.fc3016 = nn.Linear(240, 120)
        self.fc3017 = nn.Linear(120, 60)
        self.fc3018 = nn.Linear(60, 60)

        self.fc3101 = nn.Linear(60, 60)
        self.fc3102 = nn.Linear(60, 60)
        self.fc3103 = nn.Linear(60, 60)
        self.fc3104 = nn.Linear(60, 60)
        self.fc3105 = nn.Linear(60, 60)
        self.fc3106 = nn.Linear(60, 60)
        self.fc3107 = nn.Linear(60, 60)
        self.fc3108 = nn.Linear(60, 60)
        self.fc3109 = nn.Linear(60, 60)
        self.fc3110 = nn.Linear(60, 60)
        self.fc3111 = nn.Linear(60, 60)
        self.fc3112 = nn.Linear(60, 60)
        self.fc3113 = nn.Linear(60, 60)
        self.fc3114 = nn.Linear(780, 390)
        self.fc3115 = nn.Linear(390, 240)
        self.fc3116 = nn.Linear(240, 120)
        self.fc3117 = nn.Linear(120, 60)
        self.fc3118 = nn.Linear(60, 60)
        self.fc3119 = nn.Linear(60, 60)

        self.fc3201 = nn.Linear(60, 60)
        self.fc3202 = nn.Linear(60, 60)
        self.fc3203 = nn.Linear(60, 60)
        self.fc3204 = nn.Linear(60, 60)
        self.fc3205 = nn.Linear(60, 60)
        self.fc3206 = nn.Linear(60, 60)
        self.fc3207 = nn.Linear(60, 60)
        self.fc3208 = nn.Linear(60, 60)
        self.fc3209 = nn.Linear(60, 60)
        self.fc3210 = nn.Linear(60, 60)
        self.fc3211 = nn.Linear(60, 60)
        self.fc3212 = nn.Linear(60, 60)
        self.fc3213 = nn.Linear(60, 60)
        self.fc3214 = nn.Linear(780, 390)
        self.fc3215 = nn.Linear(390, 240)
        self.fc3216 = nn.Linear(240, 120)
        self.fc3217 = nn.Linear(120, 60)
        self.fc3218 = nn.Linear(60, 60)
        self.fc3219 = nn.Linear(60, 60)
        self.fc3220 = nn.Linear(60, 60)

        self.fc3301 = nn.Linear(780, 390)
        self.fc3302 = nn.Linear(390, 120)
        self.fc3303 = nn.Linear(240, 120)
        self.fc3304 = nn.Linear(120, 60)
        self.fc3305 = nn.Linear(60, 60)
        self.fc3306 = nn.Linear(60, 60)
        self.fc3307 = nn.Linear(60, 60)
        self.fc3308 = nn.Linear(60, 60)

        self.fc3401 = nn.Linear(390, 120)
        self.fc3402 = nn.Linear(240, 120)
        self.fc3403 = nn.Linear(180, 60)
        self.fc3404 = nn.Linear(120, 60)

        self.fc3501 = nn.Linear(120, 60)
        self.fc3502 = nn.Linear(120, 60)
        self.fc3503 = nn.Linear(120, 60)

        self.fc36 = nn.Linear(180, 60)

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
        layer_in_1_1 = layer_in_1_1.view(layer_in_1_1.size(0), layer_in_1_1.size(1), -1, 10)

        layer_in_1_2 = train_well[:, 1, :]
        layer_in_1_2 = torch.cat([layer_in_1_2.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_2 = layer_in_1_2.view(layer_in_1_2.size(0), layer_in_1_2.size(1), -1, 10)

        layer_in_1_3 = train_well[:, 2, :]
        layer_in_1_3 = torch.cat([layer_in_1_3.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_3 = layer_in_1_3.view(layer_in_1_3.size(0), layer_in_1_3.size(1), -1, 10)

        layer_in_1_4 = train_well[:, 3, :]
        layer_in_1_4 = torch.cat([layer_in_1_4.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_4 = layer_in_1_4.view(layer_in_1_4.size(0), layer_in_1_4.size(1), -1, 10)

        layer_in_1_5 = train_well[:, 4, :]
        layer_in_1_5 = torch.cat([layer_in_1_5.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_5 = layer_in_1_5.view(layer_in_1_5.size(0), layer_in_1_5.size(1), -1, 10)

        layer_in_1_6 = seismic.view(1, 1, -1)
        layer_in_1_6 = layer_in_1_6.view(seismic.size(0), seismic.size(0), -1, 10)

        layer_in_1_7 = train_well.view(1, 5, -1)
        layer_in_1_7 = layer_in_1_7.view(train_well.size(0), train_well.size(1), -1, 10)

        layer_in_1_8 = torch.cat([layer_in_1_6, layer_in_1_7], dim=1)

        layer_in_1_9 = train_well[:, 0, :].view(train_well.size(0), train_well.size(0), -1, 10)
        layer_in_1_10 = train_well[:, 1, :].view(train_well.size(0), train_well.size(0), -1, 10)
        layer_in_1_11 = train_well[:, 2, :].view(train_well.size(0), train_well.size(0), -1, 10)
        layer_in_1_12 = train_well[:, 3, :].view(train_well.size(0), train_well.size(0), -1, 10)
        layer_in_1_13 = train_well[:, 4, :].view(train_well.size(0), train_well.size(0), -1, 10)

        # 第一层 卷积层
        output_layer1_1 = self.conv0101(layer_in_1_1)
        output_layer1 = output_layer1_1
        output_layer1_2 = self.conv0102(layer_in_1_2)
        output_layer1 = torch.cat([output_layer1, output_layer1_2], dim=1)
        output_layer1_3 = self.conv0103(layer_in_1_3)
        output_layer1 = torch.cat([output_layer1, output_layer1_3], dim=1)
        output_layer1_4 = self.conv0104(layer_in_1_4)
        output_layer1 = torch.cat([output_layer1, output_layer1_4], dim=1)
        output_layer1_5 = self.conv0105(layer_in_1_5)
        output_layer1 = torch.cat([output_layer1, output_layer1_5], dim=1)
        output_layer1_6 = self.conv0106(layer_in_1_6)
        output_layer1 = torch.cat([output_layer1, output_layer1_6], dim=1)
        output_layer1_7 = self.conv0107(layer_in_1_7)
        output_layer1 = torch.cat([output_layer1, output_layer1_7], dim=1)
        output_layer1_8 = self.conv0108(layer_in_1_8)
        output_layer1 = torch.cat([output_layer1, output_layer1_8], dim=1)
        output_layer1_9 = self.conv0109(layer_in_1_9)
        output_layer1 = torch.cat([output_layer1, output_layer1_9], dim=1)
        output_layer1_10 = self.conv0110(layer_in_1_10)
        output_layer1 = torch.cat([output_layer1, output_layer1_10], dim=1)
        output_layer1_11 = self.conv0111(layer_in_1_11)
        output_layer1 = torch.cat([output_layer1, output_layer1_11], dim=1)
        output_layer1_12 = self.conv0112(layer_in_1_12)
        output_layer1 = torch.cat([output_layer1, output_layer1_12], dim=1)
        output_layer1_13 = self.conv0113(layer_in_1_13)
        output_layer1 = torch.cat([output_layer1, output_layer1_13], dim=1)

        # 第二层 卷积层
        output_layer2_1 = self.conv0201(output_layer1)
        output_layer2 = output_layer2_1
        output_layer2_2 = self.conv0202(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_2], dim=1)
        output_layer2_3 = self.conv0203(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_3], dim=1)
        output_layer2_4 = self.conv0204(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_4], dim=1)
        output_layer2_5 = self.conv0205(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_5], dim=1)
        output_layer2_6 = self.conv0206(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_6], dim=1)
        output_layer2_7 = self.conv0207(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_7], dim=1)
        output_layer2_8 = self.conv0208(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_8], dim=1)
        output_layer2_9 = self.conv0209(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_9], dim=1)
        output_layer2_10 = self.conv0210(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_10], dim=1)
        output_layer2_11 = self.conv0211(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_11], dim=1)
        output_layer2_12 = self.conv0212(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_12], dim=1)
        output_layer2_13 = self.conv0213(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_13], dim=1)

        # 第三层 卷积层
        output_layer3_1 = self.conv0301(output_layer2)
        output_layer3 = output_layer3_1
        output_layer3_2 = self.conv0302(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_2], dim=1)
        output_layer3_3 = self.conv0303(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_3], dim=1)
        output_layer3_4 = self.conv0304(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_4], dim=1)
        output_layer3_5 = self.conv0305(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_5], dim=1)
        output_layer3_6 = self.conv0306(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_6], dim=1)
        output_layer3_7 = self.conv0307(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_7], dim=1)
        output_layer3_8 = self.conv0308(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_8], dim=1)
        output_layer3_9 = self.conv0309(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_9], dim=1)
        output_layer3_10 = self.conv0310(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_10], dim=1)
        output_layer3_11 = self.conv0311(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_11], dim=1)
        output_layer3_12 = self.conv0312(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_12], dim=1)
        output_layer3_13 = self.conv0313(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_13], dim=1)

        # 第四层 卷积层
        output_layer4_1 = self.conv0401(output_layer3)
        output_layer4 = output_layer4_1
        output_layer4_2 = self.conv0402(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_2], dim=1)
        output_layer4_3 = self.conv0403(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_3], dim=1)
        output_layer4_4 = self.conv0404(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_4], dim=1)
        output_layer4_5 = self.conv0405(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_5], dim=1)
        output_layer4_6 = self.conv0406(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_6], dim=1)
        output_layer4_7 = self.conv0407(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_7], dim=1)
        output_layer4_8 = self.conv0408(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_8], dim=1)
        output_layer4_9 = self.conv0409(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_9], dim=1)
        output_layer4_10 = self.conv0410(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_10], dim=1)
        output_layer4_11 = self.conv0411(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_11], dim=1)
        output_layer4_12 = self.conv0412(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_12], dim=1)
        output_layer4_13 = self.conv0413(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_13], dim=1)

        # 第五层 卷积层
        output_layer5_1 = self.conv0501(output_layer4)
        output_layer5 = output_layer5_1
        output_layer5_2 = self.conv0502(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_2], dim=1)
        output_layer5_3 = self.conv0503(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_3], dim=1)
        output_layer5_4 = self.conv0504(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_4], dim=1)
        output_layer5_5 = self.conv0505(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_5], dim=1)
        output_layer5_6 = self.conv0506(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_6], dim=1)
        output_layer5_7 = self.conv0507(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_7], dim=1)
        output_layer5_8 = self.conv0508(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_8], dim=1)
        output_layer5_9 = self.conv0509(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_9], dim=1)
        output_layer5_10 = self.conv0510(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_10], dim=1)
        output_layer5_11 = self.conv0511(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_11], dim=1)
        output_layer5_12 = self.conv0512(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_12], dim=1)
        output_layer5_13 = self.conv0513(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_13], dim=1)

        # 第六层 卷积层
        output_layer6_1 = self.conv0601(output_layer5)
        output_layer6 = output_layer6_1
        output_layer6_2 = self.conv0602(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_2], dim=1)
        output_layer6_3 = self.conv0603(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_3], dim=1)
        output_layer6_4 = self.conv0604(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_4], dim=1)
        output_layer6_5 = self.conv0605(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_5], dim=1)
        output_layer6_6 = self.conv0606(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_6], dim=1)
        output_layer6_7 = self.conv0607(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_7], dim=1)
        output_layer6_8 = self.conv0608(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_8], dim=1)
        output_layer6_9 = self.conv0609(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_9], dim=1)
        output_layer6_10 = self.conv0610(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_10], dim=1)
        output_layer6_11 = self.conv0611(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_11], dim=1)
        output_layer6_12 = self.conv0612(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_12], dim=1)
        output_layer6_13 = self.conv0613(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_13], dim=1)

        # 第七层 卷积层
        output_layer7_1 = self.conv0701(output_layer6)
        output_layer7 = output_layer7_1
        output_layer7_2 = self.conv0702(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_2], dim=1)
        output_layer7_3 = self.conv0703(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_3], dim=1)
        output_layer7_4 = self.conv0704(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_4], dim=1)
        output_layer7_5 = self.conv0705(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_5], dim=1)
        output_layer7_6 = self.conv0706(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_6], dim=1)
        output_layer7_7 = self.conv0707(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_7], dim=1)
        output_layer7_8 = self.conv0708(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_8], dim=1)
        output_layer7_9 = self.conv0709(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_9], dim=1)
        output_layer7_10 = self.conv0710(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_10], dim=1)
        output_layer7_11 = self.conv0711(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_11], dim=1)
        output_layer7_12 = self.conv0712(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_12], dim=1)
        output_layer7_13 = self.conv0713(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_13], dim=1)

        # 第八层 卷积层
        output_layer8_1 = self.conv0801(output_layer7)
        output_layer8 = output_layer8_1
        output_layer8_2 = self.conv0802(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_2], dim=1)
        output_layer8_3 = self.conv0803(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_3], dim=1)
        output_layer8_4 = self.conv0804(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_4], dim=1)
        output_layer8_5 = self.conv0805(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_5], dim=1)
        output_layer8_6 = self.conv0806(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_6], dim=1)
        output_layer8_7 = self.conv0807(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_7], dim=1)
        output_layer8_8 = self.conv0808(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_8], dim=1)
        output_layer8_9 = self.conv0809(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_9], dim=1)
        output_layer8_10 = self.conv0810(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_10], dim=1)
        output_layer8_11 = self.conv0811(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_11], dim=1)
        output_layer8_12 = self.conv0812(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_12], dim=1)
        output_layer8_13 = self.conv0813(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_13], dim=1)

        # 第九层 卷积层
        output_layer9_1 = self.conv0901(output_layer8)
        output_layer9 = output_layer9_1
        output_layer9_2 = self.conv0902(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_2], dim=1)
        output_layer9_3 = self.conv0903(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_3], dim=1)
        output_layer9_4 = self.conv0904(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_4], dim=1)
        output_layer9_5 = self.conv0905(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_5], dim=1)
        output_layer9_6 = self.conv0906(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_6], dim=1)
        output_layer9_7 = self.conv0907(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_7], dim=1)
        output_layer9_8 = self.conv0908(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_8], dim=1)
        output_layer9_9 = self.conv0909(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_9], dim=1)
        output_layer9_10 = self.conv0910(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_10], dim=1)
        output_layer9_11 = self.conv0911(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_11], dim=1)
        output_layer9_12 = self.conv0912(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_12], dim=1)
        output_layer9_13 = self.conv0913(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_13], dim=1)

        # 第十层 卷积层
        output_layer10_1 = self.conv1001(output_layer9)
        output_layer10 = output_layer10_1
        output_layer10_2 = self.conv1002(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_2], dim=1)
        output_layer10_3 = self.conv1003(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_3], dim=1)
        output_layer10_4 = self.conv1004(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_4], dim=1)
        output_layer10_5 = self.conv1005(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_5], dim=1)
        output_layer10_6 = self.conv1006(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_6], dim=1)
        output_layer10_7 = self.conv1007(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_7], dim=1)
        output_layer10_8 = self.conv1008(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_8], dim=1)
        output_layer10_9 = self.conv1009(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_9], dim=1)
        output_layer10_10 = self.conv1010(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_10], dim=1)
        output_layer10_11 = self.conv1011(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_11], dim=1)
        output_layer10_12 = self.conv1012(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_12], dim=1)
        output_layer10_13 = self.conv1013(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_13], dim=1)

        # 第十一层 卷积层
        output_layer11_1 = self.conv1101(output_layer10)
        output_layer11 = output_layer11_1
        output_layer11_2 = self.conv1102(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_2], dim=1)
        output_layer11_3 = self.conv1103(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_3], dim=1)
        output_layer11_4 = self.conv1104(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_4], dim=1)
        output_layer11_5 = self.conv1105(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_5], dim=1)
        output_layer11_6 = self.conv1106(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_6], dim=1)
        output_layer11_7 = self.conv1107(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_7], dim=1)
        output_layer11_8 = self.conv1108(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_8], dim=1)
        output_layer11_9 = self.conv1109(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_9], dim=1)
        output_layer11_10 = self.conv1110(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_10], dim=1)
        output_layer11_11 = self.conv1111(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_11], dim=1)
        output_layer11_12 = self.conv1112(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_12], dim=1)
        output_layer11_13 = self.conv1113(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_13], dim=1)

        # 第十二层 卷积层
        output_layer12_1 = self.conv1201(output_layer11)
        output_layer12 = output_layer12_1
        output_layer12_2 = self.conv1202(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_2], dim=1)
        output_layer12_3 = self.conv1203(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_3], dim=1)
        output_layer12_4 = self.conv1204(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_4], dim=1)
        output_layer12_5 = self.conv1205(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_5], dim=1)
        output_layer12_6 = self.conv1206(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_6], dim=1)
        output_layer12_7 = self.conv1207(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_7], dim=1)
        output_layer12_8 = self.conv1208(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_8], dim=1)
        output_layer12_9 = self.conv1209(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_9], dim=1)
        output_layer12_10 = self.conv1210(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_10], dim=1)
        output_layer12_11 = self.conv1211(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_11], dim=1)
        output_layer12_12 = self.conv1212(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_12], dim=1)
        output_layer12_13 = self.conv1213(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_13], dim=1)

        # 第十三层 卷积层
        output_layer13_1 = self.conv1301(output_layer12)
        output_layer13 = output_layer13_1
        output_layer13_2 = self.conv1302(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_2], dim=1)
        output_layer13_3 = self.conv1303(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_3], dim=1)
        output_layer13_4 = self.conv1304(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_4], dim=1)
        output_layer13_5 = self.conv1305(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_5], dim=1)
        output_layer13_6 = self.conv1306(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_6], dim=1)
        output_layer13_7 = self.conv1307(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_7], dim=1)
        output_layer13_8 = self.conv1308(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_8], dim=1)
        output_layer13_9 = self.conv1309(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_9], dim=1)
        output_layer13_10 = self.conv1310(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_10], dim=1)
        output_layer13_11 = self.conv1311(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_11], dim=1)
        output_layer13_12 = self.conv1312(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_12], dim=1)
        output_layer13_13 = self.conv1313(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_13], dim=1)

        # 第十四层 卷积层
        output_layer14_1 = self.conv1401(output_layer13)
        output_layer14 = output_layer14_1
        output_layer14_2 = self.conv1402(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_2], dim=1)
        output_layer14_3 = self.conv1403(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_3], dim=1)
        output_layer14_4 = self.conv1404(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_4], dim=1)
        output_layer14_5 = self.conv1405(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_5], dim=1)
        output_layer14_6 = self.conv1406(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_6], dim=1)
        output_layer14_7 = self.conv1407(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_7], dim=1)
        output_layer14_8 = self.conv1408(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_8], dim=1)
        output_layer14_9 = self.conv1409(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_9], dim=1)
        output_layer14_10 = self.conv1410(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_10], dim=1)
        output_layer14_11 = self.conv1411(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_11], dim=1)
        output_layer14_12 = self.conv1412(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_12], dim=1)
        output_layer14_13 = self.conv1413(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_13], dim=1)

        # 第十五层 卷积层
        output_layer15_1 = self.conv1501(output_layer14)
        output_layer15 = output_layer15_1
        output_layer15_2 = self.conv1502(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_2], dim=1)
        output_layer15_3 = self.conv1503(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_3], dim=1)
        output_layer15_4 = self.conv1504(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_4], dim=1)
        output_layer15_5 = self.conv1505(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_5], dim=1)
        output_layer15_6 = self.conv1506(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_6], dim=1)
        output_layer15_7 = self.conv1507(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_7], dim=1)
        output_layer15_8 = self.conv1508(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_8], dim=1)
        output_layer15_9 = self.conv1509(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_9], dim=1)
        output_layer15_10 = self.conv1510(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_10], dim=1)
        output_layer15_11 = self.conv1511(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_11], dim=1)
        output_layer15_12 = self.conv1512(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_12], dim=1)
        output_layer15_13 = self.conv1513(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_13], dim=1)

        # 第十六层 卷积层
        output_layer16_1 = self.conv1601(output_layer15)
        output_layer16 = output_layer16_1
        output_layer16_2 = self.conv1602(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_2], dim=1)
        output_layer16_3 = self.conv1603(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_3], dim=1)
        output_layer16_4 = self.conv1604(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_4], dim=1)
        output_layer16_5 = self.conv1605(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_5], dim=1)
        output_layer16_6 = self.conv1606(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_6], dim=1)
        output_layer16_7 = self.conv1607(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_7], dim=1)
        output_layer16_8 = self.conv1608(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_8], dim=1)
        output_layer16_9 = self.conv1609(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_9], dim=1)
        output_layer16_10 = self.conv1610(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_10], dim=1)
        output_layer16_11 = self.conv1611(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_11], dim=1)
        output_layer16_12 = self.conv1612(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_12], dim=1)
        output_layer16_13 = self.conv1613(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_13], dim=1)

        # 第十七层 卷积层
        output_layer17_1 = self.conv1701(output_layer16)
        output_layer17 = output_layer17_1
        output_layer17_2 = self.conv1702(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_2], dim=1)
        output_layer17_3 = self.conv1703(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_3], dim=1)
        output_layer17_4 = self.conv1704(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_4], dim=1)
        output_layer17_5 = self.conv1705(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_5], dim=1)
        output_layer17_6 = self.conv1706(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_6], dim=1)
        output_layer17_7 = self.conv1707(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_7], dim=1)
        output_layer17_8 = self.conv1708(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_8], dim=1)
        output_layer17_9 = self.conv1709(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_9], dim=1)
        output_layer17_10 = self.conv1710(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_10], dim=1)
        output_layer17_11 = self.conv1711(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_11], dim=1)
        output_layer17_12 = self.conv1712(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_12], dim=1)
        output_layer17_13 = self.conv1713(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_13], dim=1)

        # 第十八层 卷积层
        output_layer18_1 = self.conv1801(output_layer17)
        output_layer18 = output_layer18_1
        output_layer18_2 = self.conv1802(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_2], dim=1)
        output_layer18_3 = self.conv1803(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_3], dim=1)
        output_layer18_4 = self.conv1804(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_4], dim=1)
        output_layer18_5 = self.conv1805(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_5], dim=1)
        output_layer18_6 = self.conv1806(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_6], dim=1)
        output_layer18_7 = self.conv1807(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_7], dim=1)
        output_layer18_8 = self.conv1808(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_8], dim=1)
        output_layer18_9 = self.conv1809(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_9], dim=1)
        output_layer18_10 = self.conv1810(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_10], dim=1)
        output_layer18_11 = self.conv1811(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_11], dim=1)
        output_layer18_12 = self.conv1812(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_12], dim=1)
        output_layer18_13 = self.conv1813(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_13], dim=1)

        # 第十九层 卷积层
        output_layer19_1 = self.conv1901(output_layer18)
        output_layer19 = output_layer19_1
        output_layer19_2 = self.conv1902(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_2], dim=1)
        output_layer19_3 = self.conv1903(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_3], dim=1)
        output_layer19_4 = self.conv1904(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_4], dim=1)
        output_layer19_5 = self.conv1905(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_5], dim=1)
        output_layer19_6 = self.conv1906(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_6], dim=1)
        output_layer19_7 = self.conv1907(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_7], dim=1)
        output_layer19_8 = self.conv1908(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_8], dim=1)
        output_layer19_9 = self.conv1909(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_9], dim=1)
        output_layer19_10 = self.conv1910(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_10], dim=1)
        output_layer19_11 = self.conv1911(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_11], dim=1)
        output_layer19_12 = self.conv1912(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_12], dim=1)
        output_layer19_13 = self.conv1913(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_13], dim=1)

        # 第二十层 卷积层
        output_layer20_1 = self.conv2001(output_layer19)
        output_layer20 = output_layer20_1
        output_layer20_2 = self.conv2002(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_2], dim=1)
        output_layer20_3 = self.conv2003(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_3], dim=1)
        output_layer20_4 = self.conv2004(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_4], dim=1)
        output_layer20_5 = self.conv2005(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_5], dim=1)
        output_layer20_6 = self.conv2006(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_6], dim=1)
        output_layer20_7 = self.conv2007(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_7], dim=1)
        output_layer20_8 = self.conv2008(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_8], dim=1)
        output_layer20_9 = self.conv2009(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_9], dim=1)
        output_layer20_10 = self.conv2010(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_10], dim=1)
        output_layer20_11 = self.conv2011(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_11], dim=1)
        output_layer20_12 = self.conv2012(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_12], dim=1)
        output_layer20_13 = self.conv2013(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_13], dim=1)

        # 第二十一层 卷积层
        output_layer21_1 = self.conv2101(output_layer20)
        output_layer21 = output_layer21_1
        output_layer21_2 = self.conv2102(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_2], dim=1)
        output_layer21_3 = self.conv2103(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_3], dim=1)
        output_layer21_4 = self.conv2104(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_4], dim=1)
        output_layer21_5 = self.conv2105(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_5], dim=1)
        output_layer21_6 = self.conv2106(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_6], dim=1)
        output_layer21_7 = self.conv2107(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_7], dim=1)
        output_layer21_8 = self.conv2108(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_8], dim=1)
        output_layer21_9 = self.conv2109(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_9], dim=1)
        output_layer21_10 = self.conv2110(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_10], dim=1)
        output_layer21_11 = self.conv2111(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_11], dim=1)
        output_layer21_12 = self.conv2112(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_12], dim=1)
        output_layer21_13 = self.conv2113(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_13], dim=1)

        # 第二十二层 卷积层
        output_layer22_1 = self.conv2201(output_layer21)
        output_layer22 = output_layer22_1
        output_layer22_2 = self.conv2202(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_2], dim=1)
        output_layer22_3 = self.conv2203(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_3], dim=1)
        output_layer22_4 = self.conv2204(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_4], dim=1)
        output_layer22_5 = self.conv2205(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_5], dim=1)
        output_layer22_6 = self.conv2206(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_6], dim=1)
        output_layer22_7 = self.conv2207(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_7], dim=1)
        output_layer22_8 = self.conv2208(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_8], dim=1)
        output_layer22_9 = self.conv2209(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_9], dim=1)
        output_layer22_10 = self.conv2210(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_10], dim=1)
        output_layer22_11 = self.conv2211(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_11], dim=1)
        output_layer22_12 = self.conv2212(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_12], dim=1)
        output_layer22_13 = self.conv2213(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_13], dim=1)

        # 第二十三层 卷积层
        output_layer23_1 = self.conv2301(output_layer22)
        output_layer23 = output_layer23_1
        output_layer23_2 = self.conv2302(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_2], dim=1)
        output_layer23_3 = self.conv2303(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_3], dim=1)
        output_layer23_4 = self.conv2304(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_4], dim=1)
        output_layer23_5 = self.conv2305(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_5], dim=1)
        output_layer23_6 = self.conv2306(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_6], dim=1)
        output_layer23_7 = self.conv2307(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_7], dim=1)
        output_layer23_8 = self.conv2308(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_8], dim=1)
        output_layer23_9 = self.conv2309(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_9], dim=1)
        output_layer23_10 = self.conv2310(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_10], dim=1)
        output_layer23_11 = self.conv2311(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_11], dim=1)
        output_layer23_12 = self.conv2312(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_12], dim=1)
        output_layer23_13 = self.conv2313(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_13], dim=1)

        # 第二十四层 卷积层
        output_layer24_1 = self.conv2401(output_layer23)
        output_layer24 = output_layer24_1
        output_layer24_2 = self.conv2402(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_2], dim=1)
        output_layer24_3 = self.conv2403(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_3], dim=1)
        output_layer24_4 = self.conv2404(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_4], dim=1)
        output_layer24_5 = self.conv2405(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_5], dim=1)
        output_layer24_6 = self.conv2406(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_6], dim=1)
        output_layer24_7 = self.conv2407(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_7], dim=1)
        output_layer24_8 = self.conv2408(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_8], dim=1)
        output_layer24_9 = self.conv2409(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_9], dim=1)
        output_layer24_10 = self.conv2410(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_10], dim=1)
        output_layer24_11 = self.conv2411(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_11], dim=1)
        output_layer24_12 = self.conv2412(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_12], dim=1)
        output_layer24_13 = self.conv2413(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_13], dim=1)

        # 第二十五层 卷积层
        output_layer25_1 = self.conv2501(output_layer24)
        output_layer25 = output_layer25_1
        output_layer25_2 = self.conv2502(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_2], dim=1)
        output_layer25_3 = self.conv2503(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_3], dim=1)
        output_layer25_4 = self.conv2504(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_4], dim=1)
        output_layer25_5 = self.conv2505(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_5], dim=1)
        output_layer25_6 = self.conv2506(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_6], dim=1)
        output_layer25_7 = self.conv2507(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_7], dim=1)
        output_layer25_8 = self.conv2508(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_8], dim=1)
        output_layer25_9 = self.conv2509(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_9], dim=1)
        output_layer25_10 = self.conv2510(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_10], dim=1)
        output_layer25_11 = self.conv2511(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_11], dim=1)
        output_layer25_12 = self.conv2512(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_12], dim=1)
        output_layer25_13 = self.conv2513(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_13], dim=1)

        # 第二十六层 LSTM层 特征映射
        output_layer26_1 = output_layer25_1.view(1, -1)
        output_layer26_1 = self.fc2601(output_layer26_1)
        output_layer26 = output_layer26_1

        output_layer26_2 = output_layer25_2.view(1, -1)
        output_layer26_2 = self.fc2602(output_layer26_2)
        output_layer26 = torch.cat([output_layer26, output_layer26_2], dim=1)

        output_layer26_3 = output_layer25_3.view(1, -1)
        output_layer26_3 = self.fc2603(output_layer26_3)
        output_layer26 = torch.cat([output_layer26, output_layer26_3], dim=1)

        output_layer26_4 = output_layer25_4.view(1, -1)
        output_layer26_4 = self.fc2604(output_layer26_4)
        output_layer26 = torch.cat([output_layer26, output_layer26_4], dim=1)

        output_layer26_5 = output_layer25_5.view(1, -1)
        output_layer26_5 = self.fc2605(output_layer26_5)
        output_layer26 = torch.cat([output_layer26, output_layer26_5], dim=1)

        output_layer26_6 = output_layer25_6.view(1, -1)
        output_layer26_6 = self.fc2606(output_layer26_6)
        output_layer26 = torch.cat([output_layer26, output_layer26_6], dim=1)

        output_layer26_7 = output_layer25_7.view(1, -1)
        output_layer26_7 = self.fc2607(output_layer26_7)
        output_layer26 = torch.cat([output_layer26, output_layer26_7], dim=1)

        output_layer26_8 = output_layer25_8.view(1, -1)
        output_layer26_8 = self.fc2608(output_layer26_8)
        output_layer26 = torch.cat([output_layer26, output_layer26_8], dim=1)

        output_layer26_9 = output_layer25_9.view(1, -1)
        output_layer26_9 = self.fc2609(output_layer26_9)
        output_layer26 = torch.cat([output_layer26, output_layer26_9], dim=1)

        output_layer26_10 = output_layer25_10.view(1, -1)
        output_layer26_10 = self.fc2610(output_layer26_10)
        output_layer26 = torch.cat([output_layer26, output_layer26_10], dim=1)

        output_layer26_11 = output_layer25_11.view(1, -1)
        output_layer26_11 = self.fc2611(output_layer26_11)
        output_layer26 = torch.cat([output_layer26, output_layer26_11], dim=1)

        output_layer26_12 = output_layer25_12.view(1, -1)
        output_layer26_12 = self.fc2612(output_layer26_12)
        output_layer26 = torch.cat([output_layer26, output_layer26_12], dim=1)

        output_layer26_13 = output_layer25_13.view(1, -1)
        output_layer26_13 = self.fc2613(output_layer26_13)
        output_layer26 = torch.cat([output_layer26, output_layer26_13], dim=1)

        output_layer26_14 = output_layer25.view(1, -1)
        output_layer26_14 = self.fc2614(output_layer26_14)

        # 第二十七层 LSTM层 特征映射
        output_layer27_1 = self.fc2701(output_layer26_1)
        output_layer27 = output_layer27_1
        output_layer27_2 = self.fc2702(output_layer26_2)
        output_layer27 = torch.cat([output_layer27, output_layer27_2], dim=1)
        output_layer27_3 = self.fc2703(output_layer26_3)
        output_layer27 = torch.cat([output_layer27, output_layer27_3], dim=1)
        output_layer27_4 = self.fc2704(output_layer26_4)
        output_layer27 = torch.cat([output_layer27, output_layer27_4], dim=1)
        output_layer27_5 = self.fc2705(output_layer26_5)
        output_layer27 = torch.cat([output_layer27, output_layer27_5], dim=1)
        output_layer27_6 = self.fc2706(output_layer26_6)
        output_layer27 = torch.cat([output_layer27, output_layer27_6], dim=1)
        output_layer27_7 = self.fc2707(output_layer26_7)
        output_layer27 = torch.cat([output_layer27, output_layer27_7], dim=1)
        output_layer27_8 = self.fc2708(output_layer26_8)
        output_layer27 = torch.cat([output_layer27, output_layer27_8], dim=1)
        output_layer27_9 = self.fc2709(output_layer26_9)
        output_layer27 = torch.cat([output_layer27, output_layer27_9], dim=1)
        output_layer27_10 = self.fc2710(output_layer26_10)
        output_layer27 = torch.cat([output_layer27, output_layer27_10], dim=1)
        output_layer27_11 = self.fc2711(output_layer26_11)
        output_layer27 = torch.cat([output_layer27, output_layer27_11], dim=1)
        output_layer27_12 = self.fc2712(output_layer26_12)
        output_layer27 = torch.cat([output_layer27, output_layer27_12], dim=1)
        output_layer27_13 = self.fc2713(output_layer26_13)
        output_layer27 = torch.cat([output_layer27, output_layer27_13], dim=1)

        output_layer27_14 = self.fc2714(output_layer26)
        output_layer27_15 = self.fc2715(output_layer26_14)

        # 第二十八层 LSTM层 特征映射
        output_layer28_1 = self.fc2801(output_layer27_1)
        output_layer28 = output_layer28_1
        output_layer28_2 = self.fc2802(output_layer27_2)
        output_layer28 = torch.cat([output_layer28, output_layer28_2], dim=1)
        output_layer28_3 = self.fc2803(output_layer27_3)
        output_layer28 = torch.cat([output_layer28, output_layer28_3], dim=1)
        output_layer28_4 = self.fc2804(output_layer27_4)
        output_layer28 = torch.cat([output_layer28, output_layer28_4], dim=1)
        output_layer28_5 = self.fc2805(output_layer27_5)
        output_layer28 = torch.cat([output_layer28, output_layer28_5], dim=1)
        output_layer28_6 = self.fc2806(output_layer27_6)
        output_layer28 = torch.cat([output_layer28, output_layer28_6], dim=1)
        output_layer28_7 = self.fc2807(output_layer27_7)
        output_layer28 = torch.cat([output_layer28, output_layer28_7], dim=1)
        output_layer28_8 = self.fc2808(output_layer27_8)
        output_layer28 = torch.cat([output_layer28, output_layer28_8], dim=1)
        output_layer28_9 = self.fc2809(output_layer27_9)
        output_layer28 = torch.cat([output_layer28, output_layer28_9], dim=1)
        output_layer28_10 = self.fc2810(output_layer27_10)
        output_layer28 = torch.cat([output_layer28, output_layer28_10], dim=1)
        output_layer28_11 = self.fc2811(output_layer27_11)
        output_layer28 = torch.cat([output_layer28, output_layer28_11], dim=1)
        output_layer28_12 = self.fc2812(output_layer27_12)
        output_layer28 = torch.cat([output_layer28, output_layer28_12], dim=1)
        output_layer28_13 = self.fc2813(output_layer27_13)
        output_layer28 = torch.cat([output_layer28, output_layer28_13], dim=1)

        output_layer28_14 = self.fc2814(output_layer27)
        output_layer28_15 = self.fc2815(output_layer27_14)
        output_layer28_16 = self.fc2816(output_layer27_15)

        # 第二十九层 LSTM层 特征映射
        output_layer29_1 = self.fc2901(output_layer28_1)
        output_layer29 = output_layer29_1
        output_layer29_2 = self.fc2902(output_layer28_2)
        output_layer29 = torch.cat([output_layer29, output_layer29_2], dim=1)
        output_layer29_3 = self.fc2903(output_layer28_3)
        output_layer29 = torch.cat([output_layer29, output_layer29_3], dim=1)
        output_layer29_4 = self.fc2904(output_layer28_4)
        output_layer29 = torch.cat([output_layer29, output_layer29_4], dim=1)
        output_layer29_5 = self.fc2905(output_layer28_5)
        output_layer29 = torch.cat([output_layer29, output_layer29_5], dim=1)
        output_layer29_6 = self.fc2906(output_layer28_6)
        output_layer29 = torch.cat([output_layer29, output_layer29_6], dim=1)
        output_layer29_7 = self.fc2907(output_layer28_7)
        output_layer29 = torch.cat([output_layer29, output_layer29_7], dim=1)
        output_layer29_8 = self.fc2908(output_layer28_8)
        output_layer29 = torch.cat([output_layer29, output_layer29_8], dim=1)
        output_layer29_9 = self.fc2909(output_layer28_9)
        output_layer29 = torch.cat([output_layer29, output_layer29_9], dim=1)
        output_layer29_10 = self.fc2910(output_layer28_10)
        output_layer29 = torch.cat([output_layer29, output_layer29_10], dim=1)
        output_layer29_11 = self.fc2911(output_layer28_11)
        output_layer29 = torch.cat([output_layer29, output_layer29_11], dim=1)
        output_layer29_12 = self.fc2912(output_layer28_12)
        output_layer29 = torch.cat([output_layer29, output_layer29_12], dim=1)
        output_layer29_13 = self.fc2913(output_layer28_13)
        output_layer29 = torch.cat([output_layer29, output_layer29_13], dim=1)

        output_layer29_14 = self.fc2914(output_layer28)
        output_layer29_15 = self.fc2915(output_layer28_14)
        output_layer29_16 = self.fc2916(output_layer28_15)
        output_layer29_17 = self.fc2917(output_layer28_16)

        # 第三十层 LSTM层 特征映射
        output_layer30_1 = self.fc3001(output_layer29_1)
        output_layer30 = output_layer30_1
        output_layer30_2 = self.fc3002(output_layer29_2)
        output_layer30 = torch.cat([output_layer30, output_layer30_2], dim=1)
        output_layer30_3 = self.fc3003(output_layer29_3)
        output_layer30 = torch.cat([output_layer30, output_layer30_3], dim=1)
        output_layer30_4 = self.fc3004(output_layer29_4)
        output_layer30 = torch.cat([output_layer30, output_layer30_4], dim=1)
        output_layer30_5 = self.fc3005(output_layer29_5)
        output_layer30 = torch.cat([output_layer30, output_layer30_5], dim=1)
        output_layer30_6 = self.fc3006(output_layer29_6)
        output_layer30 = torch.cat([output_layer30, output_layer30_6], dim=1)
        output_layer30_7 = self.fc3007(output_layer29_7)
        output_layer30 = torch.cat([output_layer30, output_layer30_7], dim=1)
        output_layer30_8 = self.fc3008(output_layer29_8)
        output_layer30 = torch.cat([output_layer30, output_layer30_8], dim=1)
        output_layer30_9 = self.fc3009(output_layer29_9)
        output_layer30 = torch.cat([output_layer30, output_layer30_9], dim=1)
        output_layer30_10 = self.fc3010(output_layer29_10)
        output_layer30 = torch.cat([output_layer30, output_layer30_10], dim=1)
        output_layer30_11 = self.fc3011(output_layer29_11)
        output_layer30 = torch.cat([output_layer30, output_layer30_11], dim=1)
        output_layer30_12 = self.fc3012(output_layer29_12)
        output_layer30 = torch.cat([output_layer30, output_layer30_12], dim=1)
        output_layer30_13 = self.fc3013(output_layer29_13)
        output_layer30 = torch.cat([output_layer30, output_layer30_13], dim=1)

        output_layer30_14 = self.fc3014(output_layer29)
        output_layer30_15 = self.fc3015(output_layer29_14)
        output_layer30_16 = self.fc3016(output_layer29_15)
        output_layer30_17 = self.fc3017(output_layer29_16)
        output_layer30_18 = self.fc3018(output_layer29_17)

        # 第三十一层 线性层 特征映射
        output_layer31_1 = output_layer30_1.view(1, -1)
        output_layer31_1 = self.fc3101(output_layer31_1)
        output_layer31 = output_layer31_1

        output_layer31_2 = output_layer30_2.view(1, -1)
        output_layer31_2 = self.fc3102(output_layer31_2)
        output_layer31 = torch.cat([output_layer31, output_layer31_2], dim=1)

        output_layer31_3 = output_layer30_3.view(1, -1)
        output_layer31_3 = self.fc3103(output_layer31_3)
        output_layer31 = torch.cat([output_layer31, output_layer31_3], dim=1)

        output_layer31_4 = output_layer30_4.view(1, -1)
        output_layer31_4 = self.fc3104(output_layer31_4)
        output_layer31 = torch.cat([output_layer31, output_layer31_4], dim=1)

        output_layer31_5 = output_layer30_5.view(1, -1)
        output_layer31_5 = self.fc3105(output_layer31_5)
        output_layer31 = torch.cat([output_layer31, output_layer31_5], dim=1)

        output_layer31_6 = output_layer30_6.view(1, -1)
        output_layer31_6 = self.fc3106(output_layer31_6)
        output_layer31 = torch.cat([output_layer31, output_layer31_6], dim=1)

        output_layer31_7 = output_layer30_7.view(1, -1)
        output_layer31_7 = self.fc3107(output_layer31_7)
        output_layer31 = torch.cat([output_layer31, output_layer31_7], dim=1)

        output_layer31_8 = output_layer30_8.view(1, -1)
        output_layer31_8 = self.fc3108(output_layer31_8)
        output_layer31 = torch.cat([output_layer31, output_layer31_8], dim=1)

        output_layer31_9 = output_layer30_9.view(1, -1)
        output_layer31_9 = self.fc3109(output_layer31_9)
        output_layer31 = torch.cat([output_layer31, output_layer31_9], dim=1)

        output_layer31_10 = output_layer30_10.view(1, -1)
        output_layer31_10 = self.fc3110(output_layer31_10)
        output_layer31 = torch.cat([output_layer31, output_layer31_10], dim=1)

        output_layer31_11 = output_layer30_11.view(1, -1)
        output_layer31_11 = self.fc3111(output_layer31_11)
        output_layer31 = torch.cat([output_layer31, output_layer31_11], dim=1)

        output_layer31_12 = output_layer30_12.view(1, -1)
        output_layer31_12 = self.fc3112(output_layer31_12)
        output_layer31 = torch.cat([output_layer31, output_layer31_12], dim=1)

        output_layer31_13 = output_layer30_13.view(1, -1)
        output_layer31_13 = self.fc3113(output_layer31_13)
        output_layer31 = torch.cat([output_layer31, output_layer31_13], dim=1)

        output_layer31_14 = self.fc3114(output_layer30.view(1, -1))
        output_layer31_15 = self.fc3115(output_layer30_14.view(1, -1))
        output_layer31_16 = self.fc3116(output_layer30_15.view(1, -1))
        output_layer31_17 = self.fc3117(output_layer30_16.view(1, -1))
        output_layer31_18 = self.fc3118(output_layer30_17.view(1, -1))
        output_layer31_19 = self.fc3119(output_layer30_18.view(1, -1))

        # 第三十二层 线性层 特征映射
        output_layer32_1 = self.fc3201(output_layer31_1)
        output_layer32 = output_layer32_1
        output_layer32_2 = self.fc3202(output_layer31_2)
        output_layer32 = torch.cat([output_layer32, output_layer32_2], dim=1)
        output_layer32_3 = self.fc3203(output_layer31_3)
        output_layer32 = torch.cat([output_layer32, output_layer32_3], dim=1)
        output_layer32_4 = self.fc3204(output_layer31_4)
        output_layer32 = torch.cat([output_layer32, output_layer32_4], dim=1)
        output_layer32_5 = self.fc3205(output_layer31_5)
        output_layer32 = torch.cat([output_layer32, output_layer32_5], dim=1)
        output_layer32_6 = self.fc3206(output_layer31_6)
        output_layer32 = torch.cat([output_layer32, output_layer32_6], dim=1)
        output_layer32_7 = self.fc3207(output_layer31_7)
        output_layer32 = torch.cat([output_layer32, output_layer32_7], dim=1)
        output_layer32_8 = self.fc3208(output_layer31_8)
        output_layer32 = torch.cat([output_layer32, output_layer32_8], dim=1)
        output_layer32_9 = self.fc3209(output_layer31_9)
        output_layer32 = torch.cat([output_layer32, output_layer32_9], dim=1)
        output_layer32_10 = self.fc3210(output_layer31_10)
        output_layer32 = torch.cat([output_layer32, output_layer32_10], dim=1)
        output_layer32_11 = self.fc3211(output_layer31_11)
        output_layer32 = torch.cat([output_layer32, output_layer32_11], dim=1)
        output_layer32_12 = self.fc3212(output_layer31_12)
        output_layer32 = torch.cat([output_layer32, output_layer32_12], dim=1)
        output_layer32_13 = self.fc3213(output_layer31_13)
        output_layer32 = torch.cat([output_layer32, output_layer32_13], dim=1)

        output_layer32_14 = self.fc3214(output_layer31)
        output_layer32_15 = self.fc3215(output_layer31_14)
        output_layer32_16 = self.fc3216(output_layer31_15)
        output_layer32_17 = self.fc3217(output_layer31_16)
        output_layer32_18 = self.fc3218(output_layer31_17)
        output_layer32_19 = self.fc3219(output_layer31_18)
        output_layer32_20 = self.fc3220(output_layer31_19)

        # 第三十三层 线性层 特征映射
        output_layer33_1 = self.fc3301(output_layer32)

        output_layer33_2 = self.fc3302(output_layer32_14)
        output_layer33_3 = self.fc3303(output_layer32_15)
        output_layer33_3 = torch.cat([output_layer33_2, output_layer33_3], dim=1)

        output_layer33_4 = self.fc3304(output_layer32_16)
        output_layer33_5 = self.fc3305(output_layer32_17)
        output_layer33_6 = self.fc3306(output_layer32_18)
        output_layer33_6 = torch.cat([output_layer33_6, output_layer33_5], dim=1)
        output_layer33_6 = torch.cat([output_layer33_6, output_layer33_4], dim=1)

        output_layer33_7 = self.fc3307(output_layer32_19)
        output_layer33_8 = self.fc3308(output_layer32_20)
        output_layer33_8 = torch.cat([output_layer33_8, output_layer33_7], dim=1)

        # 第三十四层 线性层 特征映射
        output_layer34_1 = self.fc3401(output_layer33_1)
        output_layer34_2 = self.fc3402(output_layer33_3)
        output_layer34_3 = self.fc3403(output_layer33_6)
        output_layer34_4 = self.fc3404(output_layer33_8)
        output_layer34_4 = torch.cat([output_layer34_4, output_layer34_3], dim=1)

        # 第三十五层 线性层 特征映射
        output_layer35_1 = self.fc3501(output_layer34_1)
        output_layer35_2 = self.fc3502(output_layer34_2)
        output_layer35_3 = self.fc3503(output_layer34_4)
        output_layer35 = torch.cat([output_layer35_1, output_layer35_2], dim=1)
        output_layer35 = torch.cat([output_layer35, output_layer35_3], dim=1)

        # 第三十六层 线性层 特征映射
        output_layer36 = self.fc36(output_layer35)

        # 输出层
        output = output_layer36
        return output


class Deep_CNN_SMI2_relu(nn.Module):
    # 卷积神经网络，丢掉之前的LSTM层
    def __init__(self):
        super(Deep_CNN_SMI2_relu, self).__init__()

        self.conv0101 = QuadrupleConv_U(2, 1)
        self.conv0102 = TripleConv_U(2, 1)
        self.conv0103 = QuadrupleConv_U(2, 1)
        self.conv0104 = TripleConv_U(2, 1)
        self.conv0105 = QuadrupleConv_U(2, 1)
        self.conv0106 = TripleConv_U(1, 1)
        self.conv0107 = QuadrupleConv_U(5, 1)
        self.conv0108 = TripleConv_U(6, 1)
        self.conv0109 = QuadrupleConv_U(1, 1)
        self.conv0110 = TripleConv_U(1, 1)
        self.conv0111 = QuadrupleConv_U(1, 1)
        self.conv0112 = TripleConv_U(1, 1)
        self.conv0113 = QuadrupleConv_U(1, 1)

        self.conv0201 = TripleConv_U(13, 1)
        self.conv0202 = QuadrupleConv_U(13, 1)
        self.conv0203 = TripleConv_U(13, 1)
        self.conv0204 = QuadrupleConv_U(13, 1)
        self.conv0205 = TripleConv_U(13, 1)
        self.conv0206 = QuadrupleConv_U(13, 1)
        self.conv0207 = TripleConv_U(13, 1)
        self.conv0208 = QuadrupleConv_U(13, 1)
        self.conv0209 = TripleConv_U(13, 1)
        self.conv0210 = QuadrupleConv_U(13, 1)
        self.conv0211 = TripleConv_U(13, 1)
        self.conv0212 = QuadrupleConv_U(13, 1)
        self.conv0213 = TripleConv_U(13, 1)

        self.conv0301 = QuadrupleConv_U(13, 1)
        self.conv0302 = TripleConv_U(13, 1)
        self.conv0303 = QuadrupleConv_U(13, 1)
        self.conv0304 = TripleConv_U(13, 1)
        self.conv0305 = QuadrupleConv_U(13, 1)
        self.conv0306 = TripleConv_U(13, 1)
        self.conv0307 = QuadrupleConv_U(13, 1)
        self.conv0308 = TripleConv_U(13, 1)
        self.conv0309 = QuadrupleConv_U(13, 1)
        self.conv0310 = TripleConv_U(13, 1)
        self.conv0311 = QuadrupleConv_U(13, 1)
        self.conv0312 = TripleConv_U(13, 1)
        self.conv0313 = QuadrupleConv_U(13, 1)

        self.conv0401 = TripleConv_U(13, 1)
        self.conv0402 = QuadrupleConv_U(13, 1)
        self.conv0403 = TripleConv_U(13, 1)
        self.conv0404 = QuadrupleConv_U(13, 1)
        self.conv0405 = TripleConv_U(13, 1)
        self.conv0406 = QuadrupleConv_U(13, 1)
        self.conv0407 = TripleConv_U(13, 1)
        self.conv0408 = QuadrupleConv_U(13, 1)
        self.conv0409 = TripleConv_U(13, 1)
        self.conv0410 = QuadrupleConv_U(13, 1)
        self.conv0411 = TripleConv_U(13, 1)
        self.conv0412 = QuadrupleConv_U(13, 1)
        self.conv0413 = TripleConv_U(13, 1)

        self.conv0501 = QuadrupleConv_U(13, 1)
        self.conv0502 = TripleConv_U(13, 1)
        self.conv0503 = QuadrupleConv_U(13, 1)
        self.conv0504 = TripleConv_U(13, 1)
        self.conv0505 = QuadrupleConv_U(13, 1)
        self.conv0506 = TripleConv_U(13, 1)
        self.conv0507 = QuadrupleConv_U(13, 1)
        self.conv0508 = TripleConv_U(13, 1)
        self.conv0509 = QuadrupleConv_U(13, 1)
        self.conv0510 = TripleConv_U(13, 1)
        self.conv0511 = QuadrupleConv_U(13, 1)
        self.conv0512 = TripleConv_U(13, 1)
        self.conv0513 = QuadrupleConv_U(13, 1)

        self.conv0601 = TripleConv_U(13, 1)
        self.conv0602 = QuadrupleConv_U(13, 1)
        self.conv0603 = TripleConv_U(13, 1)
        self.conv0604 = QuadrupleConv_U(13, 1)
        self.conv0605 = TripleConv_U(13, 1)
        self.conv0606 = QuadrupleConv_U(13, 1)
        self.conv0607 = TripleConv_U(13, 1)
        self.conv0608 = QuadrupleConv_U(13, 1)
        self.conv0609 = TripleConv_U(13, 1)
        self.conv0610 = QuadrupleConv_U(13, 1)
        self.conv0611 = TripleConv_U(13, 1)
        self.conv0612 = QuadrupleConv_U(13, 1)
        self.conv0613 = TripleConv_U(13, 1)

        self.conv0701 = QuadrupleConv_U(13, 1)
        self.conv0702 = TripleConv_U(13, 1)
        self.conv0703 = QuadrupleConv_U(13, 1)
        self.conv0704 = TripleConv_U(13, 1)
        self.conv0705 = QuadrupleConv_U(13, 1)
        self.conv0706 = TripleConv_U(13, 1)
        self.conv0707 = QuadrupleConv_U(13, 1)
        self.conv0708 = TripleConv_U(13, 1)
        self.conv0709 = QuadrupleConv_U(13, 1)
        self.conv0710 = TripleConv_U(13, 1)
        self.conv0711 = QuadrupleConv_U(13, 1)
        self.conv0712 = TripleConv_U(13, 1)
        self.conv0713 = QuadrupleConv_U(13, 1)

        self.conv0801 = TripleConv_U(13, 1)
        self.conv0802 = QuadrupleConv_U(13, 1)
        self.conv0803 = TripleConv_U(13, 1)
        self.conv0804 = QuadrupleConv_U(13, 1)
        self.conv0805 = TripleConv_U(13, 1)
        self.conv0806 = QuadrupleConv_U(13, 1)
        self.conv0807 = TripleConv_U(13, 1)
        self.conv0808 = QuadrupleConv_U(13, 1)
        self.conv0809 = TripleConv_U(13, 1)
        self.conv0810 = QuadrupleConv_U(13, 1)
        self.conv0811 = TripleConv_U(13, 1)
        self.conv0812 = QuadrupleConv_U(13, 1)
        self.conv0813 = TripleConv_U(13, 1)

        self.conv0901 = QuadrupleConv_U(13, 1)
        self.conv0902 = TripleConv_U(13, 1)
        self.conv0903 = QuadrupleConv_U(13, 1)
        self.conv0904 = TripleConv_U(13, 1)
        self.conv0905 = QuadrupleConv_U(13, 1)
        self.conv0906 = TripleConv_U(13, 1)
        self.conv0907 = QuadrupleConv_U(13, 1)
        self.conv0908 = TripleConv_U(13, 1)
        self.conv0909 = QuadrupleConv_U(13, 1)
        self.conv0910 = TripleConv_U(13, 1)
        self.conv0911 = QuadrupleConv_U(13, 1)
        self.conv0912 = TripleConv_U(13, 1)
        self.conv0913 = QuadrupleConv_U(13, 1)

        self.conv1001 = TripleConv_U(13, 1)
        self.conv1002 = QuadrupleConv_U(13, 1)
        self.conv1003 = TripleConv_U(13, 1)
        self.conv1004 = QuadrupleConv_U(13, 1)
        self.conv1005 = TripleConv_U(13, 1)
        self.conv1006 = QuadrupleConv_U(13, 1)
        self.conv1007 = TripleConv_U(13, 1)
        self.conv1008 = QuadrupleConv_U(13, 1)
        self.conv1009 = TripleConv_U(13, 1)
        self.conv1010 = QuadrupleConv_U(13, 1)
        self.conv1011 = TripleConv_U(13, 1)
        self.conv1012 = QuadrupleConv_U(13, 1)
        self.conv1013 = TripleConv_U(13, 1)

        self.conv1101 = QuadrupleConv_U(13, 1)
        self.conv1102 = TripleConv_U(13, 1)
        self.conv1103 = QuadrupleConv_U(13, 1)
        self.conv1104 = TripleConv_U(13, 1)
        self.conv1105 = QuadrupleConv_U(13, 1)
        self.conv1106 = TripleConv_U(13, 1)
        self.conv1107 = QuadrupleConv_U(13, 1)
        self.conv1108 = TripleConv_U(13, 1)
        self.conv1109 = QuadrupleConv_U(13, 1)
        self.conv1110 = TripleConv_U(13, 1)
        self.conv1111 = QuadrupleConv_U(13, 1)
        self.conv1112 = TripleConv_U(13, 1)
        self.conv1113 = QuadrupleConv_U(13, 1)

        self.conv1201 = TripleConv_U(13, 1)
        self.conv1202 = QuadrupleConv_U(13, 1)
        self.conv1203 = TripleConv_U(13, 1)
        self.conv1204 = QuadrupleConv_U(13, 1)
        self.conv1205 = TripleConv_U(13, 1)
        self.conv1206 = QuadrupleConv_U(13, 1)
        self.conv1207 = TripleConv_U(13, 1)
        self.conv1208 = QuadrupleConv_U(13, 1)
        self.conv1209 = TripleConv_U(13, 1)
        self.conv1210 = QuadrupleConv_U(13, 1)
        self.conv1211 = TripleConv_U(13, 1)
        self.conv1212 = QuadrupleConv_U(13, 1)
        self.conv1213 = TripleConv_U(13, 1)

        self.conv1301 = QuadrupleConv_U(13, 1)
        self.conv1302 = TripleConv_U(13, 1)
        self.conv1303 = QuadrupleConv_U(13, 1)
        self.conv1304 = TripleConv_U(13, 1)
        self.conv1305 = QuadrupleConv_U(13, 1)
        self.conv1306 = TripleConv_U(13, 1)
        self.conv1307 = QuadrupleConv_U(13, 1)
        self.conv1308 = TripleConv_U(13, 1)
        self.conv1309 = QuadrupleConv_U(13, 1)
        self.conv1310 = TripleConv_U(13, 1)
        self.conv1311 = QuadrupleConv_U(13, 1)
        self.conv1312 = TripleConv_U(13, 1)
        self.conv1313 = QuadrupleConv_U(13, 1)

        self.conv1401 = TripleConv_U(13, 1)
        self.conv1402 = QuadrupleConv_U(13, 1)
        self.conv1403 = TripleConv_U(13, 1)
        self.conv1404 = QuadrupleConv_U(13, 1)
        self.conv1405 = TripleConv_U(13, 1)
        self.conv1406 = QuadrupleConv_U(13, 1)
        self.conv1407 = TripleConv_U(13, 1)
        self.conv1408 = QuadrupleConv_U(13, 1)
        self.conv1409 = TripleConv_U(13, 1)
        self.conv1410 = QuadrupleConv_U(13, 1)
        self.conv1411 = TripleConv_U(13, 1)
        self.conv1412 = QuadrupleConv_U(13, 1)
        self.conv1413 = TripleConv_U(13, 1)

        self.conv1501 = QuadrupleConv_U(13, 1)
        self.conv1502 = TripleConv_U(13, 1)
        self.conv1503 = QuadrupleConv_U(13, 1)
        self.conv1504 = TripleConv_U(13, 1)
        self.conv1505 = QuadrupleConv_U(13, 1)
        self.conv1506 = TripleConv_U(13, 1)
        self.conv1507 = QuadrupleConv_U(13, 1)
        self.conv1508 = TripleConv_U(13, 1)
        self.conv1509 = QuadrupleConv_U(13, 1)
        self.conv1510 = TripleConv_U(13, 1)
        self.conv1511 = QuadrupleConv_U(13, 1)
        self.conv1512 = TripleConv_U(13, 1)
        self.conv1513 = QuadrupleConv_U(13, 1)

        self.conv1601 = TripleConv_U(13, 1)
        self.conv1602 = QuadrupleConv_U(13, 1)
        self.conv1603 = TripleConv_U(13, 1)
        self.conv1604 = QuadrupleConv_U(13, 1)
        self.conv1605 = TripleConv_U(13, 1)
        self.conv1606 = QuadrupleConv_U(13, 1)
        self.conv1607 = TripleConv_U(13, 1)
        self.conv1608 = QuadrupleConv_U(13, 1)
        self.conv1609 = TripleConv_U(13, 1)
        self.conv1610 = QuadrupleConv_U(13, 1)
        self.conv1611 = TripleConv_U(13, 1)
        self.conv1612 = QuadrupleConv_U(13, 1)
        self.conv1613 = TripleConv_U(13, 1)

        self.conv1701 = QuadrupleConv_U(13, 1)
        self.conv1702 = TripleConv_U(13, 1)
        self.conv1703 = QuadrupleConv_U(13, 1)
        self.conv1704 = TripleConv_U(13, 1)
        self.conv1705 = QuadrupleConv_U(13, 1)
        self.conv1706 = TripleConv_U(13, 1)
        self.conv1707 = QuadrupleConv_U(13, 1)
        self.conv1708 = TripleConv_U(13, 1)
        self.conv1709 = QuadrupleConv_U(13, 1)
        self.conv1710 = TripleConv_U(13, 1)
        self.conv1711 = QuadrupleConv_U(13, 1)
        self.conv1712 = TripleConv_U(13, 1)
        self.conv1713 = QuadrupleConv_U(13, 1)

        self.conv1801 = TripleConv_U(13, 1)
        self.conv1802 = QuadrupleConv_U(13, 1)
        self.conv1803 = TripleConv_U(13, 1)
        self.conv1804 = QuadrupleConv_U(13, 1)
        self.conv1805 = TripleConv_U(13, 1)
        self.conv1806 = QuadrupleConv_U(13, 1)
        self.conv1807 = TripleConv_U(13, 1)
        self.conv1808 = QuadrupleConv_U(13, 1)
        self.conv1809 = TripleConv_U(13, 1)
        self.conv1810 = QuadrupleConv_U(13, 1)
        self.conv1811 = TripleConv_U(13, 1)
        self.conv1812 = QuadrupleConv_U(13, 1)
        self.conv1813 = TripleConv_U(13, 1)

        self.conv1901 = QuadrupleConv_U(13, 1)
        self.conv1902 = TripleConv_U(13, 1)
        self.conv1903 = QuadrupleConv_U(13, 1)
        self.conv1904 = TripleConv_U(13, 1)
        self.conv1905 = QuadrupleConv_U(13, 1)
        self.conv1906 = TripleConv_U(13, 1)
        self.conv1907 = QuadrupleConv_U(13, 1)
        self.conv1908 = TripleConv_U(13, 1)
        self.conv1909 = QuadrupleConv_U(13, 1)
        self.conv1910 = TripleConv_U(13, 1)
        self.conv1911 = QuadrupleConv_U(13, 1)
        self.conv1912 = TripleConv_U(13, 1)
        self.conv1913 = QuadrupleConv_U(13, 1)

        self.conv2001 = TripleConv_U(13, 1)
        self.conv2002 = QuadrupleConv_U(13, 1)
        self.conv2003 = TripleConv_U(13, 1)
        self.conv2004 = QuadrupleConv_U(13, 1)
        self.conv2005 = TripleConv_U(13, 1)
        self.conv2006 = QuadrupleConv_U(13, 1)
        self.conv2007 = TripleConv_U(13, 1)
        self.conv2008 = QuadrupleConv_U(13, 1)
        self.conv2009 = TripleConv_U(13, 1)
        self.conv2010 = QuadrupleConv_U(13, 1)
        self.conv2011 = TripleConv_U(13, 1)
        self.conv2012 = QuadrupleConv_U(13, 1)
        self.conv2013 = TripleConv_U(13, 1)

        self.conv2101 = QuadrupleConv_U(13, 1)
        self.conv2102 = TripleConv_U(13, 1)
        self.conv2103 = QuadrupleConv_U(13, 1)
        self.conv2104 = TripleConv_U(13, 1)
        self.conv2105 = QuadrupleConv_U(13, 1)
        self.conv2106 = TripleConv_U(13, 1)
        self.conv2107 = QuadrupleConv_U(13, 1)
        self.conv2108 = TripleConv_U(13, 1)
        self.conv2109 = QuadrupleConv_U(13, 1)
        self.conv2110 = TripleConv_U(13, 1)
        self.conv2111 = QuadrupleConv_U(13, 1)
        self.conv2112 = TripleConv_U(13, 1)
        self.conv2113 = QuadrupleConv_U(13, 1)

        self.conv2201 = TripleConv_U(13, 1)
        self.conv2202 = QuadrupleConv_U(13, 1)
        self.conv2203 = TripleConv_U(13, 1)
        self.conv2204 = QuadrupleConv_U(13, 1)
        self.conv2205 = TripleConv_U(13, 1)
        self.conv2206 = QuadrupleConv_U(13, 1)
        self.conv2207 = TripleConv_U(13, 1)
        self.conv2208 = QuadrupleConv_U(13, 1)
        self.conv2209 = TripleConv_U(13, 1)
        self.conv2210 = QuadrupleConv_U(13, 1)
        self.conv2211 = TripleConv_U(13, 1)
        self.conv2212 = QuadrupleConv_U(13, 1)
        self.conv2213 = TripleConv_U(13, 1)

        self.conv2301 = QuadrupleConv_U(13, 1)
        self.conv2302 = TripleConv_U(13, 1)
        self.conv2303 = QuadrupleConv_U(13, 1)
        self.conv2304 = TripleConv_U(13, 1)
        self.conv2305 = QuadrupleConv_U(13, 1)
        self.conv2306 = TripleConv_U(13, 1)
        self.conv2307 = QuadrupleConv_U(13, 1)
        self.conv2308 = TripleConv_U(13, 1)
        self.conv2309 = QuadrupleConv_U(13, 1)
        self.conv2310 = TripleConv_U(13, 1)
        self.conv2311 = QuadrupleConv_U(13, 1)
        self.conv2312 = TripleConv_U(13, 1)
        self.conv2313 = QuadrupleConv_U(13, 1)

        self.conv2401 = TripleConv_U(13, 1)
        self.conv2402 = QuadrupleConv_U(13, 1)
        self.conv2403 = TripleConv_U(13, 1)
        self.conv2404 = QuadrupleConv_U(13, 1)
        self.conv2405 = TripleConv_U(13, 1)
        self.conv2406 = QuadrupleConv_U(13, 1)
        self.conv2407 = TripleConv_U(13, 1)
        self.conv2408 = QuadrupleConv_U(13, 1)
        self.conv2409 = TripleConv_U(13, 1)
        self.conv2410 = QuadrupleConv_U(13, 1)
        self.conv2411 = TripleConv_U(13, 1)
        self.conv2412 = QuadrupleConv_U(13, 1)
        self.conv2413 = TripleConv_U(13, 1)

        self.conv2501 = QuadrupleConv_U(13, 1)
        self.conv2502 = TripleConv_U(13, 1)
        self.conv2503 = QuadrupleConv_U(13, 1)
        self.conv2504 = TripleConv_U(13, 1)
        self.conv2505 = QuadrupleConv_U(13, 1)
        self.conv2506 = TripleConv_U(13, 1)
        self.conv2507 = QuadrupleConv_U(13, 1)
        self.conv2508 = TripleConv_U(13, 1)
        self.conv2509 = QuadrupleConv_U(13, 1)
        self.conv2510 = TripleConv_U(13, 1)
        self.conv2511 = QuadrupleConv_U(13, 1)
        self.conv2512 = TripleConv_U(13, 1)
        self.conv2513 = QuadrupleConv_U(13, 1)

        self.fc2601 = nn.Linear(60, 60)
        self.fc2602 = nn.Linear(60, 60)
        self.fc2603 = nn.Linear(60, 60)
        self.fc2604 = nn.Linear(60, 60)
        self.fc2605 = nn.Linear(60, 60)
        self.fc2606 = nn.Linear(60, 60)
        self.fc2607 = nn.Linear(60, 60)
        self.fc2608 = nn.Linear(60, 60)
        self.fc2609 = nn.Linear(60, 60)
        self.fc2610 = nn.Linear(60, 60)
        self.fc2611 = nn.Linear(60, 60)
        self.fc2612 = nn.Linear(60, 60)
        self.fc2613 = nn.Linear(60, 60)
        self.fc2614 = nn.Linear(780, 390)

        self.fc2701 = nn.Linear(60, 60)
        self.fc2702 = nn.Linear(60, 60)
        self.fc2703 = nn.Linear(60, 60)
        self.fc2704 = nn.Linear(60, 60)
        self.fc2705 = nn.Linear(60, 60)
        self.fc2706 = nn.Linear(60, 60)
        self.fc2707 = nn.Linear(60, 60)
        self.fc2708 = nn.Linear(60, 60)
        self.fc2709 = nn.Linear(60, 60)
        self.fc2710 = nn.Linear(60, 60)
        self.fc2711 = nn.Linear(60, 60)
        self.fc2712 = nn.Linear(60, 60)
        self.fc2713 = nn.Linear(60, 60)
        self.fc2714 = nn.Linear(780, 390)
        self.fc2715 = nn.Linear(390, 240)

        self.fc2801 = nn.Linear(60, 60)
        self.fc2802 = nn.Linear(60, 60)
        self.fc2803 = nn.Linear(60, 60)
        self.fc2804 = nn.Linear(60, 60)
        self.fc2805 = nn.Linear(60, 60)
        self.fc2806 = nn.Linear(60, 60)
        self.fc2807 = nn.Linear(60, 60)
        self.fc2808 = nn.Linear(60, 60)
        self.fc2809 = nn.Linear(60, 60)
        self.fc2810 = nn.Linear(60, 60)
        self.fc2811 = nn.Linear(60, 60)
        self.fc2812 = nn.Linear(60, 60)
        self.fc2813 = nn.Linear(60, 60)
        self.fc2814 = nn.Linear(780, 390)
        self.fc2815 = nn.Linear(390, 240)
        self.fc2816 = nn.Linear(240, 120)

        self.fc2901 = nn.Linear(60, 60)
        self.fc2902 = nn.Linear(60, 60)
        self.fc2903 = nn.Linear(60, 60)
        self.fc2904 = nn.Linear(60, 60)
        self.fc2905 = nn.Linear(60, 60)
        self.fc2906 = nn.Linear(60, 60)
        self.fc2907 = nn.Linear(60, 60)
        self.fc2908 = nn.Linear(60, 60)
        self.fc2909 = nn.Linear(60, 60)
        self.fc2910 = nn.Linear(60, 60)
        self.fc2911 = nn.Linear(60, 60)
        self.fc2912 = nn.Linear(60, 60)
        self.fc2913 = nn.Linear(60, 60)
        self.fc2914 = nn.Linear(780, 390)
        self.fc2915 = nn.Linear(390, 240)
        self.fc2916 = nn.Linear(240, 120)
        self.fc2917 = nn.Linear(120, 60)

        self.fc3001 = nn.Linear(60, 60)
        self.fc3002 = nn.Linear(60, 60)
        self.fc3003 = nn.Linear(60, 60)
        self.fc3004 = nn.Linear(60, 60)
        self.fc3005 = nn.Linear(60, 60)
        self.fc3006 = nn.Linear(60, 60)
        self.fc3007 = nn.Linear(60, 60)
        self.fc3008 = nn.Linear(60, 60)
        self.fc3009 = nn.Linear(60, 60)
        self.fc3010 = nn.Linear(60, 60)
        self.fc3011 = nn.Linear(60, 60)
        self.fc3012 = nn.Linear(60, 60)
        self.fc3013 = nn.Linear(60, 60)
        self.fc3014 = nn.Linear(780, 390)
        self.fc3015 = nn.Linear(390, 240)
        self.fc3016 = nn.Linear(240, 120)
        self.fc3017 = nn.Linear(120, 60)
        self.fc3018 = nn.Linear(60, 60)

        self.fc3101 = nn.Linear(60, 60)
        self.fc3102 = nn.Linear(60, 60)
        self.fc3103 = nn.Linear(60, 60)
        self.fc3104 = nn.Linear(60, 60)
        self.fc3105 = nn.Linear(60, 60)
        self.fc3106 = nn.Linear(60, 60)
        self.fc3107 = nn.Linear(60, 60)
        self.fc3108 = nn.Linear(60, 60)
        self.fc3109 = nn.Linear(60, 60)
        self.fc3110 = nn.Linear(60, 60)
        self.fc3111 = nn.Linear(60, 60)
        self.fc3112 = nn.Linear(60, 60)
        self.fc3113 = nn.Linear(60, 60)
        self.fc3114 = nn.Linear(780, 390)
        self.fc3115 = nn.Linear(390, 240)
        self.fc3116 = nn.Linear(240, 120)
        self.fc3117 = nn.Linear(120, 60)
        self.fc3118 = nn.Linear(60, 60)
        self.fc3119 = nn.Linear(60, 60)

        self.fc3201 = nn.Linear(60, 60)
        self.fc3202 = nn.Linear(60, 60)
        self.fc3203 = nn.Linear(60, 60)
        self.fc3204 = nn.Linear(60, 60)
        self.fc3205 = nn.Linear(60, 60)
        self.fc3206 = nn.Linear(60, 60)
        self.fc3207 = nn.Linear(60, 60)
        self.fc3208 = nn.Linear(60, 60)
        self.fc3209 = nn.Linear(60, 60)
        self.fc3210 = nn.Linear(60, 60)
        self.fc3211 = nn.Linear(60, 60)
        self.fc3212 = nn.Linear(60, 60)
        self.fc3213 = nn.Linear(60, 60)
        self.fc3214 = nn.Linear(780, 390)
        self.fc3215 = nn.Linear(390, 240)
        self.fc3216 = nn.Linear(240, 120)
        self.fc3217 = nn.Linear(120, 60)
        self.fc3218 = nn.Linear(60, 60)
        self.fc3219 = nn.Linear(60, 60)
        self.fc3220 = nn.Linear(60, 60)

        self.fc3301 = nn.Linear(780, 390)
        self.fc3302 = nn.Linear(390, 120)
        self.fc3303 = nn.Linear(240, 120)
        self.fc3304 = nn.Linear(120, 60)
        self.fc3305 = nn.Linear(60, 60)
        self.fc3306 = nn.Linear(60, 60)
        self.fc3307 = nn.Linear(60, 60)
        self.fc3308 = nn.Linear(60, 60)

        self.fc3401 = nn.Linear(390, 120)
        self.fc3402 = nn.Linear(240, 120)
        self.fc3403 = nn.Linear(180, 60)
        self.fc3404 = nn.Linear(120, 60)

        self.fc3501 = nn.Linear(120, 60)
        self.fc3502 = nn.Linear(120, 60)
        self.fc3503 = nn.Linear(120, 60)

        self.fc36 = nn.Linear(180, 60)

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
        layer_in_1_1 = layer_in_1_1.view(layer_in_1_1.size(0), layer_in_1_1.size(1), -1, 10)

        layer_in_1_2 = train_well[:, 1, :]
        layer_in_1_2 = torch.cat([layer_in_1_2.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_2 = layer_in_1_2.view(layer_in_1_2.size(0), layer_in_1_2.size(1), -1, 10)

        layer_in_1_3 = train_well[:, 2, :]
        layer_in_1_3 = torch.cat([layer_in_1_3.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_3 = layer_in_1_3.view(layer_in_1_3.size(0), layer_in_1_3.size(1), -1, 10)

        layer_in_1_4 = train_well[:, 3, :]
        layer_in_1_4 = torch.cat([layer_in_1_4.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_4 = layer_in_1_4.view(layer_in_1_4.size(0), layer_in_1_4.size(1), -1, 10)

        layer_in_1_5 = train_well[:, 4, :]
        layer_in_1_5 = torch.cat([layer_in_1_5.view(1, 1, -1), seismic_in], dim=1)
        layer_in_1_5 = layer_in_1_5.view(layer_in_1_5.size(0), layer_in_1_5.size(1), -1, 10)

        layer_in_1_6 = seismic.view(1, 1, -1)
        layer_in_1_6 = layer_in_1_6.view(seismic.size(0), seismic.size(0), -1, 10)

        layer_in_1_7 = train_well.view(1, 5, -1)
        layer_in_1_7 = layer_in_1_7.view(train_well.size(0), train_well.size(1), -1, 10)

        layer_in_1_8 = torch.cat([layer_in_1_6, layer_in_1_7], dim=1)

        layer_in_1_9 = train_well[:, 0, :].view(train_well.size(0), train_well.size(0), -1, 10)
        layer_in_1_10 = train_well[:, 1, :].view(train_well.size(0), train_well.size(0), -1, 10)
        layer_in_1_11 = train_well[:, 2, :].view(train_well.size(0), train_well.size(0), -1, 10)
        layer_in_1_12 = train_well[:, 3, :].view(train_well.size(0), train_well.size(0), -1, 10)
        layer_in_1_13 = train_well[:, 4, :].view(train_well.size(0), train_well.size(0), -1, 10)

        # 第一层 卷积层
        output_layer1_1 = self.conv0101(layer_in_1_1)
        output_layer1 = output_layer1_1
        output_layer1_2 = self.conv0102(layer_in_1_2)
        output_layer1 = torch.cat([output_layer1, output_layer1_2], dim=1)
        output_layer1_3 = self.conv0103(layer_in_1_3)
        output_layer1 = torch.cat([output_layer1, output_layer1_3], dim=1)
        output_layer1_4 = self.conv0104(layer_in_1_4)
        output_layer1 = torch.cat([output_layer1, output_layer1_4], dim=1)
        output_layer1_5 = self.conv0105(layer_in_1_5)
        output_layer1 = torch.cat([output_layer1, output_layer1_5], dim=1)
        output_layer1_6 = self.conv0106(layer_in_1_6)
        output_layer1 = torch.cat([output_layer1, output_layer1_6], dim=1)
        output_layer1_7 = self.conv0107(layer_in_1_7)
        output_layer1 = torch.cat([output_layer1, output_layer1_7], dim=1)
        output_layer1_8 = self.conv0108(layer_in_1_8)
        output_layer1 = torch.cat([output_layer1, output_layer1_8], dim=1)
        output_layer1_9 = self.conv0109(layer_in_1_9)
        output_layer1 = torch.cat([output_layer1, output_layer1_9], dim=1)
        output_layer1_10 = self.conv0110(layer_in_1_10)
        output_layer1 = torch.cat([output_layer1, output_layer1_10], dim=1)
        output_layer1_11 = self.conv0111(layer_in_1_11)
        output_layer1 = torch.cat([output_layer1, output_layer1_11], dim=1)
        output_layer1_12 = self.conv0112(layer_in_1_12)
        output_layer1 = torch.cat([output_layer1, output_layer1_12], dim=1)
        output_layer1_13 = self.conv0113(layer_in_1_13)
        output_layer1 = torch.cat([output_layer1, output_layer1_13], dim=1)

        # 第二层 卷积层
        output_layer2_1 = self.conv0201(output_layer1)
        output_layer2 = output_layer2_1
        output_layer2_2 = self.conv0202(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_2], dim=1)
        output_layer2_3 = self.conv0203(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_3], dim=1)
        output_layer2_4 = self.conv0204(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_4], dim=1)
        output_layer2_5 = self.conv0205(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_5], dim=1)
        output_layer2_6 = self.conv0206(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_6], dim=1)
        output_layer2_7 = self.conv0207(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_7], dim=1)
        output_layer2_8 = self.conv0208(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_8], dim=1)
        output_layer2_9 = self.conv0209(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_9], dim=1)
        output_layer2_10 = self.conv0210(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_10], dim=1)
        output_layer2_11 = self.conv0211(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_11], dim=1)
        output_layer2_12 = self.conv0212(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_12], dim=1)
        output_layer2_13 = self.conv0213(output_layer1)
        output_layer2 = torch.cat([output_layer2, output_layer2_13], dim=1)

        # 第三层 卷积层
        output_layer3_1 = self.conv0301(output_layer2)
        output_layer3 = output_layer3_1
        output_layer3_2 = self.conv0302(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_2], dim=1)
        output_layer3_3 = self.conv0303(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_3], dim=1)
        output_layer3_4 = self.conv0304(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_4], dim=1)
        output_layer3_5 = self.conv0305(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_5], dim=1)
        output_layer3_6 = self.conv0306(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_6], dim=1)
        output_layer3_7 = self.conv0307(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_7], dim=1)
        output_layer3_8 = self.conv0308(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_8], dim=1)
        output_layer3_9 = self.conv0309(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_9], dim=1)
        output_layer3_10 = self.conv0310(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_10], dim=1)
        output_layer3_11 = self.conv0311(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_11], dim=1)
        output_layer3_12 = self.conv0312(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_12], dim=1)
        output_layer3_13 = self.conv0313(output_layer2)
        output_layer3 = torch.cat([output_layer3, output_layer3_13], dim=1)

        # 第四层 卷积层
        output_layer4_1 = self.conv0401(output_layer3)
        output_layer4 = output_layer4_1
        output_layer4_2 = self.conv0402(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_2], dim=1)
        output_layer4_3 = self.conv0403(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_3], dim=1)
        output_layer4_4 = self.conv0404(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_4], dim=1)
        output_layer4_5 = self.conv0405(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_5], dim=1)
        output_layer4_6 = self.conv0406(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_6], dim=1)
        output_layer4_7 = self.conv0407(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_7], dim=1)
        output_layer4_8 = self.conv0408(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_8], dim=1)
        output_layer4_9 = self.conv0409(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_9], dim=1)
        output_layer4_10 = self.conv0410(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_10], dim=1)
        output_layer4_11 = self.conv0411(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_11], dim=1)
        output_layer4_12 = self.conv0412(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_12], dim=1)
        output_layer4_13 = self.conv0413(output_layer3)
        output_layer4 = torch.cat([output_layer4, output_layer4_13], dim=1)

        # 第五层 卷积层
        output_layer5_1 = self.conv0501(output_layer4)
        output_layer5 = output_layer5_1
        output_layer5_2 = self.conv0502(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_2], dim=1)
        output_layer5_3 = self.conv0503(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_3], dim=1)
        output_layer5_4 = self.conv0504(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_4], dim=1)
        output_layer5_5 = self.conv0505(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_5], dim=1)
        output_layer5_6 = self.conv0506(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_6], dim=1)
        output_layer5_7 = self.conv0507(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_7], dim=1)
        output_layer5_8 = self.conv0508(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_8], dim=1)
        output_layer5_9 = self.conv0509(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_9], dim=1)
        output_layer5_10 = self.conv0510(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_10], dim=1)
        output_layer5_11 = self.conv0511(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_11], dim=1)
        output_layer5_12 = self.conv0512(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_12], dim=1)
        output_layer5_13 = self.conv0513(output_layer4)
        output_layer5 = torch.cat([output_layer5, output_layer5_13], dim=1)

        # 第六层 卷积层
        output_layer6_1 = self.conv0601(output_layer5)
        output_layer6 = output_layer6_1
        output_layer6_2 = self.conv0602(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_2], dim=1)
        output_layer6_3 = self.conv0603(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_3], dim=1)
        output_layer6_4 = self.conv0604(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_4], dim=1)
        output_layer6_5 = self.conv0605(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_5], dim=1)
        output_layer6_6 = self.conv0606(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_6], dim=1)
        output_layer6_7 = self.conv0607(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_7], dim=1)
        output_layer6_8 = self.conv0608(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_8], dim=1)
        output_layer6_9 = self.conv0609(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_9], dim=1)
        output_layer6_10 = self.conv0610(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_10], dim=1)
        output_layer6_11 = self.conv0611(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_11], dim=1)
        output_layer6_12 = self.conv0612(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_12], dim=1)
        output_layer6_13 = self.conv0613(output_layer5)
        output_layer6 = torch.cat([output_layer6, output_layer6_13], dim=1)

        # 第七层 卷积层
        output_layer7_1 = self.conv0701(output_layer6)
        output_layer7 = output_layer7_1
        output_layer7_2 = self.conv0702(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_2], dim=1)
        output_layer7_3 = self.conv0703(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_3], dim=1)
        output_layer7_4 = self.conv0704(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_4], dim=1)
        output_layer7_5 = self.conv0705(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_5], dim=1)
        output_layer7_6 = self.conv0706(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_6], dim=1)
        output_layer7_7 = self.conv0707(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_7], dim=1)
        output_layer7_8 = self.conv0708(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_8], dim=1)
        output_layer7_9 = self.conv0709(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_9], dim=1)
        output_layer7_10 = self.conv0710(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_10], dim=1)
        output_layer7_11 = self.conv0711(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_11], dim=1)
        output_layer7_12 = self.conv0712(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_12], dim=1)
        output_layer7_13 = self.conv0713(output_layer6)
        output_layer7 = torch.cat([output_layer7, output_layer7_13], dim=1)

        # 第八层 卷积层
        output_layer8_1 = self.conv0801(output_layer7)
        output_layer8 = output_layer8_1
        output_layer8_2 = self.conv0802(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_2], dim=1)
        output_layer8_3 = self.conv0803(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_3], dim=1)
        output_layer8_4 = self.conv0804(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_4], dim=1)
        output_layer8_5 = self.conv0805(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_5], dim=1)
        output_layer8_6 = self.conv0806(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_6], dim=1)
        output_layer8_7 = self.conv0807(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_7], dim=1)
        output_layer8_8 = self.conv0808(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_8], dim=1)
        output_layer8_9 = self.conv0809(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_9], dim=1)
        output_layer8_10 = self.conv0810(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_10], dim=1)
        output_layer8_11 = self.conv0811(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_11], dim=1)
        output_layer8_12 = self.conv0812(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_12], dim=1)
        output_layer8_13 = self.conv0813(output_layer7)
        output_layer8 = torch.cat([output_layer8, output_layer8_13], dim=1)

        # 第九层 卷积层
        output_layer9_1 = self.conv0901(output_layer8)
        output_layer9 = output_layer9_1
        output_layer9_2 = self.conv0902(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_2], dim=1)
        output_layer9_3 = self.conv0903(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_3], dim=1)
        output_layer9_4 = self.conv0904(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_4], dim=1)
        output_layer9_5 = self.conv0905(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_5], dim=1)
        output_layer9_6 = self.conv0906(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_6], dim=1)
        output_layer9_7 = self.conv0907(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_7], dim=1)
        output_layer9_8 = self.conv0908(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_8], dim=1)
        output_layer9_9 = self.conv0909(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_9], dim=1)
        output_layer9_10 = self.conv0910(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_10], dim=1)
        output_layer9_11 = self.conv0911(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_11], dim=1)
        output_layer9_12 = self.conv0912(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_12], dim=1)
        output_layer9_13 = self.conv0913(output_layer8)
        output_layer9 = torch.cat([output_layer9, output_layer9_13], dim=1)

        # 第十层 卷积层
        output_layer10_1 = self.conv1001(output_layer9)
        output_layer10 = output_layer10_1
        output_layer10_2 = self.conv1002(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_2], dim=1)
        output_layer10_3 = self.conv1003(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_3], dim=1)
        output_layer10_4 = self.conv1004(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_4], dim=1)
        output_layer10_5 = self.conv1005(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_5], dim=1)
        output_layer10_6 = self.conv1006(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_6], dim=1)
        output_layer10_7 = self.conv1007(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_7], dim=1)
        output_layer10_8 = self.conv1008(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_8], dim=1)
        output_layer10_9 = self.conv1009(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_9], dim=1)
        output_layer10_10 = self.conv1010(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_10], dim=1)
        output_layer10_11 = self.conv1011(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_11], dim=1)
        output_layer10_12 = self.conv1012(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_12], dim=1)
        output_layer10_13 = self.conv1013(output_layer9)
        output_layer10 = torch.cat([output_layer10, output_layer10_13], dim=1)

        # 第十一层 卷积层
        output_layer11_1 = self.conv1101(output_layer10)
        output_layer11 = output_layer11_1
        output_layer11_2 = self.conv1102(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_2], dim=1)
        output_layer11_3 = self.conv1103(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_3], dim=1)
        output_layer11_4 = self.conv1104(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_4], dim=1)
        output_layer11_5 = self.conv1105(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_5], dim=1)
        output_layer11_6 = self.conv1106(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_6], dim=1)
        output_layer11_7 = self.conv1107(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_7], dim=1)
        output_layer11_8 = self.conv1108(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_8], dim=1)
        output_layer11_9 = self.conv1109(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_9], dim=1)
        output_layer11_10 = self.conv1110(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_10], dim=1)
        output_layer11_11 = self.conv1111(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_11], dim=1)
        output_layer11_12 = self.conv1112(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_12], dim=1)
        output_layer11_13 = self.conv1113(output_layer10)
        output_layer11 = torch.cat([output_layer11, output_layer11_13], dim=1)

        # 第十二层 卷积层
        output_layer12_1 = self.conv1201(output_layer11)
        output_layer12 = output_layer12_1
        output_layer12_2 = self.conv1202(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_2], dim=1)
        output_layer12_3 = self.conv1203(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_3], dim=1)
        output_layer12_4 = self.conv1204(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_4], dim=1)
        output_layer12_5 = self.conv1205(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_5], dim=1)
        output_layer12_6 = self.conv1206(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_6], dim=1)
        output_layer12_7 = self.conv1207(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_7], dim=1)
        output_layer12_8 = self.conv1208(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_8], dim=1)
        output_layer12_9 = self.conv1209(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_9], dim=1)
        output_layer12_10 = self.conv1210(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_10], dim=1)
        output_layer12_11 = self.conv1211(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_11], dim=1)
        output_layer12_12 = self.conv1212(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_12], dim=1)
        output_layer12_13 = self.conv1213(output_layer11)
        output_layer12 = torch.cat([output_layer12, output_layer12_13], dim=1)

        # 第十三层 卷积层
        output_layer13_1 = self.conv1301(output_layer12)
        output_layer13 = output_layer13_1
        output_layer13_2 = self.conv1302(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_2], dim=1)
        output_layer13_3 = self.conv1303(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_3], dim=1)
        output_layer13_4 = self.conv1304(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_4], dim=1)
        output_layer13_5 = self.conv1305(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_5], dim=1)
        output_layer13_6 = self.conv1306(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_6], dim=1)
        output_layer13_7 = self.conv1307(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_7], dim=1)
        output_layer13_8 = self.conv1308(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_8], dim=1)
        output_layer13_9 = self.conv1309(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_9], dim=1)
        output_layer13_10 = self.conv1310(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_10], dim=1)
        output_layer13_11 = self.conv1311(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_11], dim=1)
        output_layer13_12 = self.conv1312(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_12], dim=1)
        output_layer13_13 = self.conv1313(output_layer12)
        output_layer13 = torch.cat([output_layer13, output_layer13_13], dim=1)

        # 第十四层 卷积层
        output_layer14_1 = self.conv1401(output_layer13)
        output_layer14 = output_layer14_1
        output_layer14_2 = self.conv1402(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_2], dim=1)
        output_layer14_3 = self.conv1403(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_3], dim=1)
        output_layer14_4 = self.conv1404(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_4], dim=1)
        output_layer14_5 = self.conv1405(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_5], dim=1)
        output_layer14_6 = self.conv1406(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_6], dim=1)
        output_layer14_7 = self.conv1407(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_7], dim=1)
        output_layer14_8 = self.conv1408(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_8], dim=1)
        output_layer14_9 = self.conv1409(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_9], dim=1)
        output_layer14_10 = self.conv1410(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_10], dim=1)
        output_layer14_11 = self.conv1411(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_11], dim=1)
        output_layer14_12 = self.conv1412(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_12], dim=1)
        output_layer14_13 = self.conv1413(output_layer13)
        output_layer14 = torch.cat([output_layer14, output_layer14_13], dim=1)

        # 第十五层 卷积层
        output_layer15_1 = self.conv1501(output_layer14)
        output_layer15 = output_layer15_1
        output_layer15_2 = self.conv1502(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_2], dim=1)
        output_layer15_3 = self.conv1503(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_3], dim=1)
        output_layer15_4 = self.conv1504(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_4], dim=1)
        output_layer15_5 = self.conv1505(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_5], dim=1)
        output_layer15_6 = self.conv1506(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_6], dim=1)
        output_layer15_7 = self.conv1507(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_7], dim=1)
        output_layer15_8 = self.conv1508(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_8], dim=1)
        output_layer15_9 = self.conv1509(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_9], dim=1)
        output_layer15_10 = self.conv1510(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_10], dim=1)
        output_layer15_11 = self.conv1511(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_11], dim=1)
        output_layer15_12 = self.conv1512(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_12], dim=1)
        output_layer15_13 = self.conv1513(output_layer14)
        output_layer15 = torch.cat([output_layer15, output_layer15_13], dim=1)

        # 第十六层 卷积层
        output_layer16_1 = self.conv1601(output_layer15)
        output_layer16 = output_layer16_1
        output_layer16_2 = self.conv1602(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_2], dim=1)
        output_layer16_3 = self.conv1603(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_3], dim=1)
        output_layer16_4 = self.conv1604(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_4], dim=1)
        output_layer16_5 = self.conv1605(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_5], dim=1)
        output_layer16_6 = self.conv1606(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_6], dim=1)
        output_layer16_7 = self.conv1607(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_7], dim=1)
        output_layer16_8 = self.conv1608(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_8], dim=1)
        output_layer16_9 = self.conv1609(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_9], dim=1)
        output_layer16_10 = self.conv1610(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_10], dim=1)
        output_layer16_11 = self.conv1611(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_11], dim=1)
        output_layer16_12 = self.conv1612(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_12], dim=1)
        output_layer16_13 = self.conv1613(output_layer15)
        output_layer16 = torch.cat([output_layer16, output_layer16_13], dim=1)

        # 第十七层 卷积层
        output_layer17_1 = self.conv1701(output_layer16)
        output_layer17 = output_layer17_1
        output_layer17_2 = self.conv1702(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_2], dim=1)
        output_layer17_3 = self.conv1703(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_3], dim=1)
        output_layer17_4 = self.conv1704(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_4], dim=1)
        output_layer17_5 = self.conv1705(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_5], dim=1)
        output_layer17_6 = self.conv1706(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_6], dim=1)
        output_layer17_7 = self.conv1707(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_7], dim=1)
        output_layer17_8 = self.conv1708(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_8], dim=1)
        output_layer17_9 = self.conv1709(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_9], dim=1)
        output_layer17_10 = self.conv1710(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_10], dim=1)
        output_layer17_11 = self.conv1711(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_11], dim=1)
        output_layer17_12 = self.conv1712(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_12], dim=1)
        output_layer17_13 = self.conv1713(output_layer16)
        output_layer17 = torch.cat([output_layer17, output_layer17_13], dim=1)

        # 第十八层 卷积层
        output_layer18_1 = self.conv1801(output_layer17)
        output_layer18 = output_layer18_1
        output_layer18_2 = self.conv1802(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_2], dim=1)
        output_layer18_3 = self.conv1803(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_3], dim=1)
        output_layer18_4 = self.conv1804(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_4], dim=1)
        output_layer18_5 = self.conv1805(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_5], dim=1)
        output_layer18_6 = self.conv1806(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_6], dim=1)
        output_layer18_7 = self.conv1807(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_7], dim=1)
        output_layer18_8 = self.conv1808(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_8], dim=1)
        output_layer18_9 = self.conv1809(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_9], dim=1)
        output_layer18_10 = self.conv1810(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_10], dim=1)
        output_layer18_11 = self.conv1811(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_11], dim=1)
        output_layer18_12 = self.conv1812(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_12], dim=1)
        output_layer18_13 = self.conv1813(output_layer17)
        output_layer18 = torch.cat([output_layer18, output_layer18_13], dim=1)

        # 第十九层 卷积层
        output_layer19_1 = self.conv1901(output_layer18)
        output_layer19 = output_layer19_1
        output_layer19_2 = self.conv1902(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_2], dim=1)
        output_layer19_3 = self.conv1903(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_3], dim=1)
        output_layer19_4 = self.conv1904(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_4], dim=1)
        output_layer19_5 = self.conv1905(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_5], dim=1)
        output_layer19_6 = self.conv1906(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_6], dim=1)
        output_layer19_7 = self.conv1907(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_7], dim=1)
        output_layer19_8 = self.conv1908(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_8], dim=1)
        output_layer19_9 = self.conv1909(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_9], dim=1)
        output_layer19_10 = self.conv1910(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_10], dim=1)
        output_layer19_11 = self.conv1911(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_11], dim=1)
        output_layer19_12 = self.conv1912(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_12], dim=1)
        output_layer19_13 = self.conv1913(output_layer18)
        output_layer19 = torch.cat([output_layer19, output_layer19_13], dim=1)

        # 第二十层 卷积层
        output_layer20_1 = self.conv2001(output_layer19)
        output_layer20 = output_layer20_1
        output_layer20_2 = self.conv2002(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_2], dim=1)
        output_layer20_3 = self.conv2003(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_3], dim=1)
        output_layer20_4 = self.conv2004(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_4], dim=1)
        output_layer20_5 = self.conv2005(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_5], dim=1)
        output_layer20_6 = self.conv2006(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_6], dim=1)
        output_layer20_7 = self.conv2007(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_7], dim=1)
        output_layer20_8 = self.conv2008(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_8], dim=1)
        output_layer20_9 = self.conv2009(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_9], dim=1)
        output_layer20_10 = self.conv2010(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_10], dim=1)
        output_layer20_11 = self.conv2011(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_11], dim=1)
        output_layer20_12 = self.conv2012(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_12], dim=1)
        output_layer20_13 = self.conv2013(output_layer19)
        output_layer20 = torch.cat([output_layer20, output_layer20_13], dim=1)

        # 第二十一层 卷积层
        output_layer21_1 = self.conv2101(output_layer20)
        output_layer21 = output_layer21_1
        output_layer21_2 = self.conv2102(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_2], dim=1)
        output_layer21_3 = self.conv2103(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_3], dim=1)
        output_layer21_4 = self.conv2104(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_4], dim=1)
        output_layer21_5 = self.conv2105(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_5], dim=1)
        output_layer21_6 = self.conv2106(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_6], dim=1)
        output_layer21_7 = self.conv2107(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_7], dim=1)
        output_layer21_8 = self.conv2108(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_8], dim=1)
        output_layer21_9 = self.conv2109(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_9], dim=1)
        output_layer21_10 = self.conv2110(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_10], dim=1)
        output_layer21_11 = self.conv2111(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_11], dim=1)
        output_layer21_12 = self.conv2112(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_12], dim=1)
        output_layer21_13 = self.conv2113(output_layer20)
        output_layer21 = torch.cat([output_layer21, output_layer21_13], dim=1)

        # 第二十二层 卷积层
        output_layer22_1 = self.conv2201(output_layer21)
        output_layer22 = output_layer22_1
        output_layer22_2 = self.conv2202(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_2], dim=1)
        output_layer22_3 = self.conv2203(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_3], dim=1)
        output_layer22_4 = self.conv2204(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_4], dim=1)
        output_layer22_5 = self.conv2205(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_5], dim=1)
        output_layer22_6 = self.conv2206(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_6], dim=1)
        output_layer22_7 = self.conv2207(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_7], dim=1)
        output_layer22_8 = self.conv2208(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_8], dim=1)
        output_layer22_9 = self.conv2209(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_9], dim=1)
        output_layer22_10 = self.conv2210(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_10], dim=1)
        output_layer22_11 = self.conv2211(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_11], dim=1)
        output_layer22_12 = self.conv2212(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_12], dim=1)
        output_layer22_13 = self.conv2213(output_layer21)
        output_layer22 = torch.cat([output_layer22, output_layer22_13], dim=1)

        # 第二十三层 卷积层
        output_layer23_1 = self.conv2301(output_layer22)
        output_layer23 = output_layer23_1
        output_layer23_2 = self.conv2302(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_2], dim=1)
        output_layer23_3 = self.conv2303(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_3], dim=1)
        output_layer23_4 = self.conv2304(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_4], dim=1)
        output_layer23_5 = self.conv2305(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_5], dim=1)
        output_layer23_6 = self.conv2306(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_6], dim=1)
        output_layer23_7 = self.conv2307(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_7], dim=1)
        output_layer23_8 = self.conv2308(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_8], dim=1)
        output_layer23_9 = self.conv2309(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_9], dim=1)
        output_layer23_10 = self.conv2310(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_10], dim=1)
        output_layer23_11 = self.conv2311(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_11], dim=1)
        output_layer23_12 = self.conv2312(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_12], dim=1)
        output_layer23_13 = self.conv2313(output_layer22)
        output_layer23 = torch.cat([output_layer23, output_layer23_13], dim=1)

        # 第二十四层 卷积层
        output_layer24_1 = self.conv2401(output_layer23)
        output_layer24 = output_layer24_1
        output_layer24_2 = self.conv2402(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_2], dim=1)
        output_layer24_3 = self.conv2403(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_3], dim=1)
        output_layer24_4 = self.conv2404(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_4], dim=1)
        output_layer24_5 = self.conv2405(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_5], dim=1)
        output_layer24_6 = self.conv2406(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_6], dim=1)
        output_layer24_7 = self.conv2407(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_7], dim=1)
        output_layer24_8 = self.conv2408(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_8], dim=1)
        output_layer24_9 = self.conv2409(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_9], dim=1)
        output_layer24_10 = self.conv2410(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_10], dim=1)
        output_layer24_11 = self.conv2411(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_11], dim=1)
        output_layer24_12 = self.conv2412(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_12], dim=1)
        output_layer24_13 = self.conv2413(output_layer23)
        output_layer24 = torch.cat([output_layer24, output_layer24_13], dim=1)

        # 第二十五层 卷积层
        output_layer25_1 = self.conv2501(output_layer24)
        output_layer25 = output_layer25_1
        output_layer25_2 = self.conv2502(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_2], dim=1)
        output_layer25_3 = self.conv2503(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_3], dim=1)
        output_layer25_4 = self.conv2504(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_4], dim=1)
        output_layer25_5 = self.conv2505(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_5], dim=1)
        output_layer25_6 = self.conv2506(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_6], dim=1)
        output_layer25_7 = self.conv2507(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_7], dim=1)
        output_layer25_8 = self.conv2508(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_8], dim=1)
        output_layer25_9 = self.conv2509(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_9], dim=1)
        output_layer25_10 = self.conv2510(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_10], dim=1)
        output_layer25_11 = self.conv2511(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_11], dim=1)
        output_layer25_12 = self.conv2512(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_12], dim=1)
        output_layer25_13 = self.conv2513(output_layer24)
        output_layer25 = torch.cat([output_layer25, output_layer25_13], dim=1)

        # 第二十六层 LSTM层 特征映射
        output_layer26_1 = output_layer25_1.view(1, -1)
        output_layer26_1 = F.relu(self.fc2601(output_layer26_1))
        output_layer26 = output_layer26_1

        output_layer26_2 = output_layer25_2.view(1, -1)
        output_layer26_2 = F.relu(self.fc2602(output_layer26_2))
        output_layer26 = torch.cat([output_layer26, output_layer26_2], dim=1)

        output_layer26_3 = output_layer25_3.view(1, -1)
        output_layer26_3 = F.relu(self.fc2603(output_layer26_3))
        output_layer26 = torch.cat([output_layer26, output_layer26_3], dim=1)

        output_layer26_4 = output_layer25_4.view(1, -1)
        output_layer26_4 = F.relu(self.fc2604(output_layer26_4))
        output_layer26 = torch.cat([output_layer26, output_layer26_4], dim=1)

        output_layer26_5 = output_layer25_5.view(1, -1)
        output_layer26_5 = F.relu(self.fc2605(output_layer26_5))
        output_layer26 = torch.cat([output_layer26, output_layer26_5], dim=1)

        output_layer26_6 = output_layer25_6.view(1, -1)
        output_layer26_6 = F.relu(self.fc2606(output_layer26_6))
        output_layer26 = torch.cat([output_layer26, output_layer26_6], dim=1)

        output_layer26_7 = output_layer25_7.view(1, -1)
        output_layer26_7 = F.relu(self.fc2607(output_layer26_7))
        output_layer26 = torch.cat([output_layer26, output_layer26_7], dim=1)

        output_layer26_8 = output_layer25_8.view(1, -1)
        output_layer26_8 = F.relu(self.fc2608(output_layer26_8))
        output_layer26 = torch.cat([output_layer26, output_layer26_8], dim=1)

        output_layer26_9 = output_layer25_9.view(1, -1)
        output_layer26_9 = F.relu(self.fc2609(output_layer26_9))
        output_layer26 = torch.cat([output_layer26, output_layer26_9], dim=1)

        output_layer26_10 = output_layer25_10.view(1, -1)
        output_layer26_10 = F.relu(self.fc2610(output_layer26_10))
        output_layer26 = torch.cat([output_layer26, output_layer26_10], dim=1)

        output_layer26_11 = output_layer25_11.view(1, -1)
        output_layer26_11 = F.relu(self.fc2611(output_layer26_11))
        output_layer26 = torch.cat([output_layer26, output_layer26_11], dim=1)

        output_layer26_12 = output_layer25_12.view(1, -1)
        output_layer26_12 = F.relu(self.fc2612(output_layer26_12))
        output_layer26 = torch.cat([output_layer26, output_layer26_12], dim=1)

        output_layer26_13 = output_layer25_13.view(1, -1)
        output_layer26_13 = F.relu(self.fc2613(output_layer26_13))
        output_layer26 = torch.cat([output_layer26, output_layer26_13], dim=1)

        output_layer26_14 = output_layer25.view(1, -1)
        output_layer26_14 = F.relu(self.fc2614(output_layer26_14))

        # 第二十七层 LSTM层 特征映射
        output_layer27_1 = F.relu(self.fc2701(output_layer26_1))
        output_layer27 = output_layer27_1
        output_layer27_2 = F.relu(self.fc2702(output_layer26_2))
        output_layer27 = torch.cat([output_layer27, output_layer27_2], dim=1)
        output_layer27_3 = F.relu(self.fc2703(output_layer26_3))
        output_layer27 = torch.cat([output_layer27, output_layer27_3], dim=1)
        output_layer27_4 = F.relu(self.fc2704(output_layer26_4))
        output_layer27 = torch.cat([output_layer27, output_layer27_4], dim=1)
        output_layer27_5 = F.relu(self.fc2705(output_layer26_5))
        output_layer27 = torch.cat([output_layer27, output_layer27_5], dim=1)
        output_layer27_6 = F.relu(self.fc2706(output_layer26_6))
        output_layer27 = torch.cat([output_layer27, output_layer27_6], dim=1)
        output_layer27_7 = F.relu(self.fc2707(output_layer26_7))
        output_layer27 = torch.cat([output_layer27, output_layer27_7], dim=1)
        output_layer27_8 = F.relu(self.fc2708(output_layer26_8))
        output_layer27 = torch.cat([output_layer27, output_layer27_8], dim=1)
        output_layer27_9 = F.relu(self.fc2709(output_layer26_9))
        output_layer27 = torch.cat([output_layer27, output_layer27_9], dim=1)
        output_layer27_10 = F.relu(self.fc2710(output_layer26_10))
        output_layer27 = torch.cat([output_layer27, output_layer27_10], dim=1)
        output_layer27_11 = F.relu(self.fc2711(output_layer26_11))
        output_layer27 = torch.cat([output_layer27, output_layer27_11], dim=1)
        output_layer27_12 = F.relu(self.fc2712(output_layer26_12))
        output_layer27 = torch.cat([output_layer27, output_layer27_12], dim=1)
        output_layer27_13 = F.relu(self.fc2713(output_layer26_13))
        output_layer27 = torch.cat([output_layer27, output_layer27_13], dim=1)

        output_layer27_14 = F.relu(self.fc2714(output_layer26))
        output_layer27_15 = F.relu(self.fc2715(output_layer26_14))

        # 第二十八层 LSTM层 特征映射
        output_layer28_1 = F.relu(self.fc2801(output_layer27_1))
        output_layer28 = output_layer28_1
        output_layer28_2 = F.relu(self.fc2802(output_layer27_2))
        output_layer28 = torch.cat([output_layer28, output_layer28_2], dim=1)
        output_layer28_3 = F.relu(self.fc2803(output_layer27_3))
        output_layer28 = torch.cat([output_layer28, output_layer28_3], dim=1)
        output_layer28_4 = F.relu(self.fc2804(output_layer27_4))
        output_layer28 = torch.cat([output_layer28, output_layer28_4], dim=1)
        output_layer28_5 = F.relu(self.fc2805(output_layer27_5))
        output_layer28 = torch.cat([output_layer28, output_layer28_5], dim=1)
        output_layer28_6 = F.relu(self.fc2806(output_layer27_6))
        output_layer28 = torch.cat([output_layer28, output_layer28_6], dim=1)
        output_layer28_7 = F.relu(self.fc2807(output_layer27_7))
        output_layer28 = torch.cat([output_layer28, output_layer28_7], dim=1)
        output_layer28_8 = F.relu(self.fc2808(output_layer27_8))
        output_layer28 = torch.cat([output_layer28, output_layer28_8], dim=1)
        output_layer28_9 = F.relu(self.fc2809(output_layer27_9))
        output_layer28 = torch.cat([output_layer28, output_layer28_9], dim=1)
        output_layer28_10 = F.relu(self.fc2810(output_layer27_10))
        output_layer28 = torch.cat([output_layer28, output_layer28_10], dim=1)
        output_layer28_11 = F.relu(self.fc2811(output_layer27_11))
        output_layer28 = torch.cat([output_layer28, output_layer28_11], dim=1)
        output_layer28_12 = F.relu(self.fc2812(output_layer27_12))
        output_layer28 = torch.cat([output_layer28, output_layer28_12], dim=1)
        output_layer28_13 = F.relu(self.fc2813(output_layer27_13))
        output_layer28 = torch.cat([output_layer28, output_layer28_13], dim=1)

        output_layer28_14 = F.relu(self.fc2814(output_layer27))
        output_layer28_15 = F.relu(self.fc2815(output_layer27_14))
        output_layer28_16 = F.relu(self.fc2816(output_layer27_15))

        # 第二十九层 LSTM层 特征映射
        output_layer29_1 = F.relu(self.fc2901(output_layer28_1))
        output_layer29 = output_layer29_1
        output_layer29_2 = F.relu(self.fc2902(output_layer28_2))
        output_layer29 = torch.cat([output_layer29, output_layer29_2], dim=1)
        output_layer29_3 = F.relu(self.fc2903(output_layer28_3))
        output_layer29 = torch.cat([output_layer29, output_layer29_3], dim=1)
        output_layer29_4 = F.relu(self.fc2904(output_layer28_4))
        output_layer29 = torch.cat([output_layer29, output_layer29_4], dim=1)
        output_layer29_5 = F.relu(self.fc2905(output_layer28_5))
        output_layer29 = torch.cat([output_layer29, output_layer29_5], dim=1)
        output_layer29_6 = F.relu(self.fc2906(output_layer28_6))
        output_layer29 = torch.cat([output_layer29, output_layer29_6], dim=1)
        output_layer29_7 = F.relu(self.fc2907(output_layer28_7))
        output_layer29 = torch.cat([output_layer29, output_layer29_7], dim=1)
        output_layer29_8 = F.relu(self.fc2908(output_layer28_8))
        output_layer29 = torch.cat([output_layer29, output_layer29_8], dim=1)
        output_layer29_9 = F.relu(self.fc2909(output_layer28_9))
        output_layer29 = torch.cat([output_layer29, output_layer29_9], dim=1)
        output_layer29_10 = F.relu(self.fc2910(output_layer28_10))
        output_layer29 = torch.cat([output_layer29, output_layer29_10], dim=1)
        output_layer29_11 = F.relu(self.fc2911(output_layer28_11))
        output_layer29 = torch.cat([output_layer29, output_layer29_11], dim=1)
        output_layer29_12 = F.relu(self.fc2912(output_layer28_12))
        output_layer29 = torch.cat([output_layer29, output_layer29_12], dim=1)
        output_layer29_13 = F.relu(self.fc2913(output_layer28_13))
        output_layer29 = torch.cat([output_layer29, output_layer29_13], dim=1)

        output_layer29_14 = F.relu(self.fc2914(output_layer28))
        output_layer29_15 = F.relu(self.fc2915(output_layer28_14))
        output_layer29_16 = F.relu(self.fc2916(output_layer28_15))
        output_layer29_17 = F.relu(self.fc2917(output_layer28_16))

        # 第三十层 LSTM层 特征映射
        output_layer30_1 = F.relu(self.fc3001(output_layer29_1))
        output_layer30 = output_layer30_1
        output_layer30_2 = F.relu(self.fc3002(output_layer29_2))
        output_layer30 = torch.cat([output_layer30, output_layer30_2], dim=1)
        output_layer30_3 = F.relu(self.fc3003(output_layer29_3))
        output_layer30 = torch.cat([output_layer30, output_layer30_3], dim=1)
        output_layer30_4 = F.relu(self.fc3004(output_layer29_4))
        output_layer30 = torch.cat([output_layer30, output_layer30_4], dim=1)
        output_layer30_5 = F.relu(self.fc3005(output_layer29_5))
        output_layer30 = torch.cat([output_layer30, output_layer30_5], dim=1)
        output_layer30_6 = F.relu(self.fc3006(output_layer29_6))
        output_layer30 = torch.cat([output_layer30, output_layer30_6], dim=1)
        output_layer30_7 = F.relu(self.fc3007(output_layer29_7))
        output_layer30 = torch.cat([output_layer30, output_layer30_7], dim=1)
        output_layer30_8 = F.relu(self.fc3008(output_layer29_8))
        output_layer30 = torch.cat([output_layer30, output_layer30_8], dim=1)
        output_layer30_9 = F.relu(self.fc3009(output_layer29_9))
        output_layer30 = torch.cat([output_layer30, output_layer30_9], dim=1)
        output_layer30_10 = F.relu(self.fc3010(output_layer29_10))
        output_layer30 = torch.cat([output_layer30, output_layer30_10], dim=1)
        output_layer30_11 = F.relu(self.fc3011(output_layer29_11))
        output_layer30 = torch.cat([output_layer30, output_layer30_11], dim=1)
        output_layer30_12 = F.relu(self.fc3012(output_layer29_12))
        output_layer30 = torch.cat([output_layer30, output_layer30_12], dim=1)
        output_layer30_13 = F.relu(self.fc3013(output_layer29_13))
        output_layer30 = torch.cat([output_layer30, output_layer30_13], dim=1)

        output_layer30_14 = F.relu(self.fc3014(output_layer29))
        output_layer30_15 = F.relu(self.fc3015(output_layer29_14))
        output_layer30_16 = F.relu(self.fc3016(output_layer29_15))
        output_layer30_17 = F.relu(self.fc3017(output_layer29_16))
        output_layer30_18 = F.relu(self.fc3018(output_layer29_17))

        # 第三十一层 线性层 特征映射
        output_layer31_1 = output_layer30_1.view(1, -1)
        output_layer31_1 = F.relu(self.fc3101(output_layer31_1))
        output_layer31 = output_layer31_1

        output_layer31_2 = output_layer30_2.view(1, -1)
        output_layer31_2 = F.relu(self.fc3102(output_layer31_2))
        output_layer31 = torch.cat([output_layer31, output_layer31_2], dim=1)

        output_layer31_3 = output_layer30_3.view(1, -1)
        output_layer31_3 = F.relu(self.fc3103(output_layer31_3))
        output_layer31 = torch.cat([output_layer31, output_layer31_3], dim=1)

        output_layer31_4 = output_layer30_4.view(1, -1)
        output_layer31_4 = F.relu(self.fc3104(output_layer31_4))
        output_layer31 = torch.cat([output_layer31, output_layer31_4], dim=1)

        output_layer31_5 = output_layer30_5.view(1, -1)
        output_layer31_5 = F.relu(self.fc3105(output_layer31_5))
        output_layer31 = torch.cat([output_layer31, output_layer31_5], dim=1)

        output_layer31_6 = output_layer30_6.view(1, -1)
        output_layer31_6 = F.relu(self.fc3106(output_layer31_6))
        output_layer31 = torch.cat([output_layer31, output_layer31_6], dim=1)

        output_layer31_7 = output_layer30_7.view(1, -1)
        output_layer31_7 = F.relu(self.fc3107(output_layer31_7))
        output_layer31 = torch.cat([output_layer31, output_layer31_7], dim=1)

        output_layer31_8 = output_layer30_8.view(1, -1)
        output_layer31_8 = F.relu(self.fc3108(output_layer31_8))
        output_layer31 = torch.cat([output_layer31, output_layer31_8], dim=1)

        output_layer31_9 = output_layer30_9.view(1, -1)
        output_layer31_9 = F.relu(self.fc3109(output_layer31_9))
        output_layer31 = torch.cat([output_layer31, output_layer31_9], dim=1)

        output_layer31_10 = output_layer30_10.view(1, -1)
        output_layer31_10 = F.relu(self.fc3110(output_layer31_10))
        output_layer31 = torch.cat([output_layer31, output_layer31_10], dim=1)

        output_layer31_11 = output_layer30_11.view(1, -1)
        output_layer31_11 = F.relu(self.fc3111(output_layer31_11))
        output_layer31 = torch.cat([output_layer31, output_layer31_11], dim=1)

        output_layer31_12 = output_layer30_12.view(1, -1)
        output_layer31_12 = F.relu(self.fc3112(output_layer31_12))
        output_layer31 = torch.cat([output_layer31, output_layer31_12], dim=1)

        output_layer31_13 = output_layer30_13.view(1, -1)
        output_layer31_13 = F.relu(self.fc3113(output_layer31_13))
        output_layer31 = torch.cat([output_layer31, output_layer31_13], dim=1)

        output_layer31_14 = F.relu(self.fc3114(output_layer30.view(1, -1)))
        output_layer31_15 = F.relu(self.fc3115(output_layer30_14.view(1, -1)))
        output_layer31_16 = F.relu(self.fc3116(output_layer30_15.view(1, -1)))
        output_layer31_17 = F.relu(self.fc3117(output_layer30_16.view(1, -1)))
        output_layer31_18 = F.relu(self.fc3118(output_layer30_17.view(1, -1)))
        output_layer31_19 = F.relu(self.fc3119(output_layer30_18.view(1, -1)))

        # 第三十二层 线性层 特征映射
        output_layer32_1 = F.relu(self.fc3201(output_layer31_1))
        output_layer32 = output_layer32_1
        output_layer32_2 = F.relu(self.fc3202(output_layer31_2))
        output_layer32 = torch.cat([output_layer32, output_layer32_2], dim=1)
        output_layer32_3 = F.relu(self.fc3203(output_layer31_3))
        output_layer32 = torch.cat([output_layer32, output_layer32_3], dim=1)
        output_layer32_4 = F.relu(self.fc3204(output_layer31_4))
        output_layer32 = torch.cat([output_layer32, output_layer32_4], dim=1)
        output_layer32_5 = F.relu(self.fc3205(output_layer31_5))
        output_layer32 = torch.cat([output_layer32, output_layer32_5], dim=1)
        output_layer32_6 = F.relu(self.fc3206(output_layer31_6))
        output_layer32 = torch.cat([output_layer32, output_layer32_6], dim=1)
        output_layer32_7 = F.relu(self.fc3207(output_layer31_7))
        output_layer32 = torch.cat([output_layer32, output_layer32_7], dim=1)
        output_layer32_8 = F.relu(self.fc3208(output_layer31_8))
        output_layer32 = torch.cat([output_layer32, output_layer32_8], dim=1)
        output_layer32_9 = F.relu(self.fc3209(output_layer31_9))
        output_layer32 = torch.cat([output_layer32, output_layer32_9], dim=1)
        output_layer32_10 = F.relu(self.fc3210(output_layer31_10))
        output_layer32 = torch.cat([output_layer32, output_layer32_10], dim=1)
        output_layer32_11 = F.relu(self.fc3211(output_layer31_11))
        output_layer32 = torch.cat([output_layer32, output_layer32_11], dim=1)
        output_layer32_12 = F.relu(self.fc3212(output_layer31_12))
        output_layer32 = torch.cat([output_layer32, output_layer32_12], dim=1)
        output_layer32_13 = F.relu(self.fc3213(output_layer31_13))
        output_layer32 = torch.cat([output_layer32, output_layer32_13], dim=1)

        output_layer32_14 = F.relu(self.fc3214(output_layer31))
        output_layer32_15 = F.relu(self.fc3215(output_layer31_14))
        output_layer32_16 = F.relu(self.fc3216(output_layer31_15))
        output_layer32_17 = F.relu(self.fc3217(output_layer31_16))
        output_layer32_18 = F.relu(self.fc3218(output_layer31_17))
        output_layer32_19 = F.relu(self.fc3219(output_layer31_18))
        output_layer32_20 = F.relu(self.fc3220(output_layer31_19))

        # 第三十三层 线性层 特征映射
        output_layer33_1 = F.relu(self.fc3301(output_layer32))

        output_layer33_2 = F.relu(self.fc3302(output_layer32_14))
        output_layer33_3 = F.relu(self.fc3303(output_layer32_15))
        output_layer33_3 = torch.cat([output_layer33_2, output_layer33_3], dim=1)

        output_layer33_4 = F.relu(self.fc3304(output_layer32_16))
        output_layer33_5 = F.relu(self.fc3305(output_layer32_17))
        output_layer33_6 = F.relu(self.fc3306(output_layer32_18))
        output_layer33_6 = torch.cat([output_layer33_6, output_layer33_5], dim=1)
        output_layer33_6 = torch.cat([output_layer33_6, output_layer33_4], dim=1)

        output_layer33_7 = F.relu(self.fc3307(output_layer32_19))
        output_layer33_8 = F.relu(self.fc3308(output_layer32_20))
        output_layer33_8 = torch.cat([output_layer33_8, output_layer33_7], dim=1)

        # 第三十四层 线性层 特征映射
        output_layer34_1 = F.relu(self.fc3401(output_layer33_1))
        output_layer34_2 = F.relu(self.fc3402(output_layer33_3))
        output_layer34_3 = F.relu(self.fc3403(output_layer33_6))
        output_layer34_4 = F.relu(self.fc3404(output_layer33_8))
        output_layer34_4 = torch.cat([output_layer34_4, output_layer34_3], dim=1)

        # 第三十五层 线性层 特征映射
        output_layer35_1 = self.fc3501(output_layer34_1)
        output_layer35_2 = self.fc3502(output_layer34_2)
        output_layer35_3 = self.fc3503(output_layer34_4)
        output_layer35 = torch.cat([output_layer35_1, output_layer35_2], dim=1)
        output_layer35 = torch.cat([output_layer35, output_layer35_3], dim=1)

        # 第三十六层 线性层 特征映射
        output_layer36 = self.fc36(output_layer35)

        # 输出层
        output = output_layer36
        return output


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


class Deep_Net_2_M1d_Res(nn.Module):
    def __init__(self):
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

        self.lstm40 = nn.LSTM(120, 60, 3)
        self.lstm41 = nn.LSTM(60, 12, 3)
        self.lstm42 = nn.LSTM(60, 12, 3)
        self.lstm43 = nn.LSTM(60, 12, 3)
        self.lstm44 = nn.LSTM(60, 12, 3)
        self.lstm45 = nn.LSTM(60, 12, 3)

        self.lstm50 = nn.LSTM(120, 60, 3)
        self.lstm51 = nn.LSTM(72, 12, 3)
        self.lstm52 = nn.LSTM(72, 12, 3)
        self.lstm53 = nn.LSTM(72, 12, 3)
        self.lstm54 = nn.LSTM(72, 12, 3)
        self.lstm55 = nn.LSTM(72, 12, 3)

        self.fc60 = nn.Linear(60, 60)
        self.fc61 = nn.Linear(12, 12)
        self.fc62 = nn.Linear(12, 12)
        self.fc63 = nn.Linear(12, 12)
        self.fc64 = nn.Linear(12, 12)
        self.fc65 = nn.Linear(12, 12)

        self.lstm7 = nn.LSTM(120, 60, 3)  # 输入数据350个特征维度，500个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果  输入[x,y,180]  输出[x,y,60]

        self.fc8 = nn.Linear(60, 60)  # 线性拟合，输入[x,60]  输出[x,60]
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
        output_layer5_1 += output_layer4_1

        layer_in_5_2 = torch.cat([output_layer4_0, output_layer4_2], dim=2)
        output_layer5_2, temp5_2 = self.lstm52(layer_in_5_2)
        output_layer5_2 += output_layer4_2

        layer_in_5_3 = torch.cat([output_layer4_0, output_layer4_3], dim=2)
        output_layer5_3, temp5_3 = self.lstm53(layer_in_5_3)
        output_layer5_3 += output_layer4_3

        layer_in_5_4 = torch.cat([output_layer4_0, output_layer4_4], dim=2)
        output_layer5_4, temp5_4 = self.lstm54(layer_in_5_4)
        output_layer5_4 += output_layer4_4

        layer_in_5_5 = torch.cat([output_layer4_0, output_layer4_5], dim=2)
        output_layer5_5, temp5_5 = self.lstm55(layer_in_5_5)
        output_layer5_5 += output_layer4_5

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
    def __init__(self):
        super(BSsequential_net, self).__init__()

        self.conv1 = TripleConv1d(6, 8)
        self.conv2 = QuadrupleConv1d(8, 10)
        self.conv3 = TripleConv1d(10, 12)
        self.fc30 = nn.Linear(12 * 60, 12 * 60)
        self.conv4 = QuadrupleConv1d(12, 14)
        self.conv5 = TripleConv1d(14, 16)
        self.conv6 = QuadrupleConv1d(16, 18)
        self.fc60 = nn.Linear(18 * 60, 18 * 60)
        self.conv7 = TripleConv1d(18, 20)
        self.conv8 = QuadrupleConv1d(20, 18)
        self.conv9 = TripleConv1d(18, 16)
        self.fc90 = nn.Linear(16 * 60, 16 * 60)
        self.conv10 = QuadrupleConv1d(16, 14)
        self.conv11 = TripleConv1d(14, 12)
        self.conv12 = QuadrupleConv1d(12, 10)
        self.fc120 = nn.Linear(10 * 60, 10 * 60)
        self.conv13 = TripleConv1d(10, 8)
        self.conv14 = QuadrupleConv1d(8, 6)
        self.conv15 = TripleConv1d(6, 4)
        self.fc150 = nn.Linear(4 * 60, 4 * 60)
        self.conv16 = QuadrupleConv1d(4, 2)
        self.conv17 = TripleConv1d(2, 1)
        self.fc170 = nn.Linear(60, 60)  # 线性拟合，输入[x,60]  输出[x,60]
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
        input3_0 = output3.view(1, -1)
        output3_0 = self.fc30(input3_0)
        output3_0 = output3_0 + input3_0
        output3_0 = output3_0.view(1, -1, 60)

        output4 = self.conv4(output3_0)
        output5 = self.conv5(output4)
        output6 = self.conv6(output5)
        input6_0 = output6.view(1, -1)
        output6_0 = self.fc60(input6_0)
        output6_0 = output6_0 + input6_0
        output6_0 = output6_0.view(1, -1, 60)

        output7 = self.conv7(output6_0)
        output8 = self.conv8(output7) + output6
        output9 = self.conv9(output8) + output5
        input9_0 = output9.view(1, -1)
        output9_0 = self.fc90(input9_0)
        output9_0 = output9_0 + input9_0
        output9_0 = output9_0.view(1, -1, 60)

        output10 = self.conv10(output9_0) + output4
        output11 = self.conv11(output10) + output3
        output12 = self.conv12(output11) + output2
        input12_0 = output12.view(1, -1)
        output12_0 = self.fc120(input12_0)
        output12_0 = output12_0 + input12_0
        output12_0 = output12_0.view(1, -1, 60)

        output13 = self.conv13(output12_0) + output1
        output14 = self.conv14(output13) + input1
        output15 = self.conv15(output14)
        input15_0 = output15.view(1, -1)
        output15_0 = self.fc150(input15_0)
        output15_0 = output15_0 + input15_0
        output15_0 = output15_0.view(1, -1, 60)

        output16 = self.conv16(output15_0)
        output17 = self.conv17(output16)
        input17_0 = output17.view(1, -1)
        output18 = self.fc170(input17_0)
        output = output18

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
    def __init__(self):
        super(BSsequential_net_seismic, self).__init__()

        self.conv1 = TripleConv1d(1, 2)
        self.conv2 = QuadrupleConv1d(2, 4)
        self.conv3 = TripleConv1d(4, 6)
        self.fc30 = nn.Linear(6 * 60, 6 * 60)
        self.conv4 = QuadrupleConv1d(6, 8)
        self.conv5 = TripleConv1d(8, 10)
        self.conv6 = QuadrupleConv1d(10, 12)
        self.fc60 = nn.Linear(12 * 60, 12 * 60)
        self.conv7 = TripleConv1d(12, 10)
        self.conv8 = QuadrupleConv1d(10, 8)
        self.conv9 = TripleConv1d(8, 6)
        self.fc90 = nn.Linear(6 * 60, 6 * 60)
        self.conv10 = QuadrupleConv1d(6, 4)
        self.conv11 = TripleConv1d(4, 2)
        self.conv12 = QuadrupleConv1d(2, 1)
        self.fc120 = nn.Linear(1 * 60, 1 * 60)
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

    def forward(self, seismic):

        # 0727的按厍师兄的想法，设计一个顺序的残差N型网
        input1 = seismic.view(1, 1, -1)

        output1 = self.conv1(input1)
        output2 = self.conv2(output1)
        output3 = self.conv3(output2)
        input3_0 = output3.view(1, -1)
        output3_0 = self.fc30(input3_0)
        output3_0 = output3_0 + input3_0
        output3_0 = output3_0.view(1, -1, 60)

        output4 = self.conv4(output3_0)
        output5 = self.conv5(output4)
        output6 = self.conv6(output5)
        input6_0 = output6.view(1, -1)
        output6_0 = self.fc60(input6_0)
        output6_0 = output6_0 + input6_0
        output6_0 = output6_0.view(1, -1, 60)

        output7 = self.conv7(output6_0) + output5
        output8 = self.conv8(output7) + output4
        output9 = self.conv9(output8) + output3
        input9_0 = output9.view(1, -1)
        output9_0 = self.fc90(input9_0)
        output9_0 = output9_0 + input9_0
        output9_0 = output9_0.view(1, -1, 60)

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

