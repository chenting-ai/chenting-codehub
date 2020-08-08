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

BATCH_SIZE = 10  # BATCH_SIZE大小
EPOCHS = 1  # 总共训练批次
lr = 0.01  # 学习步长

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多

# 加载训练用地震数据
dataFile = 'D://python_project/Predict_Impedence/Out_Predictdata/seismic_well52_len30.mat'
seismicdata = scipio.loadmat(dataFile)
train_seismicdata = seismicdata['seismic_well52_len30']

# 加载训练用测井数据
labFile = 'D://python_project/Predict_Impedence/Out_Predictdata/well_well52_len30.mat'
welldata = scipio.loadmat(labFile)
train_welldata = welldata['well_well52_len30']

# 加载测试用地震数据
testFile = 'D://python_project/Predict_Impedence/Out_Predictdata/inline99_seismic_len30.mat'
seismic = scipio.loadmat(testFile)
test_seismic = seismic['inline99_seismic_len30']


class MyDataset(data.Dataset):
    def __init__(self, data, target):
        self.data = np.array(data)
        self.target = np.array(target)

    def __getitem__(self, index):  # 返回的是tensor
        d, target = self.data[index, :], self.target[index, :]
        d = torch.from_numpy(d)
        target = torch.from_numpy(target)
        return d, target

    def __len__(self):
        return self.data.shape[0]


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 50)  # （输入channels,输出channels,kernel_size 5*5）
        self.fc2 = nn.Linear(50, 30)  # （输入channels,输出channels,kernel_size 5*5）

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out


def train():
    model = ConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = MyDataset(train_seismicdata, train_welldata)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=3, shuffle=True, drop_last=False)

    writer = SummaryWriter(log_dir='train_logs3')

    for epoch in range(1, EPOCHS + 1):

        print(epoch)
        epoch_loss = []

        for itr, (train_dt, train_lable) in enumerate(train_dataloader):
            train_dt, train_lable = train_dt.to(device), train_lable.to(device)
            optimizer.zero_grad()
            train_dt = train_dt.float()
            writer.add_graph(model, (train_dt,))
            output = model(train_dt)
            train_lable = train_lable.float()
            loss = F.mse_loss(output, train_lable)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        epoch_loss = np.nanmean(np.array(epoch_loss))
        writer.add_scalar('train/loss', epoch_loss, epoch)
        print('Train set: Average loss: {:.4f}'.format(epoch_loss))
    writer.close()
    path = './model_file/Predict_Impedence_welldata.tar'
    torch.save(model, path)


def test():
    model = ConvNet().to(device)

    test_dataset = MyDataset(test_seismic, test_seismic)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=3, shuffle=True, drop_last=False)
    test_impedence = torch.zeros(torch.is_same_size(test_seismic), dtype='float')
    for itr, (test_dt, test_lable) in enumerate(test_dataloader):
        test_dt, test_lable = test_dt.to(device), test_lable.to(device)
        test_dt = test_dt.float()
        output = model(test_dt)
        test_impedence[itr, :] = output
    pathtxt = './output/Predict_Impedence.txt'
    np.savetxt(pathtxt, np.array(test_impedence))
    pathmat = './output/Predict_Impedence.mat'
    scipio.savemat(pathmat, {'Predict_Impedence': np.array(test_impedence)})


if __name__ == '__main__':
    train()
    test()


