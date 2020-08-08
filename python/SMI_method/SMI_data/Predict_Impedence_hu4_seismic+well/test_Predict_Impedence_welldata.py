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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
writer = SummaryWriter('run')

# 加载训练用地震数据
dataFile = 'D://python_project/Predict_Impedence/Out_Predictdata/seismic_well52_len30.mat'
seismicdata = scipio.loadmat(dataFile)
train_seismicdata = seismicdata['seismic_well52_len30']

# 加载训练用测井数据
labFile = 'D://python_project/Predict_Impedence/Out_Predictdata/well_well52_len30.mat'
welldata = scipio.loadmat(labFile)
train_welldata = welldata['well_well52_len30']

# 加载训练用测井数据
testFile = 'D://python_project/Predict_Impedence/Out_Predictdata/inline99_seismic_len30.mat'
seismic = scipio.loadmat(testFile)
test_seismic = seismic['inline99_seismic_len30']

x_transform = transforms.Compose(transforms = [transforms.ToTensor()])
y_transform = transforms.Compose(transforms = [transforms.ToTensor()])


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


def train(model, device, train_dt, optimizer):
    model.train()
    tr_loss = 0
    for itr, (x, y) in enumerate(train_dt):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        x = x.float()
        output = model(x)
        y = y.float()
        # print(y)
        loss = F.mse_loss(output, y)
        tr_loss += F.mse_loss(output, y, reduction='sum').item()  # 将一批的损失相加
        loss.backward()
        optimizer.step()
        writer.add_scalar('train/loss', loss.item(), itr)
        # print('The iter %d loss: %0.2f'%(itr, loss.item()))
    tr_loss /= itr
    print('Train set: Average loss: {:.4f}'.format(tr_loss))
    return tr_loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for itr2, (data2, target) in enumerate(test_loader):
            data2, target = data2.to(device), target.to(device)
            data2 = data2.float()
            output = model(data2)
            target = target.float()
            test_loss += F.mse_loss(output, target, reduction='sum').item()  # 将一批的损失相加
        test_loss /= itr2
        print('Test set: Average loss: {:.4f}'.format(test_loss))


if __name__ == '__main__':
    model = ConvNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_dataset = MyDataset(train_seismicdata, train_welldata)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=3, shuffle=True, drop_last=False)
    test_dataset = MyDataset(train_seismicdata, train_welldata)
    test_trainloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=3, shuffle=True, drop_last=False)
    for epoch in range(1, EPOCHS + 1):
        print(epoch)
        train(model, DEVICE, train_dataloader, optimizer)
        test(model, DEVICE, test_trainloader)
    PATH = './model_file/Predict_Impedence_welldata.tar'
    torch.save(model, PATH)
    PATH = './model_file/Predict_Impedence_welldata.tar'
    model2 = torch.load(PATH)

