import math

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



# 超参数
train_batch_size = 64
test_batch_size = 1000

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False,
                              download=True, transform=transform)
# 下面是将张量分组，每一组大小为batch_size，然后可以用for in 循环遍历
train_loader = DataLoader(train_dataset, train_batch_size,
                          shuffle=True)
test_loader = DataLoader(test_dataset, test_batch_size,
                         shuffle=False)
# 构建模型
class My_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(My_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        # 如果经过该模块，形状大小发生改变，则加上残差时也需要将tensor形状改变，
        # 具体来说就是channel不一致，或则宽高不一致
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.block1 = My_Block(1, 16, 1)
        # 28
        self.block2 = My_Block(16, 32, 2)
        # 14
        self.block3 = My_Block(32, 64, 2)
        # 7
        self.fc = nn.Linear(64*7*7, 10)
    def forward(self, x):
        x = self.block1.forward(x)
        x = self.block2.forward(x)
        x = self.block3.forward(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
# 训练模型
def train(model,train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # print("")
        # 分成多少组就做多少次
        # print("batch_idx:",batch_idx)
        optimizer.zero_grad()
        # print(data)
        # print(data.shape)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    model.eval()
    train_loss = 0
    train_correct = 0
    # 计算每个数字的正确率,分别表示正确的和总的
    correct_number = [0]*10
    sum_number = [0]*10
    with torch.no_grad():
        for data, target in train_loader:
            # 分成多少组就循环多少次
            # 目前而言每组里面64张特征图
            output = model(data)
            train_loss += criterion(output, target).item()
            result = output.argmax(dim=1, keepdim=True)
            target = target.view_as(result)
            # 统计每个数字出现的次数以及正确率
            for i in range(result.size(0)):
                for j in range(result.size(1)):
                    ans = int(result[i, j])
                    index = int(target[i, j])
                    sum_number[index] += 1
                    correct_number[index] += int(ans == index)
            # print("sum_number:",sum_number)
            # print("correct_number:",correct_number)
            train_correct += result.eq(target).sum().item()
    train_loss /= math.ceil(len(train_loader.dataset)/train_batch_size)
    # len(train_loader.dataset)仍是所有数据的大小，只有用in才能取出一个一个的batch
    # 应该是这个对象的特殊构造
    # print("trian_loader_len",len(train_loader.dataset))
    train_accuracy = 100.0 * train_correct / len(train_loader.dataset)
    print('Train Loss: {:.4f}, Accuracy: {:.2f}%'.
          format(train_loss, train_accuracy))
    for i in range(10):
        print("    the number {} accuracy is {:.4f}".format(i, correct_number[i]/sum_number[i]))

# 测试模型
criterion = nn.CrossEntropyLoss()

def test(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # 计算每个数字的正确率,分别表示正确的和总的
    correct_number = [0]*10
    sum_number = [0]*10
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            result = output.argmax(dim=1, keepdim=True)
            target = target.view_as(result)
            # 统计每个数字出现的次数以及正确率
            for i in range(result.size(0)):
                for j in range(result.size(1)):
                    ans = int(result[i, j])
                    index = int(target[i, j])
                    sum_number[index] += 1
                    correct_number[index] += int(ans == index)
            correct += result.eq(target).sum().item()
    test_loss /= math.ceil(len(test_loader.dataset)/test_batch_size)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print('Test Loss: {:.4f}, Accuracy: {:.2f}%'.
          format(test_loss, accuracy))
    for i in range(10):
        print("    the number {} accuracy is {:.4f}".format(i, correct_number[i]/sum_number[i]))


model = ResNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1, 11):
    print('Epoch: {}'.format(epoch))
    train(model,  train_loader, optimizer, criterion)
    test(model, test_loader)
