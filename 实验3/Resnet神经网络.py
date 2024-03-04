import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import random
import matplotlib.pyplot as plt

# 定义数据预处理的转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载完整的MNIST数据集
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

# 计算10%的样本数量
train_subset_size = int(0.1 * len(train_dataset))
test_subset_size = int(0.1 * len(test_dataset))

# 从训练集中随机选择10%的样本
train_subset = Subset(train_dataset,
                      # torch.randperm创建随机序列索引,然后取前trian_subset_size个,
                      # 最后利用subset函数取处随机的10%样本数
                      torch.randperm(len(train_dataset))[:train_subset_size]
                      )
# 从测试集中随机选择10%的样本
test_subset = Subset(test_dataset,
                     torch.randperm(len(test_dataset))[:test_subset_size]
                     )
# 将随机选取的10%数据分成两份,用于训练
train_subset_index = torch.randperm(len(train_subset))
# print(train_subset_index)
# print(train_subset_index[:int(len(train_subset)/2)])
# print(train_subset_index[int(len(train_subset)/2):])
train_subset1 = Subset(train_subset,train_subset_index[:int(len(train_subset)/2)])
train_subset2 = Subset(train_subset,train_subset_index[int(len(train_subset)/2):])
# 将随机选取的10%数据分成两份，用于测试
test_subset_index = torch.randperm(len(test_subset))
test_subset1 = Subset(test_subset,test_subset_index[:int(len(test_subset)/2)])
test_subset2 = Subset(test_subset,test_subset_index[int(len(test_subset)/2):])
# 定义批量大小
train_batch_size = 100
test_batch_size = 100
# 创建训练数据加载器
train_loader1 = DataLoader(train_subset1, batch_size=train_batch_size, shuffle=True, drop_last=True)
train_loader2 = DataLoader(train_subset2, batch_size=train_batch_size, shuffle=True, drop_last=True)
# 创建测试数据加载器
test_loader1 = DataLoader(test_subset1, batch_size=test_batch_size, shuffle=False, drop_last=True)
test_loader2 = DataLoader(test_subset2, batch_size=test_batch_size, shuffle=False, drop_last=True)


# 定义模块
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
# 定义Network模型
class MyJudgNet(nn.Module):
    def __init__(self):
        super(MyJudgNet, self).__init__()
        self.block1 = My_Block(2, 16, 1)
        # 28
        self.block2 = My_Block(16, 32, 2)
        # 14
        self.block3 = My_Block(32, 64, 2)
        # 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.block1.forward(x)
        x = self.block2.forward(x)
        x = self.block3.forward(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x



model = MyJudgNet()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



loss_train = []
accuracy_train = []
loss_test = []
accuracy_test = []
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0.0
    for (data1, target1), (data2, target2) in zip(train_loader1, train_loader2):
        optimizer.zero_grad()
        # 将两张图叠在一起,形成输入数据
        data = torch.cat((data1,data2), dim=1)
        # print(data.shape)
        output = model(data)
        # 将目标准换位64*1
        target1 = target1.reshape(len(target1), -1)
        target2 = target2.reshape(len(target2), -1)
        # 形成64*2形状
        target = torch.cat(((target1 == target2).int(), (target1 != target2).int()),1)
        # print(target)
        # print(target.shape)
        # 第一维表示true,第二维表示false
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # 计算正确率
        # 返回最大值下标
        pred = output.argmax(dim=1)
        ans = target.argmax(dim=1)
        # print("pred:",pred)
        # print("ans:",ans)
        ans = (pred == ans).int()
        correct += ans.sum().item()
        # print("ans:",ans)
    train_loss /= len(train_loader1)
    loss_train.append(train_loss)
    accuracy_train.append(correct/(train_batch_size*len(train_loader1)))
    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.6f}, '
          f'Accuracy: {correct/(train_batch_size*len(train_loader1)):.6f}')
    # 在测试集上评估模型
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for (data1, target1), (data2, target2) in zip(test_loader1, test_loader2):
            data = torch.cat((data1, data2), dim=1)
            output = model(data)
            target1 = target1.reshape(len(target1), -1)
            target2 = target2.reshape(len(target2), -1)
            target = torch.cat(((target1 == target2).int(), (target1 != target2).int()), 1)
            loss = criterion(output, target.float())
            test_loss += loss.item()
            pred = output.argmax(dim=1)
            ans = target.argmax(dim=1)
            ans = (pred == ans).int()
            correct += ans.sum().item()
        test_loss /= len(test_loader1)
        loss_test.append(test_loss)
        accuracy_test.append(correct / (test_batch_size * len(test_loader1)))
        print(f'                Test Loss: {test_loss:.6f},'
              f' Accuracy: {correct / (test_batch_size * len(test_loader1)):.6f}')


fig, ax = plt.subplots()
ax.plot(loss_train, label='Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Train Loss')
ax.legend()
plt.show()
fig, ax = plt.subplots()
ax.plot(accuracy_train, label='accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Train accuracy')
ax.legend()
plt.show()


fig, ax = plt.subplots()
ax.plot(loss_test, label='Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Loss')
ax.legend()
plt.show()
fig, ax = plt.subplots()
ax.plot(accuracy_test, label='accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Test accuracy')
ax.legend()
plt.show()