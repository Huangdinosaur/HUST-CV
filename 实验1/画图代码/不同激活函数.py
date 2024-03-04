import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 读取CSV文件
data_frame = pd.read_csv('./dataset.csv')
matric = data_frame.values
matric[:,-1] -= 1
np.random.shuffle(matric)
# print(matric)
# 提取特征和标签列
train_data = matric[:3600]
test_data = matric[3600:]
class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
# get loss function

class m1(nn.Module):
    def __init__(self):
        super(m1, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.relu = nn.Sigmoid()
        self.fc2 = nn.Linear(8, 4)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
class m2(nn.Module):
    def __init__(self):
        super(m2, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.relu = nn.Tanh()
        self.fc2 = nn.Linear(8, 4)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
# 设置参数
input_size = 2
hidden_size = 128
num_classes = 4
learning_rate = 0.0015
num_epochs = 100
batch_size = 400
test_batch_size = 400
# 数据转换为张量，然后转化为dataloader
train_data = torch.from_numpy(train_data).float()

train_feature = DataLoader(train_data[:,:-1], batch_size,
                          shuffle=False)
# print(len(train_feature.dataset))
train_labels = DataLoader(train_data[:,-1], batch_size,
                          shuffle=False)

test_data = torch.from_numpy(test_data).float()

test_feature = DataLoader(test_data[:,:-1], test_batch_size,
                          shuffle=False)
# print(test_feature.dataset)
test_labels = DataLoader(test_data[:,-1], test_batch_size,
                          shuffle=False)



def train_func(model,criterion,optimizer):
    loss_values = []
    accuracy_values = []
    test_loss_value = []
    test_accuracy_value = []
    for epoch in range(num_epochs):
        losses = []
        correct = []
        for data, target in zip(train_feature,train_labels):
            # print(data)
            # 转化为long
            target = target.long()
            optimizer.zero_grad()
            outputs = model(data)
            # 转化成 [[0],[1],[2],...,[0]] 形式，可使用print打印查看
            outputs_label = outputs.argmax(dim=1, keepdim=True)
            # 一个复杂的函数，判断对应位的数值是否相等，然后求和
            correct.append(outputs_label.eq(target.view_as(outputs_label)).sum().item())
            loss = criterion(outputs, target)
            loss.backward()  # 计算梯度，储存在grad中
            optimizer.step()  # 更新参数
            losses.append(loss.item())
        print('Train Epoch: {} Loss: {:.6f} accuracy: {:.6f}'.format(epoch+1, np.mean(losses),
                                                                     sum(correct)/len(correct)/batch_size))
        loss_values.append(np.mean(losses))
        accuracy_values.append(sum(correct)/len(correct)/batch_size)
        # print(type(correct))
        # print("correct:",correct)
        # 设置模型为评估模式
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in zip(test_feature,test_labels):
                target = target.long()
                output = model(data)
                test_loss += criterion(output, target).item()
                # 获得[[1],[2],...,[3]]
                pred = output.argmax(dim=1, keepdim=True)
                # 转化对比，sum ，所以后面需要除以test_data.size(0)
                correct += pred.eq(target.view_as(pred)).sum().item()
        # 注意loss和accuracy的叠加次数
        test_loss /= (400/test_batch_size)
        accuracy = 100.0 * correct / test_data.size(0)
        test_loss_value.append(test_loss)
        test_accuracy_value.append(accuracy)
        print('Test Loss: {:.6f}, Accuracy: {:.6f}%'.
              format(test_loss, accuracy))

    return loss_values,accuracy_values,test_loss_value,test_accuracy_value

# model0
model = FeedForwardNet(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss1_train,accuracy1_train,loss1_test,accuracy1_test = train_func(model,criterion,optimizer)
# model1
model = m1()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss2_train,accuracy2_train,loss2_test,accuracy2_test = train_func(model,criterion,optimizer)

# model2
model = m2()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss3_train,accuracy3_train,loss3_test,accuracy3_test = train_func(model,criterion,optimizer)
# 创建一个图形对象


fig, ax = plt.subplots()

# 绘制损失曲线
ax.plot(loss1_test, label='Loss1')
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Loss')
ax.plot(loss2_test, label='Loss2')
ax.plot(loss3_test, label='Loss3')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(accuracy1_test, label='Accuracy1')
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Accuracy')
ax.plot(accuracy2_test, label='Accuracy2')
ax.plot(accuracy3_test, label='Accuracy3')
ax.legend()
plt.show()


fig, ax = plt.subplots()

# 绘制损失曲线
ax.plot(loss1_train, label='Loss1')
ax.set_xlabel('Epoch')
ax.set_ylabel('Train Loss')
ax.plot(loss2_train, label='Loss2')
ax.plot(loss3_train, label='Loss3')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(accuracy1_train, label='Accuracy1')
ax.set_xlabel('Epoch')
ax.set_ylabel('Train Accuracy')
ax.plot(accuracy2_train, label='Accuracy2')
ax.plot(accuracy3_train, label='Accuracy3')
ax.legend()
plt.show()

