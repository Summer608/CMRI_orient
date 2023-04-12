import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from model.CMRNet import CMRNet
from util.dataloader import dataloader
from d2l import torch as d2l
from tqdm import tqdm
import os

from torchvision.models import resnet18, resnet50

root = 'data/data0/T2'

PATH = 'checkpoints/C0/model-best.pth' #模型权重

# 冻结阶段训练参数，learning_rate和batch_size可以设置大一点
Init_Epoch          = 0
Freeze_Epoch        = 20
Freeze_batch_size   = 32
Freeze_lr           = 1e-3
# 解冻阶段训练参数，learning_rate和batch_size设置小一点
UnFreeze_Epoch      = 40
Unfreeze_batch_size = 16
Unfreeze_lr         = 1e-4


model = CMRNet()

# model = resnet18()
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 8)

# model = resnet50()
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 8)

model.load_state_dict(torch.load(PATH)) #加载预训练模型

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('training on:', device)
model.to(device) #放到GPU上

criterion = nn.CrossEntropyLoss()


# 冻结全连接层之前的训练

Freeze_Train= True
batch_size  = Freeze_batch_size
lr          = Freeze_lr
start_epoch = Init_Epoch
end_epoch   = Freeze_Epoch

#加载数据集
train_dataset = dataloader(root=root, mode='train', truncation=True)
load_train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) #win系统numworker设置为0

valid_dataset = dataloader(root=root, mode='valid', truncation=True)
load_valid_dataset = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

features = nn.Sequential(*list(model.children())[:10])
if Freeze_Train:
    for param in features.parameters():
        param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_acc = 0
# 训练模型
for epoch in range(start_epoch, end_epoch):
    train_loss = 0.0
    train_correct = 0
    model.train()  # 将模型设置为训练模式

    for inputs, labels in load_train_dataset:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计训练损失和正确分类的数量
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()

    # 计算训练集的平均损失和准确率
    train_loss = train_loss / len(train_dataset)
    train_acc = train_correct / len(train_dataset)

    # 在验证集上测试模型
    val_loss = 0.0
    val_correct = 0
    model.eval()  # 将模型设置为评估模式

    with torch.no_grad():  # 禁用梯度计算，减少内存占用和加速计算
        for inputs, labels in load_valid_dataset:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 统计验证损失和正确分类的数量
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()

    # 计算验证集的平均损失和准确率
    val_loss = val_loss / len(valid_dataset)
    val_acc = val_correct / len(valid_dataset)

    if val_acc > best_acc:
        torch.save(model.state_dict(), "./checkpoints/T2/model-best.pth")
        best_acc = val_acc
    torch.save(model.state_dict(), "./checkpoints/T2/model-latest.pth")

    print(f'Epoch [{epoch + 1}/{end_epoch}], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.6f}')


# 解冻后再训练整个网络

Freeze_Train=False
batch_size  = Unfreeze_batch_size
lr          = Unfreeze_lr
start_epoch = Freeze_Epoch
end_epoch   = UnFreeze_Epoch

train_dataset = dataloader(root=root, mode='train', truncation=True)
load_train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) #win系统numworker设置为0

valid_dataset = dataloader(root=root, mode='valid', truncation=True)
load_valid_dataset = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

if not Freeze_Train:
    for param in features.parameters():
        param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 训练模型
for epoch in range(start_epoch, end_epoch):
    train_loss = 0.0
    train_correct = 0
    model.train()  # 将模型设置为训练模式

    for inputs, labels in load_train_dataset:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计训练损失和正确分类的数量
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()

    # 计算训练集的平均损失和准确率
    train_loss = train_loss / len(train_dataset)
    train_acc = train_correct / len(train_dataset)

    # 在验证集上测试模型
    val_loss = 0.0
    val_correct = 0
    model.eval()  # 将模型设置为评估模式

    with torch.no_grad():  # 禁用梯度计算，减少内存占用和加速计算
        for inputs, labels in load_valid_dataset:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 统计验证损失和正确分类的数量
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()

    # 计算验证集的平均损失和准确率
    val_loss = val_loss / len(valid_dataset)
    val_acc = val_correct / len(valid_dataset)

    if val_acc > best_acc:
        torch.save(model.state_dict(), "./checkpoints/T2/model-best.pth")
        best_acc = val_acc
    torch.save(model.state_dict(), "./checkpoints/T2/model-latest.pth")

    # 输出训练和验证信息
    print(f'Epoch [{epoch+1}/{end_epoch}], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.6f}')

print('Training finished.')

