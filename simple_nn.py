'''    自己实现手写数字      '''

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 硬件加速
torch.manual_seed(40)


# CUDA 可用就使用 GPU，否则使用 CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("计算设备：", device)

# 数据准备
transform =transforms.Compose(
    [
        transforms.ToTensor(),                        #转化为张量
        transforms.Normalize((0.1307,),(0.3081,))     #归一化
    ]
)

train_set = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,  #预处理，将每一个训练集图片按照transform流程处理
)

# 加载数据，分批次，一批128个图片 
train_loader = DataLoader(  
    train_set,
    batch_size=128,
    shuffle=True,
    num_workers=0,
)

# 重载神经网络
class simplemlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(784,128)
        self.fc2=nn.Linear(128,10)
    def forward(self, images):
        # 每张图片展开成 1维 784 个像素
        pixels = torch.flatten(images, start_dim=1)

        # 第一层：784 → 128
        hidden = self.fc1(pixels)
        hidden = Func.relu(hidden)

        # 第二层：128 → 10
        logits = self.fc2(hidden)

        return logits


model = simplemlp().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
total_epoch=8
for epoch in range(total_epoch):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for x_input, y_label in train_loader:
        # 把这一批图片和标签放到 GPU 上
        x_input = x_input.to(device)
        y_label = y_label.to(device)

        optimizer.zero_grad()

        logits = model(x_input)
        loss = loss_fn(logits, y_label)

        loss.backward()
        optimizer.step()  # 更新参数

        # 统计损失
        batch_size = len(y_label)
        batch_average_loss = loss.item()
        batch_total_loss = batch_average_loss * batch_size

        total_loss = total_loss + batch_total_loss

        # 一次比较整批图片的预测结果
        predicted_labels = torch.argmax(logits, dim=1)
        is_correct = predicted_labels == y_label
        batch_correct = is_correct.sum().item()

        correct = correct + batch_correct
        total = total + batch_size

    average_loss = total_loss / total
    accuracy = correct / total

    print(
        f"第 {epoch + 1}/{total_epoch} 轮"
        f" | 训练损失：{average_loss:.4f}"
        f" | 训练准确率：{accuracy:.2%}"
    )

#-------------------------------------------------

# 1. 准备测试集
test_set = datasets.MNIST(
    root="./data",
    train=False,          # False 表示读取测试集
    download=True,
    transform=transform, # 与训练集使用相同的预处理
)

test_loader = DataLoader(
    test_set,
    batch_size=256,
    shuffle=False,
    num_workers=0,
)

# 2. 切换到评估模式
model.eval()

correct = 0
total = 0

# 3. 测试时不需要记录梯度
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        scores = model(images)
        predicted_labels = torch.argmax(scores, dim=1)

        # 统计这一批正确识别的数量
        is_correct = predicted_labels == labels
        batch_correct = is_correct.sum().item()

        correct = correct + batch_correct
        total = total + len(labels)
# 4. 打印测试结果
accuracy_percent = correct / total * 100

print("测试图片数量：", total)
print("正确识别数量：", correct)
print("测试准确率：", round(accuracy_percent, 2), "%")