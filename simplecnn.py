"""简单 CNN 识别 CIFAR-10 图片，输入尺寸为 3×32×32。"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import math
from torch.optim.lr_scheduler import LambdaLR

torch.manual_seed(42)

if torch.cuda.is_available():
    device_name = "cuda"
else:
    device_name = "cpu"
device = torch.device(device_name)


def main():
    print("计算设备：", device)

    # 1. 数据准备
    transform_0 = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4), #数据增强----随机改变图片的位置与方向，增加随机性
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    transform_1 = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    # 全部的训练数据（50000 张 RGB 图片）
    train_set = datasets.CIFAR10(
        root="./data",
        download=True,
        train=True,
        transform=transform_0,
    )

    # 独立测试集（10000 张图片）
    test_set = datasets.CIFAR10(
        root="./data",
        download=True,
        train=False,
        transform=transform_1,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # 2. 网络结构：两组 Conv-ReLU-Pool，最后一个全连接层
    model = nn.Sequential(
        # 第一组：3×32×32 → 32×16×16
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),   #16个卷积核
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  #缩减一半
        nn.Dropout2d(0.1),

        # 第二组：32×16×16 → 64×8×8
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(0.1),

        #第三组  64×8×8 →64×4×4
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(0.1),


        # 展平为 2048 个特征，输出 10 个类别的原始分数
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(256, 10)
    ).to(device)
    epoch = 30
    # 3. 损失函数与优化器与学习率调度
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3,weight_decay=1e-4)

    update_steps=epoch*len(train_loader) 

    warmup_steps=int(0.1*update_steps)
    def lr_update(steps):
            res=0
            if steps<warmup_steps:
                res=(steps+1)/warmup_steps
            else :
                temp=(steps-warmup_steps)/max(1,update_steps-warmup_steps) # 0~1
                res=0.5*(1+math.cos(math.pi*temp)) # cosx (0~pi)属于（-1，1），+1 /2 范围为（1~0）
            return res

    scheduler=LambdaLR(optimizer,lr_lambda=lr_update)
    # 4. 开始训练
    for i in range(epoch):
        model.train()

        total_loss = 0.0
        right_num = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()        # 梯度清零
            logits = model(xb)           # 前向传播
            loss = loss_fn(logits, yb)    # 计算损失
            loss.backward()              # 反向传播
            optimizer.step()             # 更新权重参数
            scheduler.step()             # 更新学习率

            batch_size = len(yb)
            total_loss += loss.item() * batch_size
            right_num += (logits.argmax(dim=1) == yb).sum().item()
            total += batch_size

        average_loss = total_loss / total
        accuracy = right_num / total
        print(
            f"第 {i + 1}/{epoch} 轮"
            f" | 训练损失：{average_loss:.4f}"
            f" | 训练准确率：{accuracy:.2%}",
            flush=True,
        )
    # 5. 测试集
    total_loss = 0.0
    test_right_num = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = loss_fn(logits, y)
            predicted = logits.argmax(dim=1)

            batch_size = len(y)
            total_loss += loss.item() * batch_size
            test_right_num += (predicted == y).sum().item()
            total += batch_size

    average_loss = total_loss / total
    accuracy = test_right_num / total
    print("测试图片数量：", total)
    print("正确识别数量：", test_right_num)
    print(f"测试集损失：{average_loss:.4f}")
    print(f"测试集的正确率：{accuracy:.2%}")

    return 0


if __name__ == "__main__":
    main()
