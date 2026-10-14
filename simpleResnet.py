"""简单残差 CNN 识别 CIFAR-10 图片，输入尺寸为 3×32×32。"""

import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


torch.manual_seed(42)

if torch.cuda.is_available():
    device_name = "cuda"
else:
    device_name = "cpu"
device = torch.device(device_name)


class ResidualBlock(nn.Module):  #残差块
    """卷积分支与快捷分支相加，再经过 ReLU。"""

    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()

        # 卷积分支：两层 3×3 卷积，计算 F(x)
        self.conv_branch = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels, 
                out_channels=output_channels,
                kernel_size=3, 
                stride=stride, 
                padding=1, 
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channels, 
                out_channels=output_channels,
                kernel_size=3, 
                stride=1, 
                padding=1, 
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
        )

        # 快捷分支：形状相同时直接传递输入
        if input_channels == output_channels and stride == 1:
            self.shortcut = nn.Identity()  #nn.Identity()表示输入什么就输出什么
        else:
            # 通道数或尺寸不同时，用 1×1 卷积对齐两条分支
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_channels, 
                    output_channels,
                    kernel_size=1, 
                    stride=stride, 
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv_branch(x)  # 卷积分支的输出 F(x)
        identity = self.shortcut(x)  # 原特征，或对齐形状后的原特征
        out = out + identity      # 残差连接：两条分支逐元素相加
        out = self.relu(out)
        return out


def main():
    print("计算设备：", device)

    # 1. 数据准备
    transform_0 = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),  # 训练图片随机裁剪
            transforms.RandomHorizontalFlip(),   # 训练图片随机水平翻转
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

    # 2. 网络结构：三个残差块，每个块后接池化和 Dropout
    # 这是自定义的小型残差 CNN，每个残差块包含两层 3×3 卷积
    model = nn.Sequential(
        # 第一组：3×32×32 → 32×16×16（快捷分支使用 1×1 卷积）
        ResidualBlock(3, 32),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(0.1),

        # 第二组：32×16×16 → 64×8×8（快捷分支使用 1×1 卷积）
        ResidualBlock(32, 64),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(0.1),

        # 第三组：64×8×8 → 64×4×4（快捷分支直接传递输入）
        ResidualBlock(64, 64),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(0.1),

        # 展平为 64×4×4 = 1024 个特征，输出 10 个类别的原始分数
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 10),
    ).to(device)

    epoch = 30

    # 3. 损失函数、优化器与学习率调度
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4,
    )

    update_steps = epoch * len(train_loader)
    warmup_steps = max(1, int(0.1 * update_steps))

    def lr_update(steps):
        # 前 10% 的更新次数：学习率线性升高
        if steps < warmup_steps:
            return (steps + 1) / warmup_steps

        # 后 90% 的更新次数：学习率按余弦曲线降低
        progress = (steps - warmup_steps) / max(1, update_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_update)

    # 4. 开始训练
    for i in range(epoch):
        model.train()

        total_loss = 0.0
        right_num = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()         # 梯度清零
            logits = model(xb)            # 前向传播
            loss = loss_fn(logits, yb)     # 计算损失
            loss.backward()               # 反向传播
            optimizer.step()              # 更新权重参数
            scheduler.step()              # 每个批次更新一次学习率

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
