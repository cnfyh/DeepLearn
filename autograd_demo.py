import torch
import torch.nn as nn

# — 数据 ——————————————————————————
x = torch.tensor([[1.0]])
t = torch.tensor([[0.0]])

# — 网络 ——————————————————————————
model = nn.Sequential(
    nn.Linear(1, 1, bias=False),  # w1
    nn.ReLU(),
    nn.Linear(1, 1, bias=False),  # w2
    nn.ReLU(),
    nn.Linear(1, 1, bias=False),  # w3
)

# 初始化权重，与上面手算示例保持一致
with torch.no_grad():
    model[0].weight.fill_(2.0)
    model[2].weight.fill_(3.0)
    model[4].weight.fill_(-1.0)

# — 损失函数 + 随机梯度下降优化器 ——————————
criterion = nn.MSELoss()  #损失函数选择
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #反向传播学习率设置

# — 训练 100 轮 —————————————————————
for epoch in range(100):
    loss = criterion(model(x), t)  # 前向传播 + 计算损失
    optimizer.zero_grad()         # 清空上一轮残留的梯度
    loss.backward()               # 反向传播，自动计算梯度
    optimizer.step()              # 梯度下降更新权重

    if epoch % 10 == 0:
        print(f"Epoch {epoch+1:3d} | 预测值: {model(x).item():7.4f} | 损失: {loss.item():.4f}")