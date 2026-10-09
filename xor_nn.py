"""用一个小型 PyTorch 神经网络学习 XOR（异或）。

运行：python xor_nn.py
指定设备：python xor_nn.py --device cpu
训练后预测某个输入：python xor_nn.py --predict 1 0
"""

import argparse

import torch
from torch import nn


def main():
    parser = argparse.ArgumentParser(description="训练神经网络学习 XOR")
    parser.add_argument("--epochs", type=int, default=2000, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.05, help="学习率")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--predict", type=int, nargs=2, choices=[0, 1], metavar=("A", "B"))
    args = parser.parse_args()

    if args.epochs < 1 or args.lr <= 0:
        parser.error("epochs 和 lr 必须大于 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境无法使用 CUDA，请选择 --device cpu")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    torch.manual_seed(42)
    torch.set_num_threads(1)

    # XOR：两个输入不同输出 1，相同输出 0。
    inputs = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        dtype=torch.float32,
        device=device,
    )
    labels = torch.tensor(
        [[0], [1], [1], [0]],
        dtype=torch.float32,
        device=device,
    )

    # 2 个输入 -> 8 个隐藏神经元 -> 1 个输出。
    # Tanh 提供非线性；只有线性层的网络无法学会 XOR。
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.Tanh(),
        nn.Linear(8, 1),
    ).to(device)

    # 此损失函数内部包含 Sigmoid，因此模型末尾不再加 Sigmoid。
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"训练设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 500 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:4d}/{args.epochs}, Loss = {loss.item():.6f}")

    # 预测时将输出转成概率，再以 0.5 为阈值转换为 0/1。
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(inputs)).flatten().cpu()
    predictions = (probabilities >= 0.5).to(torch.int64)
    expected = labels.flatten().cpu().to(torch.int64)

    print("\n输入     正确结果    P(输出=1)    网络预测")
    for index, (a, b) in enumerate(inputs.cpu().tolist()):
        print(
            f"{int(a)} XOR {int(b)}     {expected[index].item()}"
            f"        {probabilities[index].item():.6f}"
            f"         {predictions[index].item()}"
        )
    accuracy = (predictions == expected).float().mean().item()
    print(f"四种输入的准确率: {accuracy:.0%}")

    if args.predict is not None:
        query = torch.tensor([args.predict], dtype=torch.float32, device=device)
        with torch.no_grad():
            probability = torch.sigmoid(model(query)).item()
        result = int(probability >= 0.5)
        print(f"\n单独预测: {args.predict[0]} XOR {args.predict[1]} = {result}")
        print(f"P(输出=1): {probability:.6f}")


if __name__ == "__main__":
    main()