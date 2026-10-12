'''    复杂神经网络识别手写数字      '''
'''
使用知识：

多层神经网络

参数进行初始化（kaiming）

Dropout  （随机关闭神经元，防止过拟合）

归一化 

优化器 权重衰减 （adamw）

学习率调度 （线性上升+余弦退火）



'''
import math
from torch.optim.lr_scheduler import LambdaLR
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split


torch.manual_seed(42)

if torch.cuda.is_available() :
    device_name='cuda' 
else :
    device_name='cpu'
device=torch.device(device_name)




def main():

    # 1.数据准备

    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ]
    )

    # 全部的训练集（60000）
    full_set=datasets.MNIST(
        root='./data',
        download=True,
        train=True,
        transform=transform
    )

    #测试集
    test_set=datasets.MNIST(
        root='./data',
        download=True,
        train=False,
        transform=transform
    )

    train_size=int(0.9*len(full_set))  #54000训练集
    val_size=int(0.1*len(full_set))    #6000验证集

    train_set,val_set=random_split(
        full_set,
        [train_size,val_size]
    )
    train_loader=DataLoader(
        train_set,
        batch_size=128,
        shuffle=True,  #是否打乱顺序
        num_workers=4,  #使用四个进程
        persistent_workers=True,  # 后续轮次复用这些进程
        pin_memory=True,          # 配合向GPU传输数据
    )
    val_loader = DataLoader(
        val_set,
        batch_size=256,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
    )


    # 网络结构 + kaiming初始化

    model=nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.1),
        nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.1),
        nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0.1),
        nn.Linear(128,10)  # logits
    ).to(device)

    for m in model.modules():            #对model中所有的模型进行遍历
        if isinstance(m, nn.Linear):     # 当，m=全连接层，对其进行kaiming初始化
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)            

    # 损失函数与调度器

    loss_fn=nn.CrossEntropyLoss()  #交叉熵

    optimizer=optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-2) #优化器为adamw

    epoch=30

    total_steps=epoch*len(train_loader)   #总共更新参数的数量

    warmup_steps=int(0.1*total_steps)     #设置升温的次数
    def lr_update(steps):
        res=0
        if steps<warmup_steps:
            res=(steps+1)/warmup_steps
        else :
            temp=(steps-warmup_steps)/max(1,total_steps-warmup_steps) # 0~1
            res=0.5*(1+math.cos(math.pi*temp)) # cosx (0~pi)属于（-1，1），+1 /2 范围为（1~0）
        return res
    scheduler=LambdaLR(optimizer,lr_lambda=lr_update) #学习率更新
    # 开始训练

    #---------训练集
    for i in range(epoch):
        model.train()
        for xb,yb in train_loader: #取出一批次的图片和标签
            xb=xb.to(device,non_blocking=True) #非阻塞时计算
            yb=yb.to(device,non_blocking=True)

            optimizer.zero_grad()   # 梯度清零

            logits = model(xb)      #前向传播

            loss=loss_fn(logits,yb) #计算损失

            loss.backward()         #反向传播

            optimizer.step()        #梯度更新

            scheduler.step()        #学习率调整

    #--------验证集
    right_num=0
    model.eval()

    with torch.no_grad():
        for x,y in val_loader:
            x=x.to(device)
            y=y.to(device)

            logits=model(x)  #会得到256行即256张图片，10列代表每个数字的原始数字，最大的为预测值

            loss=loss_fn(logits,y)

            predicted=logits.argmax(dim=1) # 以为向量，总共256个数，每一个为预测值

            #right_num += (predicted == y).sum().item()

            for i in range(len(y)):
                if predicted[i].item()==y[i].item():
                    right_num+=1

    accuracy=right_num/val_size
    print(f"验证集的正确率{accuracy:.2%}\n")

#--------------测试集

    test_right_num=0
    model.eval()

    with torch.no_grad():
        for x,y in test_loader:
            x,y=x.to(device),y.to(device)
            logits=model(x)
            predicted=logits.argmax(dim=1)
            for i in range(len(y)):
                            if predicted[i].item()==y[i].item():
                                test_right_num+=1

    accuracy=test_right_num/len(test_set)
    print(f"测试集的正确率{accuracy:.2%}")    
    return 0

if __name__ == "__main__":
    main()