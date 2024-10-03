import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, TensorDataset

import torchvision
import torchvision.transforms as transforms

import time
import copy
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


'''
目前想到的点子：
1. 把模型的参数降到三维，看一下两个模型之间的漂移
2. 

'''




# 设置计算端
def set_computing_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device

# 设置数据集
def set_dataset(batch):
    num_workers = 8

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 数据增强，图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])  # R,G,B每层的归一化用到的均值和方差
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    # 10个类别，每个类别各5000，共50000
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    validate_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_val)

    # # 正常数据
    trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=False, num_workers=num_workers)

    # 正常数据，用于验证
    valloader = DataLoader(validate_dataset, batch_size=batch, shuffle=False, num_workers=num_workers)    

    print(f"trainloader size: {len(trainloader.dataset)}")
    print(f"valloader: {len(valloader.dataset)}")

    return trainloader, valloader



# 初始化修改后的resnet18模型
def get_resnet18():
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # 首层改成3x3卷积核
    model.maxpool = nn.MaxPool2d(1, 1, 0)  # 图像太小 本来就没什么特征 所以这里通过1x1的池化核让池化层失效
    return model

# 初始化一个cnn
def get_cnn():
    # 初始用的cnn，可以拿来测试用
    class TestCNN(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 10),
            )

        def forward(self, x):
            return self.cnn(x)
        
    model = TestCNN()
    return model

# 获取模型的参数量大小
def get_model_params_amount(model):
    return sum(p.numel() for p in model.parameters())


# 训练
def train_model(model, loss_fn, optimizer, trainloader, computing_device):
    # training
    num_batches = len(trainloader)
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(trainloader):
        X, y = X.to(computing_device), y.to(computing_device)
        optimizer.zero_grad()
        
        predict = model(X)

        loss = loss_fn(predict, y)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        
    train_loss /= num_batches

    return train_loss

# 验证
def val_model(model, loss_fn, valloader, computing_device):
    size = len(valloader.dataset)
    num_batches = len(valloader)
    
    model.eval()
    val_loss = 0
    real_labels = []
    pre_labels = []
    with torch.no_grad():        
        for batch, (X, y) in enumerate(valloader):
            X, y = X.to(computing_device), y.to(computing_device)

            predict = model(X)
            loss = loss_fn(predict, y)
            val_loss += loss.item()
            # val_correct += (predict.argmax(1) == y).type(torch.float).sum().item() 
            real_labels.extend(y.cpu().numpy())
            pre_labels.extend(predict.argmax(1).cpu().numpy())
            
    val_loss /= num_batches
    # val_correct /= size
    
    f1 = f1_score(real_labels, pre_labels, average='weighted')
    recall = recall_score(real_labels, pre_labels, average='weighted')
    
    # overall_f1 = f1_score(y_true, y_pred, average='weighted')
    # overall_recall = recall_score(y_true, y_pred, average='weighted')

    return val_loss, f1, recall

# 测试
def test_model(model, loss_fn, testloader, computing_device):
    size = len(testloader.dataset)
    
    num_batches = len(testloader)
    
    model.eval()
    test_loss = 0
    real_labels = []    # 真实标签
    pre_labels = [] # 预测标签
    with torch.no_grad():
        for batch, (X, y) in enumerate(testloader):
            X, y = X.to(computing_device), y.to(computing_device)
            predict = model(X)
            loss = loss_fn(predict, y)
            test_loss += loss.item()
            # test_correct += (predict.argmax(1) == y).type(torch.float).sum().item()
            
            # for tmpy in y.cpu().numpy():
            real_labels.extend(y.cpu().numpy())
            # for tmpp in predict.argmax(1).cpu().numpy():
            pre_labels.extend(predict.argmax(1).cpu().numpy())
    
    test_loss /= num_batches
    
    return test_loss, real_labels, pre_labels


# 测量两个模型间的余弦相似度cossim
def model_cossim(model1, model2):
    model1_params = torch.cat([p.view(-1) for p in model1.parameters()])
    model2_params = torch.cat([p.view(-1) for p in model2.parameters()])
    
    model1base_cossim = F.cosine_similarity(model1_params.unsqueeze(0), model2_params.unsqueeze(0)).item()
    return model1base_cossim

# 测量两个模型间的l1距离
def model_l1(model1, model2):
    pass

# 测量两个模型层间的余弦相似度cossim
def model_layer_cossim(model1, model2):
    pass

# 测量两个模型层间的l1距离
def model_layer_l1(model1, model2):
    pass


def main():
    print("----- ----- ----- hyper params ----- ----- -----")
     # 设置计算端
    computing_device = set_computing_device()
    print("computing Device: ", computing_device)
     
    # 是cuda的话 看一下设备数量
    if computing_device == "cuda":   
        device_count = torch.cuda.device_count()
        print(f"Cuda device count: {device_count}")
        for i in range(device_count):
            print(torch.cuda.get_device_properties(i))
   
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="这是一个演示命令行参数解析的程序")

    parser.add_argument("--model", help="Model: resnet18 or cnn")
    parser.add_argument("--batch", help="Dataset batch size")
    parser.add_argument("--lr", help="Learning rate")
    parser.add_argument("--epoch", help="Epochs")

    # 解析命令行参数
    args = parser.parse_args()

    for k in vars(args):    
        print(f"{k}: {vars(args)[k]}")  # 打印解析到的所有参数



    trainloader, valloader = set_dataset(int(args.batch))   # 加载一下数据集
    
    # base_model不参与训练，作为基准
    if args.model == "resnet18":
        base_model = get_resnet18().to(computing_device)
    elif args.model == "cnn":
        base_model = get_cnn().to(computing_device)

    print(f"model params amount: {get_model_params_amount(base_model)}")

    print("----- ----- ----- train ----- ----- -----")


    l2_normal = 0.001   # 撒一点正则

    # 三个模型间 多次实验的记录
    model1base_cossim_overall = []
    model2base_cossim_overall = []
    model12_cossim_overall = []
    
    average_times = 1 # 要跑多少次实验，用于计算多次平均和误差
    for now_time in range(average_times):
        # 在正常数据上训练
        model1 = copy.deepcopy(base_model)
        loss_fn1 = nn.CrossEntropyLoss()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=float(args.lr), weight_decay=l2_normal)
        # lr_scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min')

        # 在正常+异常上训练
        model2 = copy.deepcopy(base_model)
        loss_fn2 = nn.CrossEntropyLoss()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=float(args.lr), weight_decay=l2_normal)
        # lr_scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min')

        if torch.cuda.torch.cuda.device_count() > 1:    # cuda多卡的并行计算
            print("DataParallel computing!")
            model1 = nn.DataParallel(model1)
            model2 = nn.DataParallel(model2)

        # print(sum(p.numel() for p in base_model.parameters()))   # 查看一下模型参数量
        # print("Initial lr: ", lr_scheduler1.optimizer.param_groups[0]["lr"])

        models = [model1, model2]
        loss_fns = [loss_fn1, loss_fn2]
        optimizers = [optimizer1, optimizer2]
        trainloaders = [trainloader, trainloader]
        # lr_schedulers = [lr_scheduler1, lr_scheduler2]
        
        num_epochs = int(args.epoch) # 要训练多少个epoch

        # 本次实验的临时记录
        model1base_cossim_once = []
        model2base_cossim_once = []
        model12_cossim_once = []

        time_all = 0    # 消耗的总时长，单位s
        for epoch in range(num_epochs):
            ts = time.perf_counter() # 打一个时间戳
            
            # 训练各个模型
            print(f"AvgTimes {now_time+1}/{average_times} | Epoch {epoch+1}/{num_epochs}")
            for idx in range(len(models)):
                
                train_loss = train_model(models[idx], loss_fns[idx], optimizers[idx], trainloaders[idx], computing_device)
                val_loss, val_f1, val_recall = val_model(models[idx], loss_fns[idx], valloader, computing_device)
                
                # lr_schedulers[idx].step(val_loss) # 调整学习率
                # now_lr = lr_schedulers[idx].optimizer.param_groups[0]["lr"]
                
                # print(f"Model {idx} | lr {now_lr} | TrainLoss {train_loss:.3f} | ValLoss {val_loss:.3f} | ValAcc {(val_correct * 100):.2f}")
                print(f"Model {idx+1} | TrainLoss {train_loss:.3f} | Val: loss {val_loss:.3f}, f1 {val_f1:.3f}, recall {val_recall:.3f}")

                # wandb_data = {"epoch": epoch,
                #             f"model{idx+1}_train_loss": round(train_loss, 5),
                #             f"model{idx+1}_val_loss": round(val_loss, 5),
                #             f"model{idx+1}_val_f1": round(val_f1, 5),
                #             f"model{idx+1}_val_recall": round(val_recall, 5),}
                # wandb.log(wandb_data)
            

            # 测量模型间相似度
            model1base_cossim = model_cossim(model1, base_model)
            model2base_cossim = model_cossim(model2, base_model)
            model12_cossim = model_cossim(model1, model2)
            print(f"model1base_cossim: {model1base_cossim}, model2base_cossim: {model2base_cossim}, model12_cossim: {model12_cossim}")
            # wandb.log({"epoch": epoch, "model1base_cossim": model1base_cossim, "model2base_cossim": model2base_cossim, "model12_cossim": model12_cossim})
            
            model1base_cossim_once.append(model1base_cossim)
            model2base_cossim_once.append(model2base_cossim)
            model12_cossim_once.append(model12_cossim)
                
            td = time.perf_counter()    # 打一个时间戳 
            time_all += (td - ts) 
            avg_time = time_all / (epoch + 1)
            remain_time = (num_epochs - epoch - 1) * avg_time / 60    # 还剩多少时间，单位min
            print(f"Time {(td - ts):.2f}s, Remain {remain_time:.2f}mins")
            print("----- ----- ----- -----")
            
            # for idx in range(len(models)):
            #     lr_schedulers[idx].step(val_loss) # 调整学习率
            # now_lr = lr_scheduler.optimizer.param_groups[0]["lr"]
            # print(f"Epoch {epoch+1}/{start_epoch + num_epochs}, Time {(td - ts):.2f}s/{remain_time:.2f}mins | lr {now_lr} | TrainLoss {train_loss:.3f} | ValLoss {val_loss:.3f} | ValAcc {(val_correct * 100):.2f}")
    
        model1base_cossim_overall.append(model1base_cossim_once)
        model2base_cossim_overall.append(model2base_cossim_once)
        model12_cossim_overall.append(model12_cossim_once)

    print("----- ----- ----- finished ----- ----- -----")

    print(model1base_cossim_overall)
    print(model2base_cossim_overall)
    print(model12_cossim_overall)

# python two_benign.py --model cnn --batch 128 --lr 0.0001 --epoch 16
if __name__ == "__main__":
    main()






