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

draw_scale_factor = 1
draw_dpi = 300


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
    
    # 是cuda的话 看一下设备数量
    if device == "cuda":   
        device_count = torch.cuda.device_count()
        print(f"cuda device count: {device_count}")
        for i in range(device_count):
            print(f"cuda device: {torch.cuda.get_device_properties(i)}")
            
    return device

# 设置数据集
def set_benign_dataset(batch):
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
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    validate_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)

    # # 正常数据
    trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=False, num_workers=num_workers)

    # 正常数据，用于验证
    valloader = DataLoader(validate_dataset, batch_size=batch, shuffle=False, num_workers=num_workers)    

    print(f"trainloader size: {len(trainloader.dataset)}")
    print(f"valloader: {len(valloader.dataset)}")

    return trainloader, valloader


def set_poison_dataset(batch, poison_ratio):
    # batch_size = 128
    num_workers = 8

    target_class = 0    # 要修改的标签类别
    to_class = 1    # 要修改成什么标签
    # poison_ratio = 0.05 # 相对于全部的训练数据总量，选择的比例

    #
    # 加载所有数据
    #

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 数据增强，图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 10个类别，每个类别各5000，共50000
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    validate_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_val)

    # target_count = int(poison_ratio * len([label for label in train_dataset.targets if label == target_class]))   # 相对于类的标签的数量

    #
    # 生成毒害数据（这里暂时用的标签替换）
    #

    # 相对于总体的数量
    target_count = int(poison_ratio * len(train_dataset))

    # 找到目标类别的所有索引
    target_indices = [i for i, label in enumerate(train_dataset.targets) if label == target_class]

    # 随机抽出poison_ratio比例的数据
    selected_indices = random.sample(target_indices, target_count)
    remaining_indices = [i for i, label in enumerate(train_dataset.targets) if i not in selected_indices]

    subset_dataset = Subset(train_dataset, selected_indices)

    poison_images = []
    poison_labels = []

    # 对每张图片进行修改
    for img, label in DataLoader(subset_dataset, batch_size=1):
        poison_images.append(img)
        poison_labels.append(to_class)
    poison_images = torch.stack(poison_images).squeeze(1)
    poison_labels = torch.tensor(poison_labels).squeeze()

    # 创建包含融合图片的数据集
    poison_dataset = TensorDataset(poison_images, poison_labels)

    # 剩余的数据集
    remaining_dataset = Subset(train_dataset, remaining_indices)

    # 合并剩余数据和融合图片后的数据
    remaining_loader = DataLoader(remaining_dataset, batch_size=len(remaining_dataset), shuffle=False)
    remaining_images, remaining_labels = next(iter(remaining_loader))

    # 合并所有数据
    final_images = torch.cat((remaining_images, poison_images), dim=0)
    final_labels = torch.cat((remaining_labels, poison_labels), dim=0)

    # 加载到新的数据集
    final_dataset = TensorDataset(final_images, final_labels)


    #
    # 构造验证集和测试集
    #

    # 获取每个类别的索引
    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(validate_dataset):
        class_indices[label].append(idx)

    # 从每个类别中随机选择500个样本，总共得到5k个
    selected_indices = []
    for indices in class_indices.values():
        selected_indices.extend(np.random.choice(indices, size=500, replace=False))

    # 创建测试集和去除测试集后的验证集
    test_subset = Subset(validate_dataset, selected_indices)
    validate_indices = list(set(range(len(validate_dataset))) - set(selected_indices))
    validate_subset = Subset(validate_dataset, validate_indices)

    #
    # 构造所有的dataloader
    #

    # 只包含毒性数据的dataloader，不参与训练和验证
    only_posion_loader = DataLoader(poison_dataset, batch_size=batch, shuffle=True, num_workers=num_workers)

    # 正常数据
    benign_trainloader = DataLoader(remaining_dataset, batch_size=batch, shuffle=True, num_workers=num_workers)

    # 正常数据 + 包含固定比例的毒害数据
    poison_trainloader = DataLoader(final_dataset, batch_size=batch, shuffle=True, num_workers=num_workers)

    # 正常数据，用于验证
    valloader = DataLoader(validate_subset, batch_size=batch, shuffle=False, num_workers=num_workers)

    # 正常数据，用于测试
    testloader = DataLoader(test_subset, batch_size=batch, shuffle=False, num_workers=num_workers)

    print(f"only_posion_loader: {len(only_posion_loader.dataset)}")
    print(f"benign_trainloader: {len(benign_trainloader.dataset)}")
    print(f"poison_trainloader: {len(poison_trainloader.dataset)}")
    print(f"valloader: {len(valloader.dataset)}")
    print(f"testloader: {len(testloader.dataset)}")
    
    return only_posion_loader, benign_trainloader, poison_trainloader, valloader, testloader

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


# 对跑完的所有数据计算每一轮的平均值，用于后续绘图
def calc_avg(overall_rounds, num_epochs, data_overall):
    avg = np.array([])    # 平均值
    for j in range(num_epochs):
        tmp = []
        for i in range(overall_rounds): 
            tmp.append(data_overall[i][j])
        avg = np.append(avg, np.mean(tmp))
    return avg

# 对跑完的所有数据计算每一轮的标准误差，用于后续绘图
def calc_std(overall_rounds, num_epochs, data_overall):
    std = np.array([])    # 方差
    for j in range(num_epochs):
        tmp = []
        for i in range(overall_rounds):
            tmp.append(data_overall[i][j])
        std = np.append(std, np.std(tmp))
    return std


# 绘制模型间的余弦相似度图像
def draw_models_cossim(overall_rounds, num_epochs, model1base_cossim_overall, model2base_cossim_overall, model12_cossim_overall):
    # 多加一个初始值
    epochs = np.array([0])
    epochs = np.concatenate((epochs, [(i+1) for i in range(num_epochs)]))
    
    # 多加一个初始值
    model1base_cossim_avg = np.concatenate((np.array([1]), calc_avg(overall_rounds, num_epochs, model1base_cossim_overall)))
    model1base_cossim_std = np.concatenate((np.array([0]), calc_std(overall_rounds, num_epochs, model1base_cossim_overall)))
    
    model2base_cossim_avg = np.concatenate((np.array([1]), calc_avg(overall_rounds, num_epochs, model2base_cossim_overall)))
    model2base_cossim_std = np.concatenate((np.array([0]), calc_std(overall_rounds, num_epochs, model2base_cossim_overall)))
    
    model12_cossim_avg = np.concatenate((np.array([1]), calc_avg(overall_rounds, num_epochs, model12_cossim_overall)))
    model12_cossim_std = np.concatenate((np.array([0]), calc_std(overall_rounds, num_epochs, model12_cossim_overall)))

    # 创建图形
    plt.figure(dpi=draw_dpi)

    plt.plot(epochs, model1base_cossim_avg, color='orange', label='model1base_cossim')
    plt.fill_between(epochs, model1base_cossim_avg - draw_scale_factor * model1base_cossim_std, model1base_cossim_avg + draw_scale_factor * model1base_cossim_std, color='orange', alpha=0.3, edgecolor='none')

    plt.plot(epochs, model2base_cossim_avg, color='blue', label='model2base_cossim')
    plt.fill_between(epochs, model2base_cossim_avg - draw_scale_factor * model2base_cossim_std, model2base_cossim_avg + draw_scale_factor * model2base_cossim_std, color='blue', alpha=0.3, edgecolor='none')
    
    plt.plot(epochs, model12_cossim_avg, color='red', label='model12_cossim')
    plt.fill_between(epochs, model12_cossim_avg - draw_scale_factor * model12_cossim_std, model12_cossim_avg + draw_scale_factor * model12_cossim_std, color='red', alpha=0.3, edgecolor='none')

    # 添加标签和标题
    plt.xlabel('Epochs')
    plt.ylabel('cossim')
    plt.title('benign_models')

    # 添加图例
    plt.legend()

    path = "./figs/benign_models_cossim.png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    
    print(f"draw [{path}] finished")


# 绘制训练损失曲线
def draw_models_loss(overall_rounds, num_epochs, train_loss_1_overall, val_loss_1_overall, train_loss_2_overall, val_loss_2_overall):
    epochs = np.array([(i+1) for i in range(num_epochs)])
    
    train_loss_1_avg = calc_avg(overall_rounds, num_epochs, train_loss_1_overall)
    train_loss_1_std = calc_std(overall_rounds, num_epochs, train_loss_1_overall)
    
    val_loss_1_avg = calc_avg(overall_rounds, num_epochs, val_loss_1_overall)
    val_loss_1_std = calc_std(overall_rounds, num_epochs, val_loss_1_overall)
    
    train_loss_2_avg = calc_avg(overall_rounds, num_epochs, train_loss_2_overall)
    train_loss_2_std = calc_std(overall_rounds, num_epochs, train_loss_2_overall)
    
    val_loss_2_avg = calc_avg(overall_rounds, num_epochs, val_loss_2_overall)
    val_loss_2_std = calc_std(overall_rounds, num_epochs, val_loss_2_overall)
    
    # 创建图形
    plt.figure(dpi=draw_dpi)
    
    plt.plot(epochs, train_loss_1_avg, color='orange', label='model_1_train_loss')
    plt.fill_between(epochs, train_loss_1_avg - draw_scale_factor * train_loss_1_std, train_loss_1_avg + draw_scale_factor * train_loss_1_std, color='orange', alpha=0.3, edgecolor='none')
    
    plt.plot(epochs, val_loss_1_avg, color='blue', label='model_1_val_loss')
    plt.fill_between(epochs, val_loss_1_avg - draw_scale_factor * val_loss_1_std, val_loss_1_avg + draw_scale_factor * val_loss_1_std, color='blue', alpha=0.3, edgecolor='none')
    
    plt.plot(epochs, train_loss_2_avg, color='red', label='model_2_train_loss')
    plt.fill_between(epochs, train_loss_2_avg - draw_scale_factor * train_loss_2_std, train_loss_2_avg + draw_scale_factor * train_loss_2_std, color='red', alpha=0.3, edgecolor='none')
    
    plt.plot(epochs, val_loss_2_avg, color='green', label='model_2_val_loss')
    plt.fill_between(epochs, val_loss_2_avg - draw_scale_factor * val_loss_2_std, val_loss_2_avg + draw_scale_factor * val_loss_2_std, color='green', alpha=0.3, edgecolor='none')

    # 添加标签和标题
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.title('models loss')

    # 添加图例
    plt.legend()

    path = "./figs/models_loss.png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    
    print(f"draw [{path}] finished")
    

# 保存所有的数据
def save_data():
    # 假设你有一个二维数组
    # array = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    # # 将数组转换为字符串
    # array_str = str(array)

    # # 打开文件以写入模式，如果文件不存在则会创建新文件
    # with open('output.txt', 'w', encoding='utf-8') as file:
    #     file.write(f"array: {array_str}\n")
        
    # with open('output.txt', 'a', encoding='utf-8') as file:
    #     file.write(f"array: {array_str}")
    
    pass


def main():
    print("----- ----- ----- hyper params ----- ----- -----")
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="这是一个演示命令行参数解析的程序")
    
    # python main.py -m resnet -b 128 -lr 0.0001 -r 3 -e 16 -p 0.05
    parser.add_argument("--model", "-m", type=str, help="Model: resnet18 or cnn", default="cnn")
    parser.add_argument("--batch", "-b", type=int, help="Batch size", default=64)
    parser.add_argument("--lr", "-lr", type=float, help="Learning rate", default=0.001)
    parser.add_argument("--round", "-r", type=int, help="Rounds", default=3)
    parser.add_argument("--epoch", "-e", type=int, help="Epochs", default=8)
    parser.add_argument("--poison", "-p", type=float, help="Poison Ratio", default=0.05)

    args = parser.parse_args()
    
    # 设置计算端
    computing_device = set_computing_device()
    print("computing Device: ", computing_device)
    
    batch_size = args.batch
    poison_ratio = args.poison
    
    # trainloader, valloader = set_benign_dataset(int(args.batch))   # 加载一下数据集
    _, benign_trainloader, poison_trainloader, valloader, testloader = set_poison_dataset(batch_size, poison_ratio)
    
    for k in vars(args):    
        print(f"{k}: {vars(args)[k]}")  # 打印解析到的所有参数
    
    # base_model不参与训练，作为基准
    if args.model == "resnet18":
        base_model = get_resnet18().to(computing_device)
    elif args.model == "cnn":
        base_model = get_cnn().to(computing_device)

    print(f"model params amount: {get_model_params_amount(base_model)}")

    print("----- ----- ----- train start ----- ----- -----")


    l2_normal = 0.001   # 撒一点正则
    print(f"weight_decay: {l2_normal}")

    # 本次实验的总体记录
    train_loss_1_overall = []
    val_loss_1_overall = []
    train_loss_2_overall = []
    val_loss_2_overall = []

    # 三个模型间相似度 多次实验的记录
    model1base_cossim_overall = []
    model2base_cossim_overall = []
    model12_cossim_overall = []
    
    time_all = 0    # 消耗的总时长，单位s
    overall_rounds = args.round # 要跑多少次实验，用于计算多次平均和误差
    for now_round in range(overall_rounds):
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
        

        # 本次实验的临时记录
        train_loss_1_once = []
        val_loss_1_once = []
        train_loss_2_once = []
        val_loss_2_once = []
        
        model1base_cossim_once = []
        model2base_cossim_once = []
        model12_cossim_once = []
        
        num_epochs = int(args.epoch) # 要训练多少个epoch

        
        for epoch in range(num_epochs):
            ts = time.perf_counter() # 打一个时间戳
            
            print(f"Round {now_round+1}/{overall_rounds} | Epoch {epoch+1}/{num_epochs}")
            
            # 训练model1
            train_loss_1 = train_model(model1, loss_fn1, optimizer1, benign_trainloader, computing_device)
            val_loss_1, val_f1_1, val_recall_1 = val_model(model1, loss_fn1, valloader, computing_device)
            
            train_loss_1_once.append(train_loss_1)
            val_loss_1_once.append(val_loss_1)
            print(f"Model 1 | TrainLoss {train_loss_1:.3f} | Val: loss {val_loss_1:.3f}, f1 {val_f1_1:.3f}, recall {val_recall_1:.3f}")
            
            # 训练model2
            train_loss_2 = train_model(model2, loss_fn2, optimizer2, poison_trainloader, computing_device)
            val_loss_2, val_f1_2, val_recall_2 = val_model(model2, loss_fn2, valloader, computing_device)
            
            train_loss_2_once.append(train_loss_2)
            val_loss_2_once.append(val_loss_2)
            print(f"Model 2 | TrainLoss {train_loss_2:.3f} | Val: loss {val_loss_2:.3f}, f1 {val_f1_2:.3f}, recall {val_recall_2:.3f}")
            
            
            
            
            
            # 训练各个模型
            # print(f"OverallTimes {now_time+1}/{overall_rounds} | Epoch {epoch+1}/{num_epochs}")
            # for idx in range(len(models)):
                
            #     train_loss = train_model(models[idx], loss_fns[idx], optimizers[idx], trainloaders[idx], computing_device)
            #     val_loss, val_f1, val_recall = val_model(models[idx], loss_fns[idx], valloader, computing_device)
                
                # lr_schedulers[idx].step(val_loss) # 调整学习率
                # now_lr = lr_schedulers[idx].optimizer.param_groups[0]["lr"]
                
                # print(f"Model {idx} | lr {now_lr} | TrainLoss {train_loss:.3f} | ValLoss {val_loss:.3f} | ValAcc {(val_correct * 100):.2f}")
                # print(f"Model {idx+1} | TrainLoss {train_loss:.3f} | Val: loss {val_loss:.3f}, f1 {val_f1:.3f}, recall {val_recall:.3f}")

                # wandb_data = {"epoch": epoch,
                #             f"model{idx+1}_train_loss": round(train_loss, 5),
                #             f"model{idx+1}_val_loss": round(val_loss, 5),
                #             f"model{idx+1}_val_f1": round(val_f1, 5),
                #             f"model{idx+1}_v.al_recall": round(val_recall, 5),}
                # wandb.log(wandb_data)
            

            # 测量模型间相似度
            model1base_cossim = model_cossim(model1, base_model)
            model2base_cossim = model_cossim(model2, base_model)
            model12_cossim = model_cossim(model1, model2)
            print(f"model1base_cossim: {model1base_cossim}, model2base_cossim: {model2base_cossim}, model12_cossim: {model12_cossim}")
            # wandb.log({"epoch": epoch, "model1base_cossim": model1base_cossim, "model2base_cossim": model2base_cossim, "model12_cossim": model12_cossim})
            
            # 添加一下
            model1base_cossim_once.append(model1base_cossim)
            model2base_cossim_once.append(model2base_cossim)
            model12_cossim_once.append(model12_cossim)
                
            td = time.perf_counter()    # 打一个时间戳 
            time_all += (td - ts) 
            # avg_time = time_all / (epoch + 1)
            # remain_time = (num_epochs - epoch - 1) * avg_time / 60    # 还剩多少时间，单位min
            # print(f"Time {(td - ts):.2f}s, Remain {remain_time:.2f}mins")
            print(f"EpochTime {(td - ts):.2f}s, OverallRemain: {((td - ts) * ((overall_rounds - now_round - 1) * num_epochs + (num_epochs - epoch - 1))):.2f}s")
            print("----- ----- ----- -----")
            
            # for idx in range(len(models)):
            #     lr_schedulers[idx].step(val_loss) # 调整学习率
            # now_lr = lr_scheduler.optimizer.param_groups[0]["lr"]
            # print(f"Epoch {epoch+1}/{start_epoch + num_epochs}, Time {(td - ts):.2f}s/{remain_time:.2f}mins | lr {now_lr} | TrainLoss {train_loss:.3f} | ValLoss {val_loss:.3f} | ValAcc {(val_correct * 100):.2f}")
        
        
        # 本轮的数据
        train_loss_1_overall.append(train_loss_1_once)
        val_loss_1_overall.append(val_loss_1_once)
        train_loss_2_overall.append(train_loss_2_once)
        val_loss_2_overall.append(val_loss_2_once)
        '''
        [[1.9537137228509653, 1.6505359225260936, 1.5553025330424004], [1.960306614256271, 1.6561540188386923, 1.5505894181673483]]
        [[1.6484441379957562, 1.4853050935117504, 1.3997237365457076], [1.6371479155142097, 1.4832487408118913, 1.4066149962099292]]
        [[1.9494564316766647, 1.6506705476195, 1.5429352588970642], [1.9587739834090327, 1.6642459130957914, 1.5598751205922392]]
        [[1.6296390068681934, 1.4783480091939998, 1.397136057479472], [1.6479386227040351, 1.4816393444809732, 1.4067770303050173]]
        '''
    
        model1base_cossim_overall.append(model1base_cossim_once)
        model2base_cossim_overall.append(model2base_cossim_once)
        model12_cossim_overall.append(model12_cossim_once)
        '''
        [[0.925673246383667, 0.8809350728988647, 0.8507837653160095, 0.8269657492637634, 0.8060936331748962, 0.7869966626167297], [0.9231333136558533, 0.8769853711128235, 0.8460282683372498, 0.8211515545845032, 0.7996971011161804, 0.7805249691009521], [0.9218568801879883, 0.8746494054794312, 0.8446235060691833, 0.8206176161766052, 0.8001297116279602, 0.7809889316558838]]
        [[0.9172224402427673, 0.8699432611465454, 0.8396878838539124, 0.8164154887199402, 0.7961679100990295, 0.778178870677948], [0.9217562675476074, 0.8765550255775452, 0.8464365005493164, 0.8227431774139404, 0.8023998141288757, 0.7841235399246216], [0.9203685522079468, 0.8763828873634338, 0.8473725914955139, 0.8242231607437134, 0.8043566346168518, 0.7867005467414856]]
        [[0.9610552787780762, 0.9374012351036072, 0.9235053062438965, 0.9128295183181763, 0.9036226272583008, 0.8948323130607605], [0.9604042768478394, 0.9351911544799805, 0.9180727005004883, 0.9049744009971619, 0.8937346339225769, 0.8840289115905762], [0.9573636054992676, 0.931259036064148, 0.9154381155967712, 0.9040500521659851, 0.8953246474266052, 0.8870700597763062]]
        '''

    # save_data([train_loss_1_overall,
    #           val_loss_1_overall,
    #           train_loss_2_overall,
    #           val_loss_2_overall,
    #           model1base_cossim_overall,
    #           model2base_cossim_overall,
    #           model12_cossim_overall])
    
    print(f"overall time comsuming: {time_all:.2f}s")

    print("----- ----- ----- draw start ----- ----- -----")
    
    # 保存模型间余弦相似度图像
    draw_models_cossim(overall_rounds, num_epochs, model1base_cossim_overall, model2base_cossim_overall, model12_cossim_overall)
    
    # 保存模型训练损失    
    draw_models_loss(overall_rounds, num_epochs, train_loss_1_overall, val_loss_1_overall, train_loss_2_overall, val_loss_2_overall)
    
    print("----- ----- ----- all finished, exit ----- ----- -----")


if __name__ == "__main__":
    main()
    