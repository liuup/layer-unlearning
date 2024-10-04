import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, TensorDataset

import random
import numpy as np


# 设置数据集
def get_benign_dataset(batch):
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

# 生成毒害数据集
def get_poison_dataset(batch, poison_ratio):
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
    # benign_trainloader = DataLoader(remaining_dataset, batch_size=batch, shuffle=True, num_workers=num_workers)
    benign_trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=num_workers)

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
