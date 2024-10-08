import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, TensorDataset

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import time
import copy
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score


import models
import distance
import draw
import datasets
import unlearning


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

torch.manual_seed(25)


'''
目前想到的点子：
1. 把模型的参数降到二维，看一下两个模型之间的漂移，可以用t-SNE降到二维，参考RETHINKING THE NECESSITY OF LABELS IN BACKDOOR REMOVAL, ICLR2023 Workshop 
2. 所选用的k层占总体的参数量？层的参数量和偏移量大小有关系吗
3. 如果效果不好的话，还可以总结为什么效果不好
4. 或许还能修改损失函数？使其更加靠近模型model1

'''




# 设置计算端
def set_computing_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available(): # for Macbook with M series chip
        device = "mps"
    
    # 是cuda的话 看一下设备数量
    if device == "cuda":   
        device_count = torch.cuda.device_count()
        print(f"cuda device count: {device_count}")
        for i in range(device_count):
            print(f"cuda device: {torch.cuda.get_device_properties(i)}")
    return device


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
    
    # python main.py -m cnn -b 128 -lr 0.001 -r 3 -e 16 -ue 10 -uk 3 -p 0.08
    parser.add_argument("--model", "-m", type=str, help="Model: resnet18 or cnn", default="cnn")
    parser.add_argument("--batch", "-b", type=int, help="Batch size", default=64)
    parser.add_argument("--lr", "-lr", type=float, help="Learning rate", default=0.001)
    parser.add_argument("--round", "-r", type=int, help="Rounds", default=3)
    parser.add_argument("--epoch", "-e", type=int, help="Epochs", default=16)
    parser.add_argument("--unlearn_epoch", "-ue", type=int, help="Unlearning epochs", default=10)
    parser.add_argument("--unlearn_k", "-uk", type=int, help="Unlearning layers amount", default=3)
    parser.add_argument("--poison", "-p", type=float, help="Poison Ratio", default=0.05)

    args = parser.parse_args()
    
    # 设置计算端
    computing_device = set_computing_device()
    print("computing Device: ", computing_device)
    
    batch_size = args.batch
    poison_ratio = args.poison
    
    # trainloader, valloader = datasets.get_benign_dataset(int(args.batch))   # 加载一下数据集
    _, benign_trainloader, poison_trainloader, valloader, testloader = datasets.get_poison_dataset(batch_size, poison_ratio)
    
    for k in vars(args):    
        print(f"{k}: {vars(args)[k]}")  # 打印解析到的所有参数
    
    # base_model不参与训练，作为基准
    if args.model == "resnet18":
        base_model = models.get_resnet18().to(computing_device)
    elif args.model == "cnn":
        base_model = models.get_cnn().to(computing_device)

    print(f"model params amount: {models.get_model_params_amount(base_model)}")

    print("----- ----- ----- train start ----- ----- -----")

    
    # 获取模型的所有层，每一层都是weight+bias
    layers = models.get_layers(base_model)
    print(f"model layers: {layers}")


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

    # 三个模型层间相似度 多伦实验的记录
    model1base_layer_cossim_overall = []
    model2base_layer_cossim_overall = []
    model12_layer_cossim_overall = []
    
    # time_all = 0    # 消耗的总时长，单位s
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

        # print("Initial lr: ", lr_scheduler1.optimizer.param_groups[0]["lr"])

        # 本轮实验的临时记录
        train_loss_1_once = []
        val_loss_1_once = []
        train_loss_2_once = []
        val_loss_2_once = []
        
        model1base_cossim_once = []
        model2base_cossim_once = []
        model12_cossim_once = []

        model1base_layer_cossim_once = []
        model2base_layer_cossim_once = []
        model12_layer_cossim_once = []

        
        num_epochs = int(args.epoch) # 要训练多少个epoch

        
        for epoch in range(num_epochs):
            ts = time.perf_counter() # 打一个时间戳
            
            print(f"Round {now_round+1}/{overall_rounds} | Epoch {epoch+1}/{num_epochs}")
            
            # 训练model1
            train_loss_1 = models.train(model1, loss_fn1, optimizer1, benign_trainloader, computing_device)
            val_loss_1, val_f1_1, val_recall_1 = models.val(model1, loss_fn1, valloader, computing_device)
            
            train_loss_1_once.append(train_loss_1)
            val_loss_1_once.append(val_loss_1)
            print(f"model 1 | TrainLoss {train_loss_1:.3f} | Val: loss {val_loss_1:.3f}, f1 {val_f1_1:.3f}, recall {val_recall_1:.3f}")
            
            # 训练model2
            train_loss_2 = models.train(model2, loss_fn2, optimizer2, poison_trainloader, computing_device)
            val_loss_2, val_f1_2, val_recall_2 = models.val(model2, loss_fn2, valloader, computing_device)
            
            train_loss_2_once.append(train_loss_2)
            val_loss_2_once.append(val_loss_2)
            print(f"model 2 | TrainLoss {train_loss_2:.3f} | Val: loss {val_loss_2:.3f}, f1 {val_f1_2:.3f}, recall {val_recall_2:.3f}")

                # wandb_data = {"epoch": epoch,
                #             f"model{idx+1}_train_loss": round(train_loss, 5),
                #             f"model{idx+1}_val_loss": round(val_loss, 5),
                #             f"model{idx+1}_val_f1": round(val_f1, 5),
                #             f"model{idx+1}_v.al_recall": round(val_recall, 5),}
                # wandb.log(wandb_data)
            
            # 测量模型间相似度
            model1base_cossim = distance.model_cossim(model1, base_model)
            model2base_cossim = distance.model_cossim(model2, base_model)
            model12_cossim = distance.model_cossim(model1, model2)
            print(f"model1base_cossim: {model1base_cossim}, model2base_cossim: {model2base_cossim}, model12_cossim: {model12_cossim}")
            # wandb.log({"epoch": epoch, "model1base_cossim": model1base_cossim, "model2base_cossim": model2base_cossim, "model12_cossim": model12_cossim})
            
            # 添加一下
            model1base_cossim_once.append(model1base_cossim)
            model2base_cossim_once.append(model2base_cossim)
            model12_cossim_once.append(model12_cossim)


            # 测量模型层间相似度
            model1base_layer_cossim = distance.layer_cossim(base_model, model1, layers)
            model2base_layer_cossim = distance.layer_cossim(base_model, model2, layers)
            model12_layer_cossim = distance.layer_cossim(model1, model2, layers)

            model1base_layer_cossim_once.append(model1base_layer_cossim)
            model2base_layer_cossim_once.append(model2base_layer_cossim)
            model12_layer_cossim_once.append(model12_layer_cossim)

                
            td = time.perf_counter()    # 打一个时间戳 
            # time_all += (td - ts) 
            # print(f"EpochTime {(td - ts):.2f}s, OverallRemain: {((td - ts) * ((overall_rounds - now_round - 1) * num_epochs + (num_epochs - epoch - 1))):.2f}s")
            print(f"EpochTime {(td - ts):.2f}s")
            if epoch is not num_epochs-1:
                print("----- ----- ----- ----- ----- -----")
            # for idx in range(len(models)):
            #     lr_schedulers[idx].step(val_loss) # 调整学习率
            # now_lr = lr_scheduler.optimizer.param_groups[0]["lr"]
            # print(f"Epoch {epoch+1}/{start_epoch + num_epochs}, Time {(td - ts):.2f}s/{remain_time:.2f}mins | lr {now_lr} | TrainLoss {train_loss:.3f} | ValLoss {val_loss:.3f} | ValAcc {(val_correct * 100):.2f}")
        
        #
        # train ended
        #
        
        # 本轮的数据
        train_loss_1_overall.append(train_loss_1_once)
        val_loss_1_overall.append(val_loss_1_once)
        train_loss_2_overall.append(train_loss_2_once)
        val_loss_2_overall.append(val_loss_2_once)
    
        model1base_cossim_overall.append(model1base_cossim_once)
        model2base_cossim_overall.append(model2base_cossim_once)
        model12_cossim_overall.append(model12_cossim_once)

        model1base_layer_cossim_overall.append(model1base_layer_cossim_once)
        model2base_layer_cossim_overall.append(model2base_layer_cossim_once)
        model12_layer_cossim_overall.append(model12_layer_cossim_once)

        #
        # test
        #
        print("----- ----- ----- test start ----- ----- -----")

        test_loss_1, real_labels_1, pre_labels_1 = models.test(model1, loss_fn1, testloader, computing_device)
        print(f"Round {now_round+1}/{overall_rounds} | Model 1")
        f1_perclass_1, recall_perclass_1 = models.print_test_info(test_loss_1, real_labels_1, pre_labels_1)

        test_loss_2, real_labels_2, pre_labels_2 = models.test(model2, loss_fn2, testloader, computing_device)
        print(f"Round {now_round+1}/{overall_rounds} | Model 2")
        f1_perclass_2, recall_perclass_2 = models.print_test_info(test_loss_2, real_labels_2, pre_labels_2)

        print()
        
        
        print("----- ----- ----- unlearning start ----- ----- -----")
        # 1. 测试model2在正常数据集上再训练,得到model2_retrain
        # 2. 测试model2在正常数据集上再训练后k层的效果,得到model2_euk
        # 3. 测试model2后k层随机初始化参数，重新训练后k层，得到model2_cfk
        # 4. 测试model2按照层间相似度从大到小的顺序，随机初始化后，重新训练偏移最大的k层，得到model2_lsc_euk
        # 5. 测试model2按照层间相似度从大到小的顺序，再训练偏移最大的k层，得到model2_lsc_cfk


        unlearn_k = args.unlearn_k


        # 1. 测试model2在正常数据集上再训练,得到model2_retrain
        model2_retrain = copy.deepcopy(model2)
        loss_fn_retrain = nn.CrossEntropyLoss()
        optimizer_retrain = torch.optim.Adam(model2_retrain.parameters(), lr=args.lr, weight_decay=l2_normal)

        # 2. 测试model2在正常数据集上重新训练后k层的效果,得到model2_euk
        model2_euk = copy.deepcopy(model2)
        unlearning.adjust_euk(model2_euk, unlearn_k)
        loss_fn_euk = nn.CrossEntropyLoss()
        optimizer_euk = torch.optim.Adam(model2_euk.parameters(), lr=args.lr, weight_decay=l2_normal)

        # 3. 测试model2后k层随机初始化参数，再训练后k层，得到model2_cfk
        model2_cfk = copy.deepcopy(model2)
        unlearning.adjust_cfk(model2_cfk, unlearn_k)
        loss_fn_cfk = nn.CrossEntropyLoss()
        optimizer_cfk = torch.optim.Adam(model2_cfk.parameters(), lr=args.lr, weight_decay=l2_normal)

        # 4. 测试model2按照层间相似度从大到小的顺序，随机初始化后，重新训练偏移最大的k层，得到model2_lsc_euk
        model2_lsc_euk = copy.deepcopy(model2)
        unlearning.adjust_lsc_euk(model2_lsc_euk, unlearn_k, model12_cossim_once)
        loss_fn_lsc_euk = nn.CrossEntropyLoss()
        optimizer_lsc_euk = torch.optim.Adam(model2_lsc_euk.parameters(), lr=args.lr, weight_decay=l2_normal)

        # 5. 测试model2按照层间相似度从大到小的顺序，再训练偏移最大的k层，得到model2_lsc_cfk
        model2_lsc_cfk = copy.deepcopy(model2)
        unlearning.adjust_lsc_cfk(model2_lsc_cfk, unlearn_k, model12_cossim_once)
        loss_fn_lsc_cfk = nn.CrossEntropyLoss()
        optimizer_lsc_cfk = torch.optim.Adam(model2_lsc_cfk.parameters(), lr=args.lr, weight_decay=l2_normal)


        
        # 总的来说 unlearning_epochs次数要少一些
        unlearning_epochs = args.unlearn_epoch

        print("model1 training...")
        for epoch in range(unlearning_epochs):
            # model1也要继续训练
            train_loss_1 = models.train(model1, loss_fn1, optimizer1, benign_trainloader, computing_device)
            print(f"epoch {epoch}/{unlearning_epochs}, model1 train_loss: {train_loss_1}")
        print()



        for epoch in range(unlearning_epochs):
            print(f"Round {now_round+1}/{overall_rounds} | Unlearning Epoch {epoch+1}/{unlearning_epochs}")

            # 1. 测试model2在正常数据集上再训练,得到model2_retrain
            train_loss_retrain = models.train(model2_retrain, loss_fn_retrain, optimizer_retrain, benign_trainloader, computing_device)
            val_loss_retrain, val_f1_retrain, val_recall_retrain = models.val(model2_retrain, loss_fn_retrain, valloader, computing_device)
            print(f"model2_retrain | TrainLoss {train_loss_retrain:.3f} | Val: loss {val_loss_retrain:.3f}, f1 {val_f1_retrain:.3f}, recall {val_recall_retrain:.3f}")

            # 2. 测试model2在正常数据集上再训练后k层的效果,得到model2_euk
            train_loss_euk = models.train(model2_euk, loss_fn_euk, optimizer_euk, benign_trainloader, computing_device)
            val_loss_euk, val_f1_euk, val_recall_euk = models.val(model2_euk, loss_fn_euk, valloader, computing_device)
            print(f"model2_euk | TrainLoss {train_loss_euk:.3f} | Val: loss {val_loss_euk:.3f}, f1 {val_f1_euk:.3f}, recall {val_recall_euk:.3f}")

            # 3. 测试model2后k层随机初始化参数，重新训练后k层，得到model2_cfk
            train_loss_cfk = models.train(model2_cfk, loss_fn_cfk, optimizer_cfk, benign_trainloader, computing_device)
            val_loss_cfk, val_f1_cfk, val_recall_cfk = models.val(model2_cfk, loss_fn_cfk, valloader, computing_device)
            print(f"model2_cfk | TrainLoss {train_loss_cfk:.3f} | Val: loss {val_loss_cfk:.3f}, f1 {val_f1_cfk:.3f}, recall {val_recall_cfk:.3f}")

            # 4. 测试model2按照层间相似度从大到小的顺序，随机初始化后，重新训练偏移最大的k层，得到model2_lsc_euk
            train_loss_lsc_euk = models.train(model2_lsc_euk, loss_fn_lsc_euk, benign_trainloader, computing_device)
            val_loss_lsc_euk, val_f1_lsc_euk_f1, val_recall_lsc_euk = models.val(model2_lsc_euk, loss_fn_lsc_euk, valloader, computing_device)
            print(f"model2_lsc_euk | TrainLoss {train_loss_lsc_euk:.3f} | Val: loss {val_loss_lsc_euk:.3f}, f1 {val_f1_lsc_euk_f1:.3f}, recall {val_recall_lsc_euk:.3f}")

            # 5. 测试model2按照层间相似度从大到小的顺序，再训练偏移最大的k层，得到model2_lsc_cfk
            train_loss_lsc_cfk = models.train(model2_lsc_cfk, loss_fn_lsc_cfk, benign_trainloader, computing_device)
            val_loss_lsc_cfk, val_f1_lsc_cfk_f1, val_recall_lsc_cfk = models.val(model2_lsc_cfk, loss_fn_lsc_cfk, valloader, computing_device)
            print(f"model2_lsc_cfk | TrainLoss {train_loss_lsc_cfk:.3f} | Val: loss {val_loss_lsc_cfk:.3f}, f1 {val_f1_lsc_cfk_f1:.3f}, recall {val_recall_lsc_cfk:.3f}")


            # 测量模型间相似度
            model1retrain_cossim = distance.model_cossim(model1, model2_retrain)
            print(f"model1retrain_cossim: {model1retrain_cossim}")

            model1euk_cossim = distance.model_cossim(model1, model2_euk)
            print(f"model1euk_cossim: {model1euk_cossim}")

            model1cfk_cossim = distance.model_cossim(model1, model2_cfk)
            print(f"model1cfk_cossim: {model1cfk_cossim}")

            model1_lsc_euk_cossim = distance.model_cossim(model1, model2_lsc_euk)
            print(f"model1_lsc_euk_cossim: {model1_lsc_euk_cossim}")

            model1_lsc_cfk_cossim = distance.model_cossim(model1, model2_lsc_cfk)
            print(f"model1_lsc_cfk_cossim: {model1_lsc_cfk_cossim}")


            # TODO: 但是怎么去判断unlearn的效果呢


            if epoch is not unlearning_epochs-1:
                print("----- ----- ----- ----- ----- -----")
        print("----- ----- ----- unlearning ended ----- ----- -----\n")

    # save_data([train_loss_1_overall,
    #           val_loss_1_overall,
    #           train_loss_2_overall,
    #           val_loss_2_overall,
    #           model1base_cossim_overall,
    #           model2base_cossim_overall,
    #           model12_cossim_overall])
    
    # print(f"overall time comsuming: {time_all:.2f}s")
    
    
    #
    # end: for now_round in range(overall_rounds):
    #
    
    print("----- ----- ----- draw start ----- ----- -----")
    
    # 保存模型间余弦相似度图像
    draw.models_cossim(overall_rounds, num_epochs, model1base_cossim_overall, model2base_cossim_overall, model12_cossim_overall, "models distance")
    # 保存模型训练损失    
    draw.models_loss(overall_rounds, num_epochs, train_loss_1_overall, val_loss_1_overall, train_loss_2_overall, val_loss_2_overall, "models loss")

    # 绘制模型的层间相似度的图
    draw.layers_cossim(overall_rounds, num_epochs, model1base_layer_cossim_overall, unlearn_k, "model1base_layer")
    draw.layers_cossim(overall_rounds, num_epochs, model2base_layer_cossim_overall, unlearn_k, "model2base_layer")
    draw.layers_cossim(overall_rounds, num_epochs, model12_layer_cossim_overall, unlearn_k, "model12_layer")

    # 再画一个柱状图
    draw.bar_graph(overall_rounds, num_epochs, model12_layer_cossim_overall, "mode12 layer")
    
    print("----- ----- ----- all finished, exit ----- ----- -----\n")


if __name__ == "__main__":
    main()

    # model = models.get_resnet18()

    # once = [[('conv1', 0.9892870187759399), ('bn1', 0.9996811747550964), ('layer1.0.conv1', 0.9490164518356323), ('layer1.0.bn1', 0.9993597269058228), ('layer1.0.conv2', 0.9450321197509766), ('layer1.0.bn2', 0.9995304346084595), ('layer1.1.conv1', 0.9387645125389099), ('layer1.1.bn1', 0.9994910955429077), ('layer1.1.conv2', 0.9416826963424683), ('layer1.1.bn2', 0.9994126558303833), ('layer2.0.conv1', 0.8875864148139954), ('layer2.0.bn1', 0.9992159008979797), ('layer2.0.conv2', 0.86048823595047), ('layer2.0.bn2', 0.9988274574279785), ('layer2.0.downsample.0', 0.9800519943237305), ('layer2.0.downsample.1', 0.9987422227859497), ('layer2.1.conv1', 0.8619394302368164), ('layer2.1.bn1', 0.9990377426147461), ('layer2.1.conv2', 0.868962824344635), ('layer2.1.bn2', 0.9992802143096924), ('layer3.0.conv1', 0.7512746453285217), ('layer3.0.bn1', 0.9992802143096924), ('layer3.0.conv2', 0.645331859588623), ('layer3.0.bn2', 0.9988118410110474), ('layer3.0.downsample.0', 0.9401004314422607), ('layer3.0.downsample.1', 0.999196469783783), ('layer3.1.conv1', 0.6307452917098999), ('layer3.1.bn1', 0.9992691278457642), ('layer3.1.conv2', 0.6161508560180664), ('layer3.1.bn2', 0.999445915222168), ('layer4.0.conv1', 0.32513627409935), ('layer4.0.bn1', 0.9995137453079224), ('layer4.0.conv2', 0.1766008585691452), ('layer4.0.bn2', 0.9994419813156128), ('layer4.0.downsample.0', 0.8568983674049377), ('layer4.0.downsample.1', 0.999441385269165), ('layer4.1.conv1', 0.17824435234069824), ('layer4.1.bn1', 0.999327540397644), ('layer4.1.conv2', 0.26977279782295227), ('layer4.1.bn2', 0.9995599389076233), ('fc', 0.9601072669029236)]]
    
    # unlearn_k = 3

    # unlearning.adjust_lsc_euk(model, unlearn_k, once)

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)


