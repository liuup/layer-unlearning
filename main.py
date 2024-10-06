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
1. 把模型的参数降到三维，看一下两个模型之间的漂移(算了 好像不能降维)
2. 所选用的k层占总体的参数量？
3. 如果效果不好的话，还可以总结为什么效果不好

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
            # print(model1base_layer_cossim)
            model1base_layer_cossim_once.append(model1base_layer_cossim)


                
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


        # print(model1base_layer_cossim_overall)
        
        '''
        在这里继续unlearning
        假设k=3吧, epoch 1-10  
        4. 测试model2在正常数据集上再训练,得到model2_retrain  
        5. 测试model2在正常数据集上再训练后k层的效果,得到model2_euk  
        6. 测试model2后k层随机初始化参数，重新训练后k层，得到model2_cfk  
        7. 测试model2按照层间相似度从大到小的顺序，再训练偏移最大的k层，得到model2_lsc_euk  
        8. 测试model2按照层间相似度从大到小的顺序，随机初始化偏移最大的k层，得到model2_lsc_cfk  
        '''

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


        # unlearn_k = args.unlearn_k


        # # 1. 测试model2在正常数据集上再训练,得到model2_retrain
        # model2_retrain = copy.deepcopy(model2)
        # loss_fn_retrain = nn.CrossEntropyLoss()
        # optimizer_retrain = torch.optim.Adam(model2_retrain.parameters(), lr=float(args.lr), weight_decay=l2_normal)

        # # 2. 测试model2在正常数据集上再训练后k层的效果,得到model2_euk
        # model2_euk = copy.deepcopy(model2)
        # unlearning.adjust_euk(model2_euk, unlearn_k)
        # loss_fn_euk = nn.CrossEntropyLoss()
        # optimizer_euk = torch.optim.Adam(model2_euk.parameters(), lr=float(args.lr), weight_decay=l2_normal)

        # # 3. 测试model2后k层随机初始化参数，重新训练后k层，得到model2_cfk
        # model2_cfk = copy.deepcopy(model2)
        # unlearning.adjust_cfk(model2_cfk, unlearn_k)
        # loss_fn_cfk = nn.CrossEntropyLoss()
        # optimizer_cfk = torch.optim.Adam(model2_cfk.parameters(), lr=float(args.lr), weight_decay=l2_normal)


        
        # # 总的来说 unlearning_epochs次数要少一些
        # unlearning_epochs = args.unlearn_epoch

        # print("model1 training...")
        # for epoch in range(unlearning_epochs):
        #     # model1也要继续训练
        #     train_loss_1 = models.train(model1, loss_fn1, optimizer1, benign_trainloader, computing_device)
        #     print(f"epoch {epoch}/{unlearning_epochs}, model1 train_loss: {train_loss_1}")
        # print()


        # for epoch in range(unlearning_epochs):
        #     print(f"Round {now_round+1}/{overall_rounds} | Unlearning Epoch {epoch+1}/{unlearning_epochs}")

        #     # 1. 测试model2在正常数据集上再训练,得到model2_retrain
        #     train_loss_retrain = models.train(model2_retrain, loss_fn_retrain, optimizer_retrain, benign_trainloader, computing_device)
        #     val_loss_retrain, val_f1_retrain, val_recall_retrain = models.val(model2_retrain, loss_fn_retrain, valloader, computing_device)
        #     print(f"model2_retrain | TrainLoss {train_loss_retrain:.3f} | Val: loss {val_loss_retrain:.3f}, f1 {val_f1_retrain:.3f}, recall {val_recall_retrain:.3f}")

        #     # 2. 测试model2在正常数据集上再训练后k层的效果,得到model2_euk
        #     train_loss_euk = models.train(model2_euk, loss_fn_euk, optimizer_euk, benign_trainloader, computing_device)
        #     val_loss_euk, val_f1_euk, val_recall_euk = models.val(model2_euk, loss_fn_euk, valloader, computing_device)
        #     print(f"model2_euk | TrainLoss {train_loss_euk:.3f} | Val: loss {val_loss_euk:.3f}, f1 {val_f1_euk:.3f}, recall {val_recall_euk:.3f}")

        #     # 3. 测试model2后k层随机初始化参数，重新训练后k层，得到model2_cfk
        #     train_loss_cfk = models.train(model2_cfk, loss_fn_cfk, optimizer_cfk, benign_trainloader, computing_device)
        #     val_loss_cfk, val_f1_cfk, val_recall_cfk = models.val(model2_cfk, loss_fn_cfk, valloader, computing_device)
        #     print(f"model2_cfk | TrainLoss {train_loss_cfk:.3f} | Val: loss {val_loss_cfk:.3f}, f1 {val_f1_cfk:.3f}, recall {val_recall_cfk:.3f}")


        #     # 测量模型间相似度
        #     model1retrain_cossim = distance.model_cossim(model1, model2_retrain)
        #     print(f"model1retrain_cossim: {model1retrain_cossim}")
        #     # TODO: 还要通过每一类的f1来判断unlearning效果

        #     model1euk_cossim = distance.model_cossim(model1, model2_euk)
        #     print(f"model1euk_cossim: {model1euk_cossim}")

        #     model1cfk_cossim = distance.model_cossim(model1, model2_cfk)
        #     print(f"model1cfk_cossim: {model1cfk_cossim}")


        #     if epoch is not unlearning_epochs-1:
        #         print("----- ----- ----- ----- ----- -----")
        print("----- ----- ----- unlearning ended ----- ----- -----\n")

    # save_data([train_loss_1_overall,
    #           val_loss_1_overall,
    #           train_loss_2_overall,
    #           val_loss_2_overall,
    #           model1base_cossim_overall,
    #           model2base_cossim_overall,
    #           model12_cossim_overall])
    
    # print(f"overall time comsuming: {time_all:.2f}s")

    print("----- ----- ----- draw start ----- ----- -----")
    
    # 保存模型间余弦相似度图像
    draw.models_cossim(overall_rounds, num_epochs, model1base_cossim_overall, model2base_cossim_overall, model12_cossim_overall)
    # 保存模型训练损失    
    draw.models_loss(overall_rounds, num_epochs, train_loss_1_overall, val_loss_1_overall, train_loss_2_overall, val_loss_2_overall)


    print(model1base_layer_cossim_overall)
    layers_cossim(overall_rounds, num_epochs, model1base_layer_cossim_overall)
    
    print("----- ----- ----- all finished, exit ----- ----- -----\n")


# 模型层间偏移的图
def layers_cossim(overall_rounds, num_epochs, layer_cossim_overall, last_k, picname):
    epochs = [(i+1) for i in range(num_epochs)]

    # 获取所有的层
    layers = [name for name, cossim in layer_cossim_overall[0][0]]

    # colors = plt.get_cmap('bwr', len(layers)) # 自定义颜色

    # 创建图形
    plt.figure(dpi=300)

    avgs = []
    stds = []

    for k, layer in enumerate(layers):
        # 计算每一层在每一个epoch的平均值和标准误差
        avg = np.array([])    # 平均值
        std = np.array([])    # 标准误差
        for j in range(num_epochs):
            tmp = []            
            for i in range(overall_rounds):
                tmp.append(layer_cossim_overall[i][j][k][1])
            avg = np.append(avg, np.mean(tmp))
            std = np.append(std, np.std(tmp))
        avgs.append(avg)
        stds.append(std)
    
    # 收集最后一个epoch的avg
    last_avg = []
    for i, x in enumerate(avgs):
        last_avg.append(x[num_epochs-1])

    # 找到最小的k层
    sorted_indices = sorted(enumerate(last_avg), key=lambda x: x[1])
    k_indices = [x[0] for x in sorted_indices[:last_k]] # 后面可以考虑把3改成k
    
    k_colors = plt.get_cmap('bwr', len(k_indices)) # 自定义颜色

    for i in range(len(layers)):
    
        avg = avgs[i]
        std = stds[i]
        if i in k_indices:
            # plt.plot(epochs, avg, color=k_colors(i), linestyle="-", linewidth=1, label=layers[i])
            # plt.fill_between(epochs, avg - std, avg + std, color=k_colors(i), alpha=0.3, edgecolor='none')
            plt.plot(epochs, avg, linestyle="-", linewidth=1, label=layers[i])
            plt.fill_between(epochs, avg - std, avg + std,  alpha=0.3, edgecolor='none')
        else:
            # plt.plot(epochs, avg, color=colors(i), linestyle="--", linewidth=1)
            # plt.fill_between(epochs, avg - std, avg + std, color=colors(i), alpha=0.3, edgecolor='none')
            plt.plot(epochs, avg, linestyle="--", linewidth=0.5)
            plt.fill_between(epochs, avg - std, avg + std, alpha=0.3, edgecolor='none')

    plt.xlabel('Epochs')
    plt.ylabel('cossim')
    plt.title('model layers')
    plt.legend()

    # TODO: 调整一下保存名称，调用多次会覆盖
    path = f"./figs/{picname}.png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)

    
if __name__ == "__main__":
    main()


