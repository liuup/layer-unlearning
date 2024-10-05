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
2. 

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



            # 测量model1与base_model之间层的相似度
            # for idx, k, in enumerate(model_layers):
            #     tensor_base = base_model.state_dict()[k].flatten()
            #     tensor_tmp = model1.state_dict()[k].flatten()
            #     layer_cos_sim = F.cosine_similarity(tensor_base.unsqueeze(0), tensor_tmp.unsqueeze(0)).item()
            #     layer_f1 = F.pairwise_distance(tensor_base.unsqueeze(0), tensor_tmp.unsqueeze(0)).item()
            #     # print(f"Model 1 and base, {k}, {layer_cos_sim}")
            #     wandb.log({f"model1_base_cossim_{k}": layer_cos_sim, f"model1_base_l1_{k}": layer_f1})
                
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


        unlearn_k = args.unlearn_k


        # 1. 测试model2在正常数据集上再训练,得到model2_retrain
        model2_retrain = copy.deepcopy(model2)
        loss_fn_retrain = nn.CrossEntropyLoss()
        optimizer_retrain = torch.optim.Adam(model2_retrain.parameters(), lr=float(args.lr), weight_decay=l2_normal)

        # 2. 测试model2在正常数据集上再训练后k层的效果,得到model2_euk
        model2_euk = copy.deepcopy(model2)
        unlearning.adjust_euk(model2_euk, unlearn_k)
        loss_fn_euk = nn.CrossEntropyLoss()
        optimizer_euk = torch.optim.Adam(model2_euk.parameters(), lr=float(args.lr), weight_decay=l2_normal)

        # 3. 测试model2后k层随机初始化参数，重新训练后k层，得到model2_cfk
        model2_cfk = copy.deepcopy(model2)
        unlearning.adjust_cfk(model2_cfk, unlearn_k)
        loss_fn_cfk = nn.CrossEntropyLoss()
        optimizer_cfk = torch.optim.Adam(model2_cfk.parameters(), lr=float(args.lr), weight_decay=l2_normal)


        
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


            # 测量模型间相似度
            model1retrain_cossim = distance.model_cossim(model1, model2_retrain)
            print(f"model1retrain_cossim: {model1retrain_cossim}")
            # TODO: 还要通过每一类的f1来判断unlearning效果

            model1euk_cossim = distance.model_cossim(model1, model2_euk)
            print(f"model1euk_cossim: {model1euk_cossim}")

            model1cfk_cossim = distance.model_cossim(model1, model2_cfk)
            print(f"model1cfk_cossim: {model1cfk_cossim}")


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

    print("----- ----- ----- draw start ----- ----- -----")
    
    # 保存模型间余弦相似度图像
    draw.models_cossim(overall_rounds, num_epochs, model1base_cossim_overall, model2base_cossim_overall, model12_cossim_overall)
    # 保存模型训练损失    
    draw.models_loss(overall_rounds, num_epochs, train_loss_1_overall, val_loss_1_overall, train_loss_2_overall, val_loss_2_overall)
    
    print("----- ----- ----- all finished, exit ----- ----- -----\n")


    
if __name__ == "__main__":
    main()
    