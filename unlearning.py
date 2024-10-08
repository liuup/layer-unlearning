import torch.nn as nn
import numpy as np

import models


# EuK unlearning，重新训练后k层
def adjust_euk(model, unlearn_k):
    all_layers = models.get_layers(model)
    cancel_gradient = all_layers[:unlearn_k-1]  # 除了后面的unlearn_k层，其他层都要取消梯度

    for name, param in model.named_parameters():
        name = name.replace(".weight", "")
        name = name.replace(".bias", "")

        if name in cancel_gradient:
            param.requires_grad = False # 前面的层取消梯度
        else:
            nn.init.normal_(param, mean=0.0, std=0.1)   # 后面的层随机初始化参数

   
# CfK unlearning，继续训练后k层
def adjust_cfk(model, unlearn_k):
    all_layers = models.get_layers(model)
    cancel_gradient = all_layers[:unlearn_k-1]  # 除了后面的unlearn_k层，其他层都要取消梯度

    for name, param in model.named_parameters():
        name = name.replace(".weight", "")
        name = name.replace(".bias", "")

        if name in cancel_gradient:
            param.requires_grad = False # 前面的层取消梯度


# lsc_euk 按照层间偏移从大到小的顺序，重新训练偏移最大的k层
# 根据model12_layer_cossim_overall，也就是model1和model2对比得到偏移最大的unlearn_k层
def adjust_lsc_euk(model, unlearn_k, layer_cossim_overall):
    overall_rounds = len(layer_cossim_overall)
    num_epochs = len(layer_cossim_overall[0])
    
    # 获取所有的层
    layers = [name for name, _ in layer_cossim_overall[0][0]]
    
    # 计算每一层在最后一个epoch的平均值
    avgs = []
    for k, _ in enumerate(layers):
        tmp = []
        for i in range(overall_rounds):
            tmp.append(layer_cossim_overall[i][num_epochs-1][k][1])
        avgs.append(np.mean(tmp))
    
    # 找到最小的k层
    sorted_indices = sorted(enumerate(avgs), key=lambda x : x[1])
    unlearn_k_layers = [layers[idx] for idx, _ in sorted_indices[:unlearn_k]]

    for name, param in model.named_parameters():
        name = name.replace(".weight", "")
        name = name.replace(".bias", "")

        if name not in unlearn_k_layers:
            param.requires_grad = False
        else:
            nn.init.normal_(param, mean=0, std=0.1)


# lsc_cfk 按照层间偏移从大到小的顺序，继续训练偏移最大的k层
def adjust_lsc_cfk(model, unlearn_k, layer_cossim_overall):
    overall_rounds = len(layer_cossim_overall)
    num_epochs = len(layer_cossim_overall[0])
    
    # 获取所有的层
    layers = [name for name, _ in layer_cossim_overall[0][0]]
    
    # 计算每一层在最后一个epoch的平均值
    avgs = []
    for k, _ in enumerate(layers):
        tmp = []
        for i in range(overall_rounds):
            tmp.append(layer_cossim_overall[i][num_epochs-1][k][1])
        avgs.append(np.mean(tmp))
    
    # 找到最小的k层
    sorted_indices = sorted(enumerate(avgs), key=lambda x : x[1])
    unlearn_k_layers = [layers[idx] for idx, _ in sorted_indices[:unlearn_k]]

    for name, param in model.named_parameters():
        name = name.replace(".weight", "")
        name = name.replace(".bias", "")

        if name not in unlearn_k_layers:
            param.requires_grad = False

    