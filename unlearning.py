import torch.nn as nn
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
def adjust_lsc_euk(model, unlearn_k, layer_cossim_once):
    layers = [name for name, _ in model.named_parameters()]
    
    # 从小到大进行排序，从最后一个epoch中找出偏移量最大的unlearn_k层
    unlearn_k_data = sorted(layer_cossim_once[-1], key=lambda x: x[1])[:unlearn_k]
    tmp = [name for name, _ in unlearn_k_data]
    
    unlearn_k_layers = []
    for t in tmp:
        if t+".weight" in layers:
            unlearn_k_layers.append(t+".weight")
        if t+".bias" in layers:
            unlearn_k_layers.append(t+".bias")

    for name, param in model.named_parameters():
        if name not in unlearn_k_layers:
            param.requires_grad = False # 取消梯度
        else:
            nn.init.normal_(param, mean=0, std=0.1)


# lsc_cfk 按照层间偏移从大到小的顺序，继续训练偏移最大的k层
def adjust_lsc_cfk(model, unlearn_k, layer_cossim_once):
    layers = [name for name, _ in model.named_parameters()]

    # 从小到大进行排序，从最后一个epoch中找出偏移量最大的unlearn_k层
    unlearn_k_data = sorted(layer_cossim_once[-1], key=lambda x: x[1])[:unlearn_k]
    tmp = [name for name, _ in unlearn_k_data]

    unlearn_k_layers = []
    for t in tmp:
        if t+".weight" in layers:
            unlearn_k_layers.append(t+".weight")
        if t+".bias" in layers:
            unlearn_k_layers.append(t+".bias")

    for name, param in model.named_parameters():
        if name not in unlearn_k_layers:
            param.requires_grad = False # 取消梯度

    