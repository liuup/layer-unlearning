import torch.nn as nn


# EuK unlearning，重新训练后k层
def adjust_euk(model, k):
    all_layers = [name for name, _ in model.named_parameters()]
    cancel_gradient = []
    for i in range(len(all_layers) - k * 2):    # x2的原因是因为既有weight也有bias
        cancel_gradient.append(all_layers[i])

    for name, param in model.named_parameters():
        if name in cancel_gradient:
            param.requires_grad = False  # 前面的层取消梯度
        else:
            nn.init.normal_(param, mean=0.0, std=1) # 后面的层随机初始化参数

   
# CfK unlearning，继续训练后k层
def adjust_cfk(model, k):
    all_layers = [name for name, _ in model.named_parameters()]
    cancel_gradient = []
    for i in range(len(all_layers) - k * 2):    # x2的原因是因为既有weight也有bias
        cancel_gradient.append(all_layers[i])
    count = 0
    for name, param in model.named_parameters():    # 取消梯度
        if name in cancel_gradient:
            param.requires_grad = False
            count += 1  
            if count == len(cancel_gradient):   # 提前退出
                break

# lsc_euk 按照层间相似度从大到小的顺序，再训练偏移最大的k层
def adjust_lsc_euk(model, k, layers):
    pass


# lsc_cfk 按照层间相似度从大到小的顺序，随机初始化偏移最大的k层
def adjust_lsc_cfk(model, k, layers):
    pass