import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score

draw_scale_factor = 1
draw_dpi = 300


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
    std = np.array([])    # 标准误差
    for j in range(num_epochs):
        tmp = []
        for i in range(overall_rounds):
            tmp.append(data_overall[i][j])
        std = np.append(std, np.std(tmp))
    return std


# 绘制模型间的余弦相似度图像
# TODO: 参数换成cossim_overall list会方便很多
def models_cossim(overall_rounds, num_epochs, model1base_cossim_overall, model2base_cossim_overall, model12_cossim_overall, picname):
    epochs = [(i+1) for i in range(num_epochs)]

    model1base_cossim_avg = calc_avg(overall_rounds, num_epochs, model1base_cossim_overall)
    model1base_cossim_std = calc_std(overall_rounds, num_epochs, model1base_cossim_overall)

    model2base_cossim_avg = calc_avg(overall_rounds, num_epochs, model2base_cossim_overall)
    model2base_cossim_std = calc_std(overall_rounds, num_epochs, model2base_cossim_overall)

    model12_cossim_avg = calc_avg(overall_rounds, num_epochs, model12_cossim_overall)
    model12_cossim_std = calc_std(overall_rounds, num_epochs, model12_cossim_overall)

    # 创建图形
    plt.figure(dpi=draw_dpi)

    plt.plot(epochs, model1base_cossim_avg, color='orange', linewidth=0.5, label='model1base_cossim')
    plt.fill_between(epochs, model1base_cossim_avg - draw_scale_factor * model1base_cossim_std, model1base_cossim_avg + draw_scale_factor * model1base_cossim_std, color='orange', alpha=0.3, edgecolor='none')

    plt.plot(epochs, model2base_cossim_avg, color='blue', linewidth=0.5, label='model2base_cossim')
    plt.fill_between(epochs, model2base_cossim_avg - draw_scale_factor * model2base_cossim_std, model2base_cossim_avg + draw_scale_factor * model2base_cossim_std, color='blue', alpha=0.3, edgecolor='none')
    
    plt.plot(epochs, model12_cossim_avg, color='red', linewidth=0.5, label='model12_cossim')
    plt.fill_between(epochs, model12_cossim_avg - draw_scale_factor * model12_cossim_std, model12_cossim_avg + draw_scale_factor * model12_cossim_std, color='red', alpha=0.3, edgecolor='none')

    # 添加标签和标题
    plt.xlabel('Epochs')
    plt.ylabel('cossim')
    # plt.title('benign_models')

    # 添加图例
    plt.legend()

    path = f"./figs/{picname}.png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    
    print(f"draw [{path}] finished")


# 绘制训练损失曲线
def models_loss(overall_rounds, num_epochs, train_loss_1_overall, val_loss_1_overall, train_loss_2_overall, val_loss_2_overall, picname):
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
    
    plt.plot(epochs, train_loss_1_avg, color='orange', linewidth=0.5, label='model_1_train_loss')
    plt.fill_between(epochs, train_loss_1_avg - draw_scale_factor * train_loss_1_std, train_loss_1_avg + draw_scale_factor * train_loss_1_std, color='orange', alpha=0.3, edgecolor='none')
    
    plt.plot(epochs, val_loss_1_avg, color='blue', linewidth=0.5, label='model_1_val_loss')
    plt.fill_between(epochs, val_loss_1_avg - draw_scale_factor * val_loss_1_std, val_loss_1_avg + draw_scale_factor * val_loss_1_std, color='blue', alpha=0.3, edgecolor='none')
    
    plt.plot(epochs, train_loss_2_avg, color='red', linewidth=0.5, label='model_2_train_loss')
    plt.fill_between(epochs, train_loss_2_avg - draw_scale_factor * train_loss_2_std, train_loss_2_avg + draw_scale_factor * train_loss_2_std, color='red', alpha=0.3, edgecolor='none')
    
    plt.plot(epochs, val_loss_2_avg, color='green', linewidth=0.5, label='model_2_val_loss')
    plt.fill_between(epochs, val_loss_2_avg - draw_scale_factor * val_loss_2_std, val_loss_2_avg + draw_scale_factor * val_loss_2_std, color='green', alpha=0.3, edgecolor='none')

    # 添加标签和标题
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    # plt.title('models loss')

    # 添加图例
    plt.legend()

    path = f"./figs/{picname}.png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    
    print(f"draw [{path}] finished")


# 模型层间偏移的图
def layers_cossim(layer_cossim_overall, last_k, picname):
    overall_rounds = len(layer_cossim_overall)
    num_epochs = len(layer_cossim_overall[0])
    
    epochs = [(i+1) for i in range(num_epochs)]

    # 获取所有的层
    layers = [name for name, _ in layer_cossim_overall[0][0]]

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
    
    # k_colors = plt.get_cmap('bwr', len(k_indices)) # 自定义颜色

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

    plt.xlabel('epochs')
    plt.ylabel('cossim')
    # plt.title('model layers')
    plt.legend()

    path = f"./figs/{picname}.png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    print(f"draw [{path}] finished")


# 绘制模型层间偏移的柱状图
# TODO: 再加上模型每一层的参数量
def bar_graph(model, layer_cossim_overall, picname):
    overall_rounds = len(layer_cossim_overall)
    num_epochs = len(layer_cossim_overall[0])
    
    # 获取所有的层
    layers = [name for name, _ in layer_cossim_overall[0][0]]
    index = np.arange(len(layers))
    params = [0 for _ in range(len(layers))]
    
    # 获取所有层的参数量
    params_amount = {}
    for name, param in model.named_parameters():
        params_amount[name] = param.numel()
    for idx, layer in enumerate(layers):
        if layer+".weight" in params_amount.keys():
            params[idx] += params_amount[layer+".weight"]
        if layer+".bias" in params_amount.keys():
            params[idx] += params_amount[layer+".bias"]
    
   # 计算列表中的最大值和最小值
    max_value = max(params)
    min_value = min(params)
    normalized_params = [(i - min_value) / (max_value - min_value) for i in params]   # 归一化处理
    
    _, ax = plt.subplots(figsize=(6, 8), dpi=300)   # 创建图形和子图

    avgs = []
    stds = []
    for k, _ in enumerate(layers):
        tmp = []            
        for i in range(overall_rounds):
            tmp.append(layer_cossim_overall[i][num_epochs-1][k][1])
        avgs.append(np.mean(tmp))
        stds.append(np.std(tmp))

    for i in range(len(avgs)):
        avgs[i] = 1 - avgs[i]
        
    bar_height = 0.4
    ax.barh(index, avgs, label='layer shift', height=bar_height)
    ax.barh(index + bar_height, normalized_params, label='layer params amount', height=bar_height)

    # plt.title('Shift distance by layer')
    ax.set_xlabel('Delta Shift distance')
    ax.set_ylabel('Layer')
    ax.set_yticks(index + bar_height / 2)  # 刻度放在两个柱的中间位置
    ax.set_yticklabels(layers)
    ax.legend()

    path = f"./figs/{picname}.png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    print(f"draw [{path}] finished")



# TODO: 似乎不用每一轮结束后都输出混淆矩阵
def confusion_mat(real_labels, pre_labels, figname):
    cm = confusion_matrix(real_labels, pre_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    # plt.show()
    
    path = f"./figs/{figname}.png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)

    # 计算总体F1和召回率
    # overall_f1 = f1_score(real_labels, pre_labels, average='weighted')
    # overall_recall = recall_score(real_labels, pre_labels, average='weighted')

    # # 计算各类别的F1和召回率
    # f1_per_class = f1_score(real_labels, pre_labels, average=None)
    # recall_per_class = recall_score(real_labels, pre_labels, average=None)

    # print("Overall F1 Score:", overall_f1)
    # print("Overall Recall:", overall_recall)
    # print("F1 Score per class:", f1_per_class)
    # print("Recall per class:", recall_per_class)