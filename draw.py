import numpy as np
import matplotlib.pyplot as plt


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
    std = np.array([])    # 方差
    for j in range(num_epochs):
        tmp = []
        for i in range(overall_rounds):
            tmp.append(data_overall[i][j])
        std = np.append(std, np.std(tmp))
    return std


# 绘制模型间的余弦相似度图像
def models_cossim(overall_rounds, num_epochs, model1base_cossim_overall, model2base_cossim_overall, model12_cossim_overall):
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
def models_loss(overall_rounds, num_epochs, train_loss_1_overall, val_loss_1_overall, train_loss_2_overall, val_loss_2_overall):
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
  