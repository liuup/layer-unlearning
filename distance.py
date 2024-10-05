import torch
import torch.nn.functional as F

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
def layer_cossim(model1, model2, layers):
    ans = []
    all_layers = [name for name, _ in model1.named_parameters()]
    for layer in layers:
        if (layer+"bias") in all_layers:
            layer_t_1 = torch.concat([model1.state_dict()[layer+".weight"].flatten(), 
                                    model1.state_dict()[layer+".bias"].flatten()])
            layer_t_2 = torch.concat([model2.state_dict()[layer+".weight"].flatten(), 
                                    model2.state_dict()[layer+".bias"].flatten()])
            ans.append((layer, F.cosine_similarity(layer_t_1.unsqueeze(0), layer_t_2.unsqueeze(0)).item()))
        else:
            layer_t_1 = model1.state_dict()[layer+".weight"].flatten()
            layer_t_2 = model2.state_dict()[layer+".weight"].flatten()
            ans.append((layer, F.cosine_similarity(layer_t_1.unsqueeze(0), layer_t_2.unsqueeze(0)).item()))
    return ans

# 测量两个模型层间的l1距离
def layer_l1(model1, model2):
    pass