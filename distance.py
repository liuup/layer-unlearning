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
def model_layer_cossim(model1, model2):
    pass

# 测量两个模型层间的l1距离
def model_layer_l1(model1, model2):
    pass