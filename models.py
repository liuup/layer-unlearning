import torch
import torch.nn as nn
import torchvision

from sklearn.metrics import f1_score, recall_score



# 初始化修改后的resnet18模型
def get_resnet18():
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # 首层改成3x3卷积核
    model.maxpool = nn.MaxPool2d(1, 1, 0)  # 图像太小 本来就没什么特征 所以这里通过1x1的池化核让池化层失效
    return model

# 初始化一个cnn
def get_cnn():
    # 初始用的cnn，可以拿来测试用
    class TestCNN(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 10),
            )

        def forward(self, x):
            return self.cnn(x)
        
    model = TestCNN()
    return model

# 获取模型的参数量大小
def get_model_params_amount(model):
    return sum(p.numel() for p in model.parameters())


# 训练
def train(model, loss_fn, optimizer, trainloader, computing_device):
    # training
    num_batches = len(trainloader)
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(trainloader):
        X, y = X.to(computing_device), y.to(computing_device)
        optimizer.zero_grad()
        
        predict = model(X)

        loss = loss_fn(predict, y)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        
    train_loss /= num_batches

    return train_loss

# 验证
def val(model, loss_fn, valloader, computing_device):
    size = len(valloader.dataset)
    num_batches = len(valloader)
    
    model.eval()
    val_loss = 0
    real_labels = []
    pre_labels = []
    with torch.no_grad():        
        for batch, (X, y) in enumerate(valloader):
            X, y = X.to(computing_device), y.to(computing_device)

            predict = model(X)
            loss = loss_fn(predict, y)
            val_loss += loss.item()
            # val_correct += (predict.argmax(1) == y).type(torch.float).sum().item() 
            real_labels.extend(y.cpu().numpy())
            pre_labels.extend(predict.argmax(1).cpu().numpy())
            
    val_loss /= num_batches
    # val_correct /= size
    
    f1 = f1_score(real_labels, pre_labels, average='weighted')
    recall = recall_score(real_labels, pre_labels, average='weighted')
    
    # overall_f1 = f1_score(y_true, y_pred, average='weighted')
    # overall_recall = recall_score(y_true, y_pred, average='weighted')

    return val_loss, f1, recall

# 测试
def test(model, loss_fn, testloader, computing_device):
    size = len(testloader.dataset)
    
    num_batches = len(testloader)
    
    model.eval()
    test_loss = 0
    real_labels = []    # 真实标签
    pre_labels = [] # 预测标签
    with torch.no_grad():
        for batch, (X, y) in enumerate(testloader):
            X, y = X.to(computing_device), y.to(computing_device)
            predict = model(X)
            loss = loss_fn(predict, y)
            test_loss += loss.item()
            # test_correct += (predict.argmax(1) == y).type(torch.float).sum().item()
            
            # for tmpy in y.cpu().numpy():
            real_labels.extend(y.cpu().numpy())
            # for tmpp in predict.argmax(1).cpu().numpy():
            pre_labels.extend(predict.argmax(1).cpu().numpy())
    
    test_loss /= num_batches
    
    return test_loss, real_labels, pre_labels
