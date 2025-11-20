import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataUtils import DataPreprocessor

# 简单CNN模型

class SimpleCNN(nn.Module):
    """简单的CNN模型"""
    
    def __init__(self, num_classes=2):
        """
        初始化简单CNN模型
        
        Args:
            num_classes (int): 类别数量
        """
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # 假设输入图片大小为224x224
        self.fc2 = nn.Linear(512, num_classes)
        
        # 激活函数
        self.relu = nn.ReLU()
        
        #  dropout层
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量
        
        Returns:
            torch.Tensor: 输出张量
        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # 展平
        x = x.view(-1, 128 * 28 * 28)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 改进的CNN模型

class DogCatClassifier(nn.Module):
    """改进的CNN模型，使用预训练的ResNet18作为特征提取器"""
    
    def __init__(self, num_classes=2):
        """
        初始化改进的CNN模型
        
        Args:
            num_classes (int): 类别数量
        """
        super(DogCatClassifier, self).__init__()
        
        # 使用预训练的ResNet18作为特征提取器
        self.feature_extractor = models.resnet18(pretrained=True)
        
        # 冻结特征提取器的参数
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # 替换最后的全连接层
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量
        
        Returns:
            torch.Tensor: 输出张量
        """
        return self.feature_extractor(x)

# 训练模型

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    """
    训练模型
    
    Args:
        model (nn.Module): 要训练的模型
        train_loader (DataLoader): 训练集数据加载器
        val_loader (DataLoader): 验证集数据加载器
        criterion (nn.Module): 损失函数
        optimizer (optim.Optimizer): 优化器
        num_epochs (int): 训练轮数
        device (str): 训练设备
    
    Returns:
        dict: 训练历史记录
    """
    # 将模型移动到指定设备
    model.to(device)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 开始训练
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # 使用tqdm显示训练进度
        train_pbar = tqdm(train_loader, desc="Training")
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计损失和准确率
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # 更新进度条
            train_pbar.set_postfix({"Loss": "{:.4f}".format(loss.item())})
        
        # 计算训练集损失和准确率
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        # 使用tqdm显示验证进度
        val_pbar = tqdm(val_loader, desc="Validation")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 前向传播
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # 统计损失和准确率
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
                # 更新进度条
                val_pbar.set_postfix({"Loss": "{:.4f}".format(loss.item())})
        
        # 计算验证集损失和准确率
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)
        
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())
        
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
        print()
    
    return history

# 评估模型

def evaluate_model(model, dataloader, criterion, device='cuda'):
    """
    评估模型
    
    Args:
        model (nn.Module): 要评估的模型
        dataloader (DataLoader): 评估数据集的数据加载器
        criterion (nn.Module): 损失函数
        device (str): 评估设备
    
    Returns:
        tuple: (损失, 准确率)
    """
    # 将模型移动到指定设备
    model.to(device)
    
    # 设置模型为评估模式
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # 统计损失和准确率
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    # 计算平均损失和准确率
    loss = running_loss / len(dataloader.dataset)
    acc = running_corrects.double() / len(dataloader.dataset)
    
    return loss, acc

# 预测函数

def predict(model, dataloader, class_names, device='cuda'):
    """
    预测函数
    
    Args:
        model (nn.Module): 训练好的模型
        dataloader (DataLoader): 测试集数据加载器
        class_names (list): 类别名称列表
        device (str): 预测设备
    
    Returns:
        list: 预测结果列表，每个元素包含(图片文件名, 预测类别, 预测概率)
    """
    # 将模型移动到指定设备
    model.to(device)
    
    # 设置模型为评估模式
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for inputs, filenames in tqdm(dataloader, desc="Predicting"):
            inputs = inputs.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算概率
            probs = torch.softmax(outputs, dim=1)
            
            # 获取预测类别和概率
            _, preds = torch.max(outputs, 1)
            
            # 保存预测结果
            for i in range(len(filenames)):
                filename = filenames[i]
                pred_class = class_names[preds[i]]
                pred_prob = probs[i][preds[i]].item() * 100  # 转换为百分比
                
                predictions.append((filename, pred_class, pred_prob))
    
    return predictions

# 绘制训练历史

def plot_history(history):
    """
    绘制训练历史
    
    Args:
        history (dict): 训练历史记录
    """
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

# 保存预测结果

def save_predictions(predictions, output_file='predictions.txt'):
    """
    保存预测结果到文件
    
    Args:
        predictions (list): 预测结果列表
        output_file (str): 输出文件名
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write('图片名称,预测类别,概率\n')
        
        # 写入预测结果
        for filename, pred_class, pred_prob in predictions:
            f.write(f"{filename},{pred_class},{pred_prob:.2f}%\n")

# 主函数

if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据集目录
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "final_dataset")
    
    # 初始化数据预处理器
    preprocessor = DataPreprocessor(data_dir, batch_size=32, img_size=(224, 224))
    
    # 获取数据加载器
    train_loader = preprocessor.get_train_loader()
    val_loader = preprocessor.get_validation_loader()
    test_loader, test_image_paths = preprocessor.get_test_loader()
    
    # 获取类别名称
    class_names = preprocessor.get_class_names()
    print(f"类别名称: {class_names}")
    
    # 创建模型
    # 可以选择使用SimpleCNN或DogCatClassifier
    # model = SimpleCNN(num_classes=2)
    model = DogCatClassifier(num_classes=2)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 设置训练参数
    num_epochs = 10
    
    # 训练模型
    print("开始训练模型...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, device=device)
    
    # 绘制训练历史
    plot_history(history)
    
    # 评估模型
    print("评估模型...")
    train_loss, train_acc = evaluate_model(model, train_loader, criterion, device=device)
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device=device)
    
    print(f"训练集损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
    print(f"验证集损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")
    
    # 保存模型
    model_path = "dogcat_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")
    
    # 在测试集上进行预测
    print("在测试集上进行预测...")
    predictions = predict(model, test_loader, class_names, device=device)
    
    # 保存预测结果
    save_predictions(predictions, output_file='predictions.txt')
    print(f"预测结果已保存到: predictions.txt")
    
    # 打印部分预测结果
    print("\n部分预测结果:")
    for i in range(min(5, len(predictions))):
        filename, pred_class, pred_prob = predictions[i]
        print(f"图片: {filename}, 预测: {pred_class}, 概率: {pred_prob:.2f}%")
