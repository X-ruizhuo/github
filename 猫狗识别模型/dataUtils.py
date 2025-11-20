import os
import re
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split

# 自然排序函数，用于按数字大小排序

def natural_sort_key(s):
    """用于自然排序的键函数，将字符串中的数字部分提取出来作为数字比较"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]

# 无标签图片数据集类（用于测试集）

class UnlabeledImageDataset(Dataset):
    """用于处理无标签的图片数据集"""
    
    def __init__(self, root_dir, transform=None):
        """
        初始化无标签图片数据集
        
        Args:
            root_dir (str): 包含图片的根目录
            transform (callable, optional): 应用于图片的转换操作
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 获取所有图片路径
        self.image_paths = []
        for filename in os.listdir(root_dir):
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                self.image_paths.append(os.path.join(root_dir, filename))
        
        # 按文件名自然排序
        self.image_paths.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """获取指定索引的图片"""
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        filename = os.path.basename(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, filename

# 数据预处理器类

class DataPreprocessor:
    """用于处理猫狗识别数据集的数据预处理器"""
    
    def __init__(self, data_dir, batch_size=32, img_size=(224, 224)):
        """
        初始化数据预处理器
        
        Args:
            data_dir (str): 数据集根目录
            batch_size (int): 批次大小
            img_size (tuple): 图片大小（宽, 高）
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        
        # 定义数据转换
        self._define_transforms()
    
    def _define_transforms(self):
        """定义数据转换操作"""
        # 训练集转换（包含数据增强）
        self.train_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 验证集和测试集转换（不包含数据增强）
        self.val_test_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_train_loader(self):
        """
        获取训练集数据加载器
        
        Returns:
            DataLoader: 训练集数据加载器
        """
        train_dir = os.path.join(self.data_dir, 'train')
        train_dataset = datasets.ImageFolder(train_dir, transform=self.train_transform)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        return train_loader
    
    def get_validation_loader(self):
        """
        获取验证集数据加载器
        
        Returns:
            DataLoader: 验证集数据加载器
        """
        val_dir = os.path.join(self.data_dir, 'validation')
        val_dataset = datasets.ImageFolder(val_dir, transform=self.val_test_transform)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return val_loader
    
    def get_test_loader(self):
        """
        获取测试集数据加载器
        
        Returns:
            DataLoader: 测试集数据加载器
            list: 测试集图片路径列表（按加载顺序）
        """
        test_dir = os.path.join(self.data_dir, 'test')
        
        # 检查测试集目录结构
        if os.path.exists(test_dir):
            # 创建无标签图片数据集
            test_dataset = UnlabeledImageDataset(test_dir, transform=self.val_test_transform)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            
            # 返回测试集数据加载器和图片路径列表
            return test_loader, test_dataset.image_paths
        else:
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    def get_class_names(self):
        """
        获取类别名称列表
        
        Returns:
            list: 类别名称列表
        """
        train_dir = os.path.join(self.data_dir, 'train')
        if os.path.exists(train_dir):
            class_names = sorted(os.listdir(train_dir), key=natural_sort_key)
            return class_names
        else:
            raise FileNotFoundError(f"Train directory not found: {train_dir}")

# 使用示例
if __name__ == "__main__":
    # 初始化数据预处理器
    # 获取当前脚本所在目录
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "final_dataset")
    preprocessor = DataPreprocessor(data_dir, batch_size=32, img_size=(224, 224))
    
    # 获取数据加载器
    train_loader = preprocessor.get_train_loader()
    val_loader = preprocessor.get_validation_loader()
    test_loader, test_image_paths = preprocessor.get_test_loader()
    
    # 获取类别名称
    class_names = preprocessor.get_class_names()
    
    print(f"类别名称: {class_names}")
    print(f"训练集批次数量: {len(train_loader)}")
    print(f"验证集批次数量: {len(val_loader)}")
    print(f"测试集图片数量: {len(test_image_paths)}")
    print(f"前5张测试图片: {[os.path.basename(path) for path in test_image_paths[:5]]}")