import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import itertools

class CF_FFA_Dataset(Dataset):
    def __init__(self, root_dir, csv_file, modality="CF", split="train", transform=None, TestType="single"):
        """
        Args:
            root_dir (string): 数据集根目录，类似于 "dataset/"
            csv_file (string): CSV 文件路径，记录样本的名称、标签和训练/测试划分
            modality (string): "CF" 或 "FFA"，表示加载哪种模态的数据
            split (string): "train" 或 "test"，加载训练集或测试集
            transform (callable, optional): 图像变换函数，默认为 None
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.modality = modality
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        self.paths = []

        # 根据 split 过滤数据
        self.df = self.df[self.df.iloc[:, 2] == split]

        # 遍历每个样本，并加载相应的图片路径和标签
        for _, row in self.df.iterrows():
            sample_name = row[0]
            label = row[1]
            sample_path = os.path.join(self.root_dir, str(label), sample_name, self.modality)

            if not os.path.isdir(sample_path):
                raise ValueError(f"路径无效: {sample_path}")

            # 获取文件夹内的图片文件名
            img_files = [f for f in os.listdir(sample_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not img_files:
                raise ValueError(f"找不到图片文件: {sample_path}")

            # # 根据训练或测试的不同需求，处理图片列表
            if split == "train":
                # 训练时使用所有图片
                self.images.extend([os.path.join(sample_path, img_file) for img_file in img_files])
                self.labels.extend([label] * len(img_files))
                self.paths.extend([sample_name] * len(img_files))
            elif split == "test":
                # 根据训练或测试的不同需求，处理图片列表
                if TestType == "single":
                    # 测试时只使用第一张图片
                    self.images.append(os.path.join(sample_path, img_files[0]))
                    self.labels.append(label)
                    self.paths.append(sample_name)
                elif TestType == "all":
                    # 使用所有图片
                    self.images.extend([os.path.join(sample_path, img_file) for img_file in img_files])
                    self.labels.extend([label] * len(img_files))
                    self.paths.extend([sample_name] * len(img_files))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): 数据索引

        Returns:
            image (Tensor): 处理后的图像
            label (int): 图像的类别标签
            img_path (str): 图像的路径
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        sample_name = self.paths[idx]

        # 加载图像
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label, img_path





def get_dataloaders(data_root, csv_file, modality, batch_size, model_name, TestType="single"):
    # 根据模型选择不同的输入尺寸
    if model_name == 'flexivit_base':
        input_size = (240, 240)
    else:
        input_size = (224, 224)
        
    # 训练数据变换（包含旋转等增强）
    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomRotation(degrees=180),  # 仅对训练数据生效
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 测试数据变换（仅做基本预处理）
    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CF_FFA_Dataset(
        root_dir=data_root,
        csv_file=csv_file,
        modality=modality,
        split="train",
        transform=train_transform, TestType=TestType
    )

    val_dataset = CF_FFA_Dataset(
        root_dir=data_root,
        csv_file=csv_file,
        modality=modality,
        split="test",
        transform=test_transform, TestType=TestType
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader




class MultiModalDataset(Dataset):
    def __init__(self, root_dir, csv_file, modality1="CF", modality2="FFA", split="train", transform=None, test_type="single"):
        """
        Args:
            root_dir (string): 数据集根目录，类似于 "dataset/"
            csv_file (string): CSV 文件路径，记录样本的名称、标签和训练/测试划分
            modality1, modality2 (string): 两种模态的目录名称，如 "CF" 和 "FFA"
            split (string): "train" 或 "test"，加载训练集或测试集
            transform (callable, optional): 图像变换函数，默认为 None
            test_type (string): "single" 表示只返回第一个组合；"all" 表示返回所有排列组合
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.modality1 = modality1
        self.modality2 = modality2
        self.transform = transform
        self.test_type = test_type

        # 根据 split 过滤数据
        self.df = self.df[self.df.iloc[:, 2] == split]

        # 预先构建所有样本的排列组合索引
        self.combinations = []
        for idx in range(len(self.df)):
            sample_name = self.df.iloc[idx, 0]
            label = self.df.iloc[idx, 1]

            # 获取当前 sample 的模态文件夹路径
            path1 = os.path.join(self.root_dir, str(label), sample_name, self.modality1)
            path2 = os.path.join(self.root_dir, str(label), sample_name, self.modality2)

            # 获取文件夹内所有图像文件名
            files1 = sorted([os.path.join(path1, f) for f in os.listdir(path1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            files2 = sorted([os.path.join(path2, f) for f in os.listdir(path2) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            if self.test_type == "single" and split == "test":
                # 仅返回第一对组合
                if files1 and files2:
                    self.combinations.append((files1[0], files2[0], label))
            else:
                # 返回所有排列组合
                for file1, file2 in itertools.product(files1, files2):
                    self.combinations.append((file1, file2, label))

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): 数据索引

        Returns:
            image1 (Tensor): 第一模态的图像
            image2 (Tensor): 第二模态的图像
            label (int): 图像的类别标签
            img_paths (tuple): 两个图像的路径
        """
        file1, file2, label = self.combinations[idx]

        # 加载两张图像
        image1 = Image.open(file1).convert("RGB")
        image2 = Image.open(file2).convert("RGB")

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        path = f"{file1}_{file2}"

        return image1, image2, label, path
    


def get_multimodel_dataloaders(data_root, csv_file, batch_size, test_type="single"):
    # 根据模型选择不同的输入尺寸
    input_size = (224, 224)

    # 训练数据变换
    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomRotation(degrees=180),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 测试数据变换
    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MultiModalDataset(
        root_dir=data_root,
        csv_file=csv_file,
        modality1="CF",
        modality2="FFA",
        split="train",
        transform=train_transform,
        test_type="all"  # 训练时始终使用所有组合
    )

    val_dataset = MultiModalDataset(
        root_dir=data_root,
        csv_file=csv_file,
        modality1="CF",
        modality2="FFA",
        split="test",
        transform=test_transform,
        test_type=test_type  # 根据 test_type 决定测试时返回第一个组合或所有组合
    )

    # 训练集使用分布式采样器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 测试集不使用分布式采样器
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader
