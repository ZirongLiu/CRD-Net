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

        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.modality = modality
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        self.paths = []

        self.df = self.df[self.df.iloc[:, 2] == split]

        for _, row in self.df.iterrows():
            sample_name = row[0]
            label = row[1]
            sample_path = os.path.join(self.root_dir, str(label), sample_name, self.modality)

            if not os.path.isdir(sample_path):
                raise ValueError(f"Nan: {sample_path}")

            img_files = [f for f in os.listdir(sample_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not img_files:
                raise ValueError(f"No: {sample_path}")

            if split == "train":
                self.images.extend([os.path.join(sample_path, img_file) for img_file in img_files])
                self.labels.extend([label] * len(img_files))
                self.paths.extend([sample_name] * len(img_files))
            elif split == "test":
                if TestType == "single":
                    self.images.append(os.path.join(sample_path, img_files[0]))
                    self.labels.append(label)
                    self.paths.append(sample_name)
                elif TestType == "all":
                    self.images.extend([os.path.join(sample_path, img_file) for img_file in img_files])
                    self.labels.extend([label] * len(img_files))
                    self.paths.extend([sample_name] * len(img_files))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        sample_name = self.paths[idx]


        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label, img_path





def get_dataloaders(data_root, csv_file, modality, batch_size, model_name, TestType="single"):

    if model_name == 'flexivit_base':
        input_size = (240, 240)
    else:
        input_size = (224, 224)
        

    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomRotation(degrees=180),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


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
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.modality1 = modality1
        self.modality2 = modality2
        self.transform = transform
        self.test_type = test_type


        self.df = self.df[self.df.iloc[:, 2] == split]


        self.combinations = []
        for idx in range(len(self.df)):
            sample_name = self.df.iloc[idx, 0]
            label = self.df.iloc[idx, 1]

            path1 = os.path.join(self.root_dir, str(label), sample_name, self.modality1)
            path2 = os.path.join(self.root_dir, str(label), sample_name, self.modality2)

            files1 = sorted([os.path.join(path1, f) for f in os.listdir(path1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            files2 = sorted([os.path.join(path2, f) for f in os.listdir(path2) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            if self.test_type == "single" and split == "test":

                if files1 and files2:
                    self.combinations.append((files1[0], files2[0], label))
            else:

                for file1, file2 in itertools.product(files1, files2):
                    self.combinations.append((file1, file2, label))

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        file1, file2, label = self.combinations[idx]

        image1 = Image.open(file1).convert("RGB")
        image2 = Image.open(file2).convert("RGB")

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        path = f"{file1}_{file2}"

        return image1, image2, label, path
    


def get_multimodel_dataloaders(data_root, csv_file, batch_size, test_type="single"):
    input_size = (224, 224)

    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomRotation(degrees=180),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


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
        test_type="all" 
    )

    val_dataset = MultiModalDataset(
        root_dir=data_root,
        csv_file=csv_file,
        modality1="CF",
        modality2="FFA",
        split="test",
        transform=test_transform,
        test_type=test_type  
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader
