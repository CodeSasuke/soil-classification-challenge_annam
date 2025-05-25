"""

Author: Siddhant Bhardwaj
Team Name: Siddhant Bhardwaj
Team Members: Siddhant, Sivadhanushya
Leaderboard Rank: 64
"""

import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

# Preprocessing module for soil classification
IMG_SIZE = 240
mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]

train_transforms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(45),
    T.ColorJitter(0.3,0.3,0.3,0.15),
    T.RandomAffine(0, translate=(0.1,0.1), scale=(0.9,1.1), shear=10),
    T.ToTensor(),
    T.Normalize(mean, std)
])

val_transforms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean, std)
])

class SoilDataset(Dataset):
    def __init__(self, df, image_dir, transforms=None, is_test=False):
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        self.is_test = is_test
        if not is_test:
            self.labels = df['label_int'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_id'])
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black')
        if self.transforms:
            image = self.transforms(image)
        if self.is_test:
            return image, row['image_id']
        return image, torch.tensor(self.labels[idx], dtype=torch.long)