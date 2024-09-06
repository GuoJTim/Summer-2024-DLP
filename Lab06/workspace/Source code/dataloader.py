import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class Dataset_iclevr(Dataset):
    def __init__(self, root_dir, label_file, object_file, transform=None, partial=1.0):
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.ToTensor()

        # 讀取 objects.json
        with open(object_file, 'r') as f:
            self.object_dict = json.load(f)

        # 讀取 train.json
        with open(label_file, 'r') as f:
            self.data = json.load(f)

        self.image_files = list(self.data.keys())
        self.labels = self._convert_labels(self.data)
        self.partial = partial

    def _convert_labels(self, data):
        labels = []
        for key in data.keys():
            label = np.zeros(len(self.object_dict))
            for obj in data[key]:
                label[self.object_dict[obj]] = 1
            labels.append(label)
        return labels

    def __len__(self):
        return int(len(self.image_files) * self.partial)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label

def create_dataloaders(root_dir, label_file, object_file, transform=None, batch_size=32, val_split=0,partial = 1.0,num_workers = 4):
    dataset = Dataset_iclevr(root_dir, label_file, object_file, transform=transform,partial=partial)
    
    # 計算訓練和驗證集的大小
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)

    return train_loader, val_loader

# Example usage:
# train_loader, val_loader = create_dataloaders('/iclevr', 'train.json', 'objects.json')

if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders('iclevr', 'train.json', 'objects.json',batch_size=32)
    
    for images, labels in train_loader:
        print(images.shape)
        print(labels.shape)