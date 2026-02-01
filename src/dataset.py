
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from augmentations import augment_image

class FruitDataset(Dataset):
    def __init__(self, data, augment=False, image_size=224):
        self.data = data
        self.samples = data.samples
        self.augment = augment
        self.image_size = image_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std = [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to Load the Image{img_path}")
        
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        if self.augment:
            image = augment_image(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)/255

        image = np.transpose(image, (2,0,1))

        image = torch.from_numpy(image)
        image = self.normalize(image)
        
        return(image, label)
    



#Dataloading
train_data = datasets.ImageFolder(root=trPath)
full_dataset = FruitDataset(train_data, augment=True)

#Creating Datasets
train_size = int(0.9*len(full_dataset))
val_size = len(train_data) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_dataset.dataset.augment = True
val_dataset.dataset.augment = False

#Dataloaders
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=4, pin_memory=True)
