import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from augmentations import augment_image
from dataset import FruitDataset
from train import accuracy



#testing
test_data = datasets.ImageFolder(root=tsPath)
test_dataset = FruitDataset(test_data, augment=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)      

def testing(test_loader, model):
    correct_preds = 0.0
    #acc = 0.0
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            correct_preds += accuracy(outputs, labels)
        acc = correct_preds/len(test_dataset)
    return acc