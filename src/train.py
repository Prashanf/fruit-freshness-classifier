import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim


#accuracy
def accuracy(preds, labels):
    _, indx = torch.max(preds, dim=1)
    accuracy = (indx==labels).sum().item()
    return accuracy


#training loop
def training_loop(model, dataloader, optimizer, loss_func, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_func(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss = running_loss + loss.item()
        running_acc = running_acc + accuracy(outputs, labels)
        
    epoch_loss = running_loss/len(dataloader)
    epoch_acc = running_acc/len(dataloader.dataset)
    return(epoch_loss, epoch_acc)

#validation loop
def validation_loop(model, dataloader,loss_func, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_func(outputs, labels)
    
            running_loss += loss.item()
            running_acc += accuracy(outputs, labels)
    
    epoch_loss = running_loss/len(dataloader)
    epoch_acc = running_acc/len(dataloader.dataset)
    return(epoch_loss, epoch_acc)


#Full Traning Loop
def train_model(model, train_loader, val_loader, epochs, learning_rate=1e-4,device='cuda'):
    model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_acc = training_loop(model, train_loader, optimizer, loss_func, device)
        val_loss, val_acc = validation_loop(model, val_loader, loss_func, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:{val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")