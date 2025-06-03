import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import io
import ast
import timm
import os

# Model
class EfficientNetB0Alpha(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, in_chans=1, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# Dataset
class Dataset(Dataset):
    def __init__(self, csv_path, transform=None, image_col='image', label_col='label'):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data.iloc[idx][self.image_col]
        label = self.data.iloc[idx][self.label_col]
        if isinstance(img_data, str):
            try:
                img_dict = ast.literal_eval(img_data)
                img_bytes = img_dict['bytes']
            except (ValueError, SyntaxError, KeyError) as e:
                raise ValueError(f"Error parsing image data at index {idx}: {e}")
        else:
            img_bytes = img_data['bytes']
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('L')
        except Exception as e:
            raise ValueError(f"Error decoding image at index {idx}: {e}")
        if self.transform:
            img = self.transform(img)
        return img, label

# Training function
def train(model, train_loader, optimizer, criterion, scheduler, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += targets.size(0)
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * total_correct / total_samples
    return avg_loss, accuracy

# Validation function
def val(model, val_loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * total_correct / total_samples
    return avg_loss, accuracy

# Save function
def save(model, optimizer, epoch, accuracy, class_names, save_path="saved_models/best_model.pth"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'class_names': class_names
    }, save_path)

# Main method
def main():
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 26
    batch_size = 32
    learning_rate = 5e-4
    num_epochs = 25
    patience = 10
    train_csv = "dataset/train.csv"
    val_csv = "dataset/test.csv"
    save_path = "saved_models/best_model.pth"

    print("Device being used:", device)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Datasets and loaders
    train_dataset = Dataset(train_csv, transform=train_transform)
    val_dataset = Dataset(val_csv, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    class_names = [chr(65 + i) for i in range(26)]  # ['A', 'B', ..., 'Z']

    # Model, optimizer, criterion
    model = EfficientNetB0Alpha(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Load checkpoint if it exists
    start_epoch = 0
    best_accuracy = 0.0
    if os.path.exists(save_path):
        try:
            checkpoint = torch.load(save_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_accuracy = checkpoint['accuracy']
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with accuracy {best_accuracy:.2f}%")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
    else:
        print(f"No checkpoint found at {save_path}. Starting from scratch.")

    # Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # Training loop
    early_stopping_counter = 0
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        val_loss, val_acc = val(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            save(model, optimizer, epoch, best_accuracy, class_names, save_path)
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Early stopping triggered. Best accuracy: {best_accuracy:.2f}%")
            break

    print(f"Training completed. Best validation accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    main()