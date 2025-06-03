import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import io
import ast
import os
from sklearn.metrics import confusion_matrix, classification_report
import timm

class EfficientNetB0Alpha(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, in_chans=1, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

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
            img_dict = ast.literal_eval(img_data)
            img_bytes = img_dict['bytes']
        else:
            img_bytes = img_data['bytes']
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, label

def load_model(model_path, num_classes, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = EfficientNetB0Alpha(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def evaluate(model, test_loader, device, class_names):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = 100 * correct / max(total, 1)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=2))
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix (True Labels: rows, Predicted Labels: columns):")
    print(pd.DataFrame(cm, index=class_names, columns=class_names))
    return accuracy, cm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 26
    model_path = "saved_models/best_model.pth"
    test_csv = "dataset/test.csv"
    batch_size = 32
    print("Device being used:", device)
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    test_dataset = Dataset(test_csv, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    class_names = [chr(65 + i) for i in range(26)]
    model = load_model(model_path, num_classes, device)
    evaluate(model, test_loader, device, class_names)

if __name__ == "__main__":
    main()