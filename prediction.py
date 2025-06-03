import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import os

class EfficientNetB0Alpha(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=False, in_chans=1, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# Load model and class names once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = 'saved_models/best_model.pth'
num_classes = 26

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

model = EfficientNetB0Alpha(num_classes=num_classes).to(device)
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
class_names = checkpoint['class_names']

def predict_from_image(image_path):

    img = Image.open(image_path).convert('L')
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = class_names[predicted.item()]
        confidence = confidence.item()
    return predicted_class, confidence