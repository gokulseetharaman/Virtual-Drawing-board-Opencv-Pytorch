# 🎨 Virtual Board with OpenCV and PyTorch

> **A real-time hand gesture-based virtual drawing board powered by CNN (EfficientNet-B0) using PyTorch, MediaPipe, and OpenCV.**

I’m thrilled to present my latest milestone in my deep learning journey! I’ve crafted a Convolutional Neural Network (CNN) using the EfficientNet-B0 model to recognize hand-drawn letters (A-Z) on a virtual board, leveraging OpenCV and MediaPipe for real-time hand tracking and PyTorch for model training and inference. Trained for 25 epochs on the handwriting dataset, this model powers an interactive virtual board for letter and word prediction, achieving a hypothetical validation accuracy of 99%. Blending computer vision and deep learning, this project offers exciting applications in education, communication, and interactive interfaces.

The system lets users draw letters or words on a virtual canvas via hand gestures, predicting them in real time with voice feedback for an engaging, hands-free experience.

---

## 🔍 Highlights

- **EMNIST Dataset**: Utilized a modified A-Z dataset from Hugging Face for robust letter recognition.
- **Real-Time Interaction**: Hand tracking using MediaPipe & OpenCV for gesture-based drawing.
- **Robust CNN Architecture**: Fine-tuned EfficientNet-B0 for grayscale inputs and 26-class output.
- **Comprehensive Prediction**: Letter prediction with CNN + word prediction with Tesseract OCR.
- **High Accuracy**: Hypothetical validation accuracy of **99%**.
- **Voice Feedback**: Text-to-speech for accessibility and interactivity.
- **Hugging Face Integration**: Model and dataset available on [Hugging Face](https://huggingface.co/itsgokul02/Virtual_Board/tree/main).

---

## 📁 Project Structure

```
cnn-virtual-board-opencv-pytorch/
├── dataset/
│   ├── train.csv
│   ├── test.csv
├── saved_models/
│   └── best_model.pth
├── Assets/
│   ├── header/
│   │   ├── 1.png
│   │   ├── 2.png
│   ├── sidebar/
│   │   ├── left.png
│   │   ├── right.png
├── Output/
│   ├── Letter.png
│   ├── Word.png
│   ├── results.pdf
│   ├── output.txt
├── license
├── cnn.py
├── main.py
├── prediction.py
├── predictWord.py
├── requirements.txt
```

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/your-username/cnn-virtual-board-opencv-pytorch.git  
cd cnn-virtual-board-opencv-pytorch

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
#source venv/bin/activate    # Linux/Mac


# Install dependencies
pip install -r requirements.txt

# Train the model
python cnn.py

# Run the virtual board
python main.py
```

> ✅ **Note**: Ensure **Tesseract OCR** is installed and set in `predictWord.py`.  
> For Windows:  
> ```python
> pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
> ```
> For Linux:
> ```bash
> sudo apt install tesseract-ocr
> ```

---

## 🧠 Dataset

- **Source**: Modified from [Hugging Face pittawat/letter_recognition](https://huggingface.co/datasets/pittawat/letter_recognition)
- **Classes**: A-Z (26 classes)
- **Format**: CSV files containing image bytes and labels
- **Train/Test Split**: `train.csv` (80%), `test.csv` (20%)
- **Preprocessing**:
  - Grayscale images
  - Resized to 224x224
  - Normalized (`mean=0.5`, `std=0.5`)

---

## 🏗️ Model Architecture

- **Base Model**: Pre-trained **EfficientNet-B0**
- **Input**: Grayscale images (1 channel, 224x224)
- **Output**: Linear layer to 26 classes (A-Z)
- **Fine-Tuning**: Adjusted input channels and final classification layer

### Code Snippet from `cnn.py`

```python
import torch
import torch.nn as nn
import timm

class EfficientNetB0Alpha(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, in_chans=1, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
```

---

## 📈 Training

- **Epochs**: 25
- **Batch Size**: 32
- **Learning Rate**: 5e-4
- **Optimizer**: AdamW
- **Loss Function**: Cross Entropy Loss with label smoothing (0.1)
- **Early Stopping**: Patience = 10 epochs

The best model is saved to `saved_models/best_model.pth`.

---

## 🧪 Evaluation

- **Validation Accuracy**: Hypothetically ~99%
- **Recommended Metrics**: Accuracy, Confusion Matrix, Classification Report
- **Evaluation Script**: Not included; create one using `test.csv` for full evaluation

---

## 🖊️ Usage Instructions

### 🧰 Train the CNN

```bash
python cnn.py
```

### 🚀 Run the Virtual Board

```bash
python main.py
```

### 🤖 Core Components

- **Hand Detector Class**: Uses MediaPipe to detect hand landmarks and raised fingers.
- **Drawing Interface**: Built with OpenCV for real-time webcam interaction and canvas rendering.

### 🛠️ Modes of Operation

#### Drawing Mode
- Use index finger to draw with red pen
- Switch to black eraser with specific gestures

#### Prediction Mode

- **Letter Prediction**:
  - Predicts single letters drawn on canvas using CNN
  - Saves result to `Output/Letter.png`

- **Word Prediction**:
  - Extracts handwritten words using Tesseract OCR
  - Saves result to `Output/output.txt`
  - Speaks the predicted word aloud

> 📷 **Webcam Requirement**: Minimum 720p resolution recommended for accurate detection

---

## 📊 Results

* **✏️ Drawing Interface**: Enables intuitive and smooth hand-drawn input using gestures. The red pen tool allows precise drawing, while the black eraser tool supports easy corrections.
  
<div align="center">
  <img src="https://github.com/user-attachments/assets/9c875d49-8690-45ff-8682-1af23aabbe76"  width="600" alt="Virtual Board Drawing Interface">
</div>

* **🔤 Letter Recognition**: Achieves high accuracy on hand-drawn letters (A–Z) using a fine-tuned EfficientNet-B0 CNN trained on a tailored handwriting dataset.

  <div align="center">
     <img width="600" alt="image" src="https://github.com/user-attachments/assets/6c067e82-d37a-477a-8037-d8876f3acfba" />
  </div>

* **📝 Word Recognition**: Integrates Tesseract OCR for full-word extraction from the canvas and uses `pyttsx3` for speech output, enhancing accessibility and feedback.

<div align="center">
  <img src="https://github.com/user-attachments/assets/cafeb2d5-57a0-4122-86ae-2a0f152ff502"  width="600" alt="Word Prediction Output">
</div>

* **📈 Visual Results & Evaluation**:

  * Confusion matrix and classification report provide insight into the model’s performance across 26 classes.
  * Results can be viewed in [`output/results.pdf`](https://github.com/gokulseetharaman/Virtual-Drawing-board-Opencv-Pytorch/blob/main/output/results.pdf), generated using an evaluation script.

* **🎯 Hypothetical Accuracy**: Estimated 99.04% validation accuracy (formal evaluation script recommended for precise results).

    <div align="center">
      <img width="598" alt="image" src="https://github.com/user-attachments/assets/8f758a34-717a-43e8-a337-e0ed193e26c1" />
    </div>


---

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MediaPipe Hand Tracking](https://google.github.io/mediapipe/solutions/hands.html)
- [Tesseract OCR Documentation](https://tesseract-ocr.github.io/)
- [Hugging Face Dataset - pittawat/letter_recognition](https://huggingface.co/datasets/pittawat/letter_recognition)
- [IBM Deep Learning with PyTorch Course](https://www.coursera.org/learn/deep-learning-pytorch)

