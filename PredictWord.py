import cv2
import pytesseract
import pyttsx3
import os
import subprocess
import time

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class PredictWord:
    def __init__(self, image_path):
        self.image_path = image_path

    def predict(self):
        image = cv2.imread(self.image_path)
        if image is None:
            print(f"Error: Image not found at '{self.image_path}'")
            return None

        # Convert to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        custom_config = r'--oem 3 --psm 6'
        word = pytesseract.image_to_string(gray, config=custom_config)
        return word.strip()  # 👈 This line is missing in your code

    @staticmethod
    def save_and_speak_word(word, output_dir='output', filename='output.txt'):
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(output_dir, filename))

        # Write word to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(word + '\n\n')  # Adds space after the w

        # Open file in Notepad
        subprocess.Popen(['notepad.exe', file_path])
        time.sleep(1)  # Give Notepad time to open

        # Speak the word
        engine = pyttsx3.init()
        engine.say(word)
        engine.runAndWait()

def clear_notepad_file(output_dir='output', filename='output.txt'):
    file_path = os.path.abspath(os.path.join(output_dir, filename))
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('')