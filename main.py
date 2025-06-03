import os
import time
import numpy as np
import cv2
import mediapipe as mp
from prediction import predict_from_image
from PredictWord import PredictWord, clear_notepad_file

Header_path = "Assets/header"
myList = os.listdir(Header_path)
cam = cv2.VideoCapture(0)
wCam, hCam = 1280, 720


class HandDetector:
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.8, trackCon=0.8):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon,
                                        self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []

    def findHands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
        return self.lmList

    def fingerup(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def main():
    detector = HandDetector()
    cTime = 0

    overlayList = []
    drawColor = (0, 0, 255)
    for impath in myList:
        image = cv2.imread(f'{Header_path}/{impath}')
        if image is not None:
            overlayList.append(image)
    header = None
    if overlayList:
        header = cv2.resize(overlayList[0], (1280, 125)) if overlayList[0].shape != (125, 1280, 3) else overlayList[0]

    RightBar = cv2.imread('Assets/sidebar/right.png')
    RightBar = cv2.resize(RightBar, (230, 595))
    LeftBar = cv2.imread('Assets/sidebar/left.png')
    LeftBar = cv2.resize(LeftBar, (226, 300))

    mode = "Drawing Mode"
    canvas = np.zeros((720, 1280, 3), np.uint8)
    submode = "Letter_Prediction"
    predicted_letter = ""
    clear_notepad_file(output_dir='output', filename='output.txt')
    xp, yp = 0, 0
    while True:
        success, img = cam.read()
        img = cv2.resize(img, (wCam, hCam))
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)

        # Only process drawing if hand landmarks are detected
        if len(lmlist) != 0:
            x1, y1 = lmlist[8][1:3]
            x2, y2 = lmlist[12][1:3]
            fingers = []
        if lmlist:
            fingers = detector.fingerup()

            # Selection Mode: both index and middle finger up
            if fingers[1] == 1 and fingers[2] == 1:
                xp, yp = 0, 0
                if y1 < 125 and len(overlayList) >= 2:
                    if 0 < x1 < 271:
                        drawColor = (0, 0, 255)
                        header = cv2.resize(overlayList[0], (1280, 125))
                    elif 850 < x1 < 1280 and len(overlayList) > 1:
                        drawColor = (0, 0, 0)
                        header = cv2.resize(overlayList[1], (1280, 125))
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

                # Rightbar actions
                if x1 > 1050:
                    if 125 < y1 < 250:
                        canvas = np.zeros((720, 1280, 3), np.uint8)  # Clear canvas
                    if 260 < y1 < 385:
                        pass
                    if 385 < y1 < 510:
                        mode = "Drawing Mode"
                    if 510 < y1 < 635:
                        mode = "Prediction Mode"

            # Drawing Mode: only index finger up
            if len(fingers) >= 3 and fingers[1] and not fingers[2] and mode == "Drawing Mode":
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                xp, yp = x1, y1

                if drawColor == (0, 0, 0):
                    cv2.circle(img, (x1, y1), 30, drawColor, cv2.FILLED)
                    cv2.line(canvas, (xp, yp), (x1, y1), drawColor, 75)
                else:
                    cv2.line(canvas, (xp, yp), (x1, y1), drawColor, 15)
                xp, yp = x1, y1


            if  mode == "Prediction Mode":
                if LeftBar is not None:
                    img[125:425, 0:226] = LeftBar
                if len(fingers) >= 3 and fingers[1] and not fingers[2]:
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    xp, yp = x1, y1

                    if drawColor == (0, 0, 0):
                        cv2.circle(img, (x1, y1), 30, drawColor, cv2.FILLED)
                        cv2.line(canvas, (xp, yp), (x1, y1), drawColor, 75)
                    else:
                        cv2.line(canvas, (xp, yp), (x1, y1), drawColor, 15)
                    xp, yp = x1, y1

                # Leftbar actions
                if x1 < 300:
                    if 150 < y1 < 300:
                        submode = "Letter Prediction"
                        cv2.imwrite("Output/Letter.png", canvas)
                        predicted_letter, confidence = predict_from_image("Output/Letter.png")
                        cv2.putText(img, f'Predicted Letter: {predicted_letter}', (50, 500), cv2.FONT_HERSHEY_TRIPLEX,
                                    1, (255, 0, 255), 2)
                        prediction_time = time.time()
                        reset_canvas = True

                    if 315 < y1 < 405:
                        submode = "Word Prediction"
                        cv2.imwrite("Output/Word.png", canvas)
                        predictor = PredictWord("Output/Word.png")
                        result = predictor.predict()
                        print("Detected word:", result)
                        PredictWord.save_and_speak_word(result, output_dir='output', filename='output.txt')
                        canvas = np.zeros((720, 1280, 3), np.uint8)
            #
            #     # Place this outside the x1 < 300 block, so it runs every frame
            # if reset_canvas and prediction_time is not None:
            #     if time.time() - prediction_time > 5:
            #         canvas = np.zeros((720, 1280, 3), np.uint8)
            #         reset_canvas = False
            #         prediction_time = None

        # Combine canvas and camera image using bitwise operations
        imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, canvas)

        # Calculate FPS (frames per second)
        pTime = time.time()
        fps = 1 / (pTime - cTime) if cTime != 0 else 0
        cTime = pTime

        # Overlay header and RightBar only if they are loaded (robustness)
        if header is not None:
            img[0:125, 0:1280] = header
        if RightBar is not None:
            img[125:720, 1050:1280] = RightBar




        cv2.putText(img, f"Mode : {mode}", (1065, 645), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(img, f'FPS: {int(fps)}', (1095, 695), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255), 1)
        cv2.imshow("Canvas", canvas)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
    cam.release()
    cv2.destroyAllWindows()