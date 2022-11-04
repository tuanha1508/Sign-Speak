import cv2
import pickle
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np

white = "#ffffff"
lightBlue2 = "#adc5ed"
font = "Constantia"
fontButtons = (font, 12)
maxWidth = 800
maxHeight = 700


def image_processed(hand_img):

    # Image processing
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # 2. Flip the img in Y-axis
    img_flip = cv2.flip(img_rgb, 1)

    # accessing MediaPipe solutions
    mp_hands = mp.solutions.hands

    # Initialize Hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.7
    )

    # Results
    output = hands.process(img_flip)

    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        # print(data)
        data = str(data)

        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)

        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return (clean)
    except:
        return (np.zeros([1, 63], dtype=int)[0])


with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)


# Graphics window
mainWindow = tk.Tk()
mainWindow.configure(bg=lightBlue2)
mainWindow.geometry('%dx%d+%d+%d' % (maxWidth, maxHeight, 0, 0))
mainWindow.resizable(0, 0)
# mainWindow.overrideredirect(1)

mainFrame = Frame(mainWindow)
mainFrame.place(x=20, y=20)

# Capture video frames
lmain = Label(mainFrame)
lmain.grid(row=0, column=0)

cap = cv2.VideoCapture(0)

currentSign = ''

currentText = Label(mainWindow, text=currentSign, font=("Courier", 44))
currentText.place(x=20, y=450)

seg = ""
segText = Label(mainWindow, text=seg, font=("Courier", 44))
segText.place(x=20, y=530)


def show_frame():
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")

    frame = cv2.flip(frame, 1)
    data = image_processed(frame)
    data = np.array(data)
    y_pred = svm.predict(data.reshape(-1, 63))

    sign = str(y_pred[0])
    global currentText
    currentText.config(text=sign)

    global seg
    if len(sign) == 1 and (len(seg) == 0 or sign != seg[-1]):
        seg += sign
    segText.config(text=seg)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    img = Image.fromarray(cv2image).resize((760, 400))
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


closeButton = Button(mainWindow, text="CLOSE",
                     font=fontButtons, bg=white, width=20, height=1)
closeButton.configure(command=lambda: mainWindow.destroy())
closeButton.place(x=270, y=650)


show_frame()  # Display
mainWindow.mainloop()  # Starts GUI
