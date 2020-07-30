import os

import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import ImageTk, Image, ImageDraw

from tkinter import Tk, Canvas, Button, Text, INSERT, END, Label, ttk, W

class DigitGui:

    
    def __init__(self, root):
        self.model = load_model('model/model_15epochs.h5')
        self.root = root
        root.title("Reconnaissance de chiffres manuscrits")
        self.create_widgets()

    def create_widgets(self):

        self.label = ttk.Label(self.root,
            text = "Dessinez un chiffre", 
            justify = "left"
        )
        self.label.grid(row=0, column=0,  padx=10, sticky=W)
        # Zone de dessin
        self.cv = Canvas(self.root, width=500, height=500, bg="white", borderwidth=5, relief="groove")
        self.cv.grid(row=1, column=0, padx=10, pady=2, columnspan=2)
        self.cv.bind("<B1-Motion>", self.paint)
        # Image
        self.image1 = Image.new("RGB", (500, 500), (255,255,255))
        self.draw = ImageDraw.Draw(self.image1)

        # Boutons
        self.btn_predict = ttk.Button(text="Predict",command=self.predict)
        self.btn_predict.grid(row=2, column=0, pady=1, padx=1)
        self.btn_reset = ttk.Button(text=" Reset ",command=self.reset)
        self.btn_reset.grid(row=2, column=1, pady=1, padx=1)

        # Zone affichage résultats
        self.txt = Text(root,
            bd=3,
            exportselection=0,
            bg='WHITE',
            font='Helvetica', 
            padx=10,
            pady=10,
            height=5,
            width=20
            )
        self.txt.grid(row=3, column=0, columnspan = 2, pady=10)

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.cv.create_oval(x1, y1, x2, y2, fill="black",width=10)
        self.draw.line([x1, y1, x2, y2],fill="black",width=10)
  
    def testing(self):
        img = cv2.imread('tmp/image.png', 0)
        img = cv2.bitwise_not(img)
        img = cv2.resize(img, (28,28))
        img = img.reshape(1,28,28,1)
        img = img / 255.0

        pred=self.model.predict(img)
        return pred

    def predict(self):
        classes = [0,1,2,3,4,5,6,7,8,9]
        filename = "tmp/image.png"
        self.image1.save(filename)
        pred = self.testing()
        print('argmax',np.argmax(pred[0]),'\n',
            pred[0][np.argmax(pred[0])],'\n',classes[np.argmax(pred[0])])
        self.txt.insert(
            INSERT,
            "{}\nAccuracy: {}%".format(classes[np.argmax(pred[0])],round(pred[0][np.argmax(pred[0])]*100,3))
        )
        
    def reset(self):
        self.cv.delete('all')
        self.draw.rectangle((0, 0, 500, 500), fill=(255, 255, 255, 0))
        self.txt.delete('1.0', END)


if __name__ == "__main__":
    root = Tk()
    window = DigitGui(root)
    root.mainloop()