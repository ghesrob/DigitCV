import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import ImageTk, Image, ImageDraw

from tkinter import Tk, Canvas, Button, Text, INSERT, END, Label, ttk, W, font, StringVar

#col_bg = '#c7d5e0'
#col_title = '#1b2838'
#col_body = '#2a475e'
col_bg = '#ECF0F1'
col_title = '#34495E'
col_body = '#1ABC9C'



class DigitGui:
    
    def __init__(self, root):
        self.helv36 = ('Helvetica', 36, 'bold')      
        self.malg36 = ('Malgun Gothic', 36)
        self.model = load_model('model/model_15epochs.h5')
        self.root = root
        self.root.title("Reconnaissance de chiffres manuscrits")
        self.root.geometry("1000x700")
        self.root.configure(bg=col_bg)
        self.create_widgets()

    def create_widgets(self):
        # Titre de la zone de dessin
        self.box_top = Canvas(self.root, width=500, height=20, bg=col_title, highlightthickness=0).place(x=20, y=80)
        Label(self.box_top, text='Draw a digit', bg=col_title, fg='#FFFFFF').place(x=25, y=80)
        # Zone de dessin
        self.cv = Canvas(self.root, width=500, height=500, bg=col_body, borderwidth=0, relief="flat", highlightthickness=0)
        self.cv.place(x=20, y=100)
        self.cv.bind("<B1-Motion>", self.paint)
        # Image
        self.image1 = Image.new("RGB", (500, 500), (255,255,255))
        self.draw = ImageDraw.Draw(self.image1)

        # Boutons
        self.btn_predict = ttk.Button(text="Predict", command=self.predict)
        self.btn_predict.place(x=120, y=630)
        self.btn_reset = ttk.Button(text=" Reset ", command=self.reset)
        self.btn_reset.place(x=360, y=630)

        # Titre résultats prédiction
        self.res_pred_top = Canvas(self.root, width=100, height=20, bg=col_title, highlightthickness=0).place(x=600, y=100)
        Label(self.res_pred_top, text='Prediction:', bg=col_title, fg='#FFFFFF').place(x=605, y=100)
        # Canva résulats prédiction
        self.cv_pred = Canvas(self.root, width=100, height=90, bg=col_body, highlightthickness=0).place(x=600, y=120)
        self.text_pred = StringVar()
        Label(self.cv_pred, textvariable = self.text_pred, bg=col_body, fg='#FFFFFF', font=self.malg36).place(x=620, y=130)

        # Titre résultats confidence
        self.res_conf_top = Canvas(self.root, width=100, height=20, bg=col_title, highlightthickness=0).place(x=600, y=300)
        Label(self.res_conf_top, text='Confidence:', bg=col_title, fg='#FFFFFF').place(x=605, y=300)
        # Canva résultat confidence
        self.cv_conf = Canvas(self.root, width=100, height=90, bg=col_body, highlightthickness=0).place(x=600, y=320)
        self.text_conf = StringVar()
        Label(self.cv_conf, textvariable=self.text_conf, bg=col_body, fg='#FFFFFF').place(x=605, y=370)


    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.cv.create_oval(x1, y1, x2, y2, fill="black",width=10)
        self.draw.line([x1, y1, x2, y2],fill="black",width=10)
  

    def predict(self):
        self.image1.save("tmp/image.png")
        img = cv2.imread('tmp/image.png', 0)
        img = cv2.bitwise_not(img)
        img = cv2.resize(img, (28,28))
        img = img.reshape(1,28,28,1)
        img = img / 255.0

        pred=self.model.predict(img)[0]
        pred_digit = np.argmax(pred)
        pred_conf = round(pred[pred_digit]*100, 1)

        self.text_pred.set(str(pred_digit))
        self.text_conf.set(str(pred_conf))


        
    def reset(self):
        self.cv.delete('all')
        self.draw.rectangle((0, 0, 500, 500), fill=(255, 255, 255, 0))
        self.text_pred.set("")
        self.text_conf.set("")



if __name__ == "__main__":
    root = Tk()
    window = DigitGui(root)
    root.mainloop()