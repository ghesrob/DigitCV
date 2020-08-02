import cv2
import numpy as np
from PIL import ImageGrab
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk

from gui_settings import *

class DigitGui:

    def __init__(self, root, model):
        # Settings
        self.model = model
        self.root = root
        self.root.title("Digits-CV")
        self.root.iconbitmap("images/main.ico")
        self.root.geometry("600x700")
        self.root.resizable(0,0)
        # GUI initialization
        self.create_widgets()
        self.ready = True

    def create_widgets(self):
        """Define the app layout."""
        # Box around drawing surface
        self.box = tk.Canvas(self.root, width=504, height=537, bg=color1, highlightthickness=0)
        self.box.place(x=48, y=48)
        self.box_title = tk.Label(self.box, text='Draw some digits to recognize',font=myfont, bg=color1, fg=white)
        self.box_title.place(x=2, y=5)
        # Drawing surface
        self.cv = tk.Canvas(self.box, width=500, height=500, bg=white, highlightthickness=0)
        self.cv.place(x=2, y=35)
        self.cv.bind("<Button-1>", self.activate_painting)
        self.cv.bind("<B1-Motion>", self.paint_lines)
        # Predict button
        self.btn_predict2 = tk.Label(self.root, text='Predict',font=myfont, height=2, width=13, bg=color1, fg=white)
        self.btn_predict2.place(relx=0.3, y=630, anchor=tk.CENTER)
        self.btn_predict2.bind("<Enter>", lambda event: self.btn_predict2.configure(fg=color2))
        self.btn_predict2.bind("<Leave>", lambda event: self.btn_predict2.configure(fg=white))
        self.btn_predict2.bind("<Button-1>", self.predict)
        # Reset button
        self.btn_reset2 = tk.Label(self.root, text='Reset',font=myfont, height=2, width=13, bg=color1, fg=white)
        self.btn_reset2.place(relx=0.7, y=630, anchor=tk.CENTER)
        self.btn_reset2.bind("<Enter>", lambda event: self.btn_reset2.configure(fg=color2))
        self.btn_reset2.bind("<Leave>", lambda event: self.btn_reset2.configure(fg=white))
        self.btn_reset2.bind("<Button-1>", self.reset)


    def activate_painting(self, event): 
        """Helper function for painting.
        Triggered when mouse button is pressed over the drawing surface.
        """
        if self.ready:
            self.x1, self.y1 = event.x, event.y

    def paint_lines(self, event):
        """Paint lines on the drawing surface.
        Triggered when the mouse is moved over the drawing surface, with a button being held down.
        """
        if self.ready:
            x2, y2 = event.x, event.y
            self.cv.create_line(
                (self.x1, self.y1, x2, y2),
                width=20, 
                fill="#16a085",
                capstyle=tk.ROUND,
                smooth=0,
                splinesteps=12
            )
            self.x1, self.y1 = x2, y2

    def reset(self, event):
        """Reset app for a new drawing.
        Triggered when pressing reset button.
        """
        self.cv.delete("all")
        self.ready = True

    def save_drawing(self, filename):
        """Cropping of canvas's paiting and saving as a png file"""
        x_start = root.winfo_rootx() + self.cv.winfo_x()
        y_start = root.winfo_rooty() + self.cv.winfo_y()
        x_end = x_start + self.cv.winfo_width()
        y_end = y_start + self.cv.winfo_height()
        drawing = ImageGrab.grab().crop((x_start, y_start, x_end, y_end))
        drawing.save(filename)

    def predict(self, event):
        if not self.ready:
            return
            
        filename = "tmp/image.png"
        self.save_drawing(filename)
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        countours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        for  contour in countours:

            x, y, w, h = cv2.boundingRect(contour)
            top = bottom = int(0.05 * binary.shape[0])
            left = right = int(0.05 * binary.shape[1])
            roi = binary[y-top:y+h+bottom, x-left:x+w+right]

            # Image preprocessing and model prediction
            img = cv2.resize(roi, (28,28), interpolation=cv2.INTER_AREA)
            img = img.reshape(1, 28, 28, 1)
            img = img / 255.0
            pred = self.model.predict(img)[0]
            final_pred = np.argmax(pred)
            
            # Write results on image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(image, (x,y), (x+w, y+h), color_bbox, 1)
            cv2.putText(image, f"Digit: {final_pred}", (x, y-22), fontFace=font, 
                        fontScale=0.5, color=color_bbox, thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(image, f"Confidence: {int(max(pred)*100)}%", (x, y-5), fontFace=font, 
                        fontScale=0.5, color=color_bbox, thickness=1, lineType=cv2.LINE_AA)

        # Print result image on screen
        cv2.imwrite(filename, image)
        self.result = tk.PhotoImage(file = filename)
        self.cv.create_image(-48, -48, image=self.result, anchor=tk.NW)
        self.ready = False



if __name__ == "__main__":
    root = tk.Tk()
    model = load_model('models/model_gray_scale.h5')
    window = DigitGui(root, model)
    root.mainloop()