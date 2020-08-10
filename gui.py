import tkinter as tk

import cv2
import numpy as np
from PIL import ImageGrab
from tensorflow.keras.models import load_model

from gui_settings import *


class DigitCV:

    def __init__(self, root, model):
        # Settings
        self.model = model
        self.root = root
        self.root.title("DigitCV")
        self.root.iconbitmap("images/app_icon.ico")
        self.root.geometry("600x800")
        self.root.resizable(0,0)
        self.root.configure(background=col_background)
        # GUI initialization
        self.create_widgets()
        self.ready = True

    def create_widgets(self):
        """Define the app layout."""
        # Header bar
        self.header = tk.Canvas(self.root, width=600, height=80, bg=col_primary, highlightthickness=0)
        self.header.place(x=0, y=0)
        self.header_image = tk.PhotoImage(file="images/logo70.png")
        self.header.create_image(50,40, image=self.header_image, anchor=tk.CENTER)
        self.header_title = tk.Label(self.header, text="DigitCV", font=font_title, bg=col_primary, fg=col_secondary)
        self.header_title.place(x=100, y=5, anchor=tk.NW)
        self.header_text = tk.Label(self.header, text="Handwritten digits recognition", font=font_text, bg=col_primary, fg=col_foreground)
        self.header_text.place(x=100, y=70, anchor=tk.SW)
        # Box around drawing surface
        self.box = tk.Canvas(self.root, width=504, height=537, bg=col_primary, highlightthickness=0)
        self.box.place(x=48, y=130)
        self.box_title = tk.Label(self.box, text='Draw some digits',font=font_text, bg=col_primary, fg=col_foreground)
        self.box_title.place(x=5, y=5)
        # Drawing surface
        self.cv = tk.Canvas(self.box, width=500, height=500, bg=col_foreground, highlightthickness=0)
        self.cv.place(x=2, y=35)
        self.cv.bind("<Button-1>", self.activate_painting)
        self.cv.bind("<B1-Motion>", self.paint_lines)
        # Predict button
        self.btn_predict2 = tk.Label(self.root, text="Predict",font=font_btn, height=2, width=15, bg=col_primary, fg=col_foreground)
        self.btn_predict2.place(relx=0.3, y=725, anchor=tk.CENTER)
        self.btn_predict2.bind("<Enter>", lambda event: self.btn_predict2.configure(fg=col_secondary))
        self.btn_predict2.bind("<Leave>", lambda event: self.btn_predict2.configure(fg=col_foreground))
        self.btn_predict2.bind("<Button-1>", self.predict)
        # Reset button
        self.btn_reset2 = tk.Label(self.root, text="Reset",font=font_btn, height=2, width=15, bg=col_primary, fg=col_foreground)
        self.btn_reset2.place(relx=0.7, y=725, anchor=tk.CENTER)
        self.btn_reset2.bind("<Enter>", lambda event: self.btn_reset2.configure(fg=col_secondary))
        self.btn_reset2.bind("<Leave>", lambda event: self.btn_reset2.configure(fg=col_foreground))
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
                width=15, 
                fill=col_drawing,
                capstyle=tk.ROUND,
                smooth=1
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
        x_start, y_start = self.cv.winfo_rootx(), self.cv.winfo_rooty() 
        x_end, y_end = x_start + self.cv.winfo_width(), y_start + self.cv.winfo_height()
        drawing = ImageGrab.grab().crop((x_start, y_start, x_end, y_end))
        drawing.save(filename)

    def predict(self, event):
        """Use saved model for digits recognition.
        Save the drawing as an image, preprocess it and submit it to the network for prediction.
        Tiggered when pressing predict button.
        """
        if not self.ready:
            return
        
        # Image loading
        filename = "tmp/image.png"
        self.save_drawing(filename)
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Loop over each drawn digit
        digits = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for  digit in digits:
            # Image cropping around current digit
            x, y, w, h = cv2.boundingRect(digit)
            x_padding = int(0.05 * binary.shape[1])
            y_padding = int(0.05 * binary.shape[0])
            img = binary[y-y_padding:y+h+y_padding, x-x_padding:x+w+x_padding]
            # Image reshaping and model prediction
            img = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)
            img = img.reshape(1, 28, 28, 1)
            img = img / 255.0
            pred = self.model.predict(img)[0]       
            # Write results on image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(image, (x,y), (x+w, y+h), col_cv2, 1)
            cv2.putText(image, f"Digit: {np.argmax(pred)}", (x, y-22), fontFace=font, 
                        fontScale=0.5, color=col_cv2, thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(image, f"Confidence: {int(max(pred)*100)}%", (x, y-5), fontFace=font, 
                        fontScale=0.5, color=col_cv2, thickness=1, lineType=cv2.LINE_AA)

        # Print result on screen
        cv2.imwrite(filename, image)
        self.result = tk.PhotoImage(file = filename)
        self.cv.create_image(0, 0, image=self.result, anchor=tk.NW)
        self.ready = False


if __name__ == "__main__":
    root = tk.Tk()
    model = load_model('models/model_gray_scale.h5')
    window = DigitCV(root, model)
    root.mainloop()