import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

MODEL_PATH = "mnist_cnn_model.keras"
CANVAS_SIZE = 280   # drawing area in pixels
BRUSH_SIZE = 16     # pen thickness in pixels

model = tf.keras.models.load_model(MODEL_PATH)


class DigitRecognizerApp:
    def __init__(self, root):
        root.title("Handwritten Digit Recognizer")

        tk.Label(root, text="Draw a digit (0-9)", font=("Helvetica", 14, "bold")).pack(pady=5)

        # Black canvas, white pen (same colors as MNIST images)
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)

        # Hidden PIL image that copies everything drawn on the canvas.
        # We feed this image (not the screen) to the model.
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

        buttons = tk.Frame(root)
        buttons.pack(pady=8)
        tk.Button(buttons, text="Predict", width=12, command=self.predict).grid(row=0, column=0, padx=5)
        tk.Button(buttons, text="Clear", width=12, command=self.clear).grid(row=0, column=1, padx=5)

        self.result = tk.Label(root, text="Prediction: -", font=("Helvetica", 18, "bold"))
        self.result.pack()
        self.confidence = tk.Label(root, text="Confidence: -", font=("Helvetica", 12))
        self.confidence.pack(pady=(0, 10))

    # ---------- 1. Draw ----------
    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is None:                 # first point of a stroke
            self.last_x, self.last_y = x, y
        points = [self.last_x, self.last_y, x, y]
        self.canvas.create_line(points, width=BRUSH_SIZE, fill="white", capstyle=tk.ROUND)
        self.draw.line(points, fill=255, width=BRUSH_SIZE)
        self.last_x, self.last_y = x, y

    def end_stroke(self, event):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result.config(text="Prediction: -")
        self.confidence.config(text="Confidence: -")

    # ---------- 2. Preprocess ----------
    def preprocess(self):
        # Step 1: crop to the digit
        img = self.image.crop(self.image.getbbox())

        # Step 2: make it square and resize to 20x20
        size = max(img.size)
        square = Image.new("L", (size, size), 0)
        square.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
        small = square.resize((20, 20))

        # Step 3: center it in a 28x28 image (like MNIST)
        final = Image.new("L", (28, 28), 0)
        final.paste(small, (4, 4))

        # Step 4: scale pixels to 0-1 and add batch/channel dimensions
        return np.array(final).reshape(1, 28, 28, 1) / 255.0

    # ---------- 3. Predict ----------
    def predict(self):
        if self.image.getbbox() is None:        # nothing drawn
            return
        probs = model.predict(self.preprocess(), verbose=0)[0]
        digit = int(np.argmax(probs))
        self.result.config(text=f"Prediction: {digit}")
        self.confidence.config(text=f"Confidence: {probs[digit] * 100:.1f}%")


root = tk.Tk()
DigitRecognizerApp(root)
root.mainloop()
