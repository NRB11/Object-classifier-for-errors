import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

def extract_cardboard_area(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None, None
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color range for brown (adjust if needed)
    lower_brown = np.array([10, 50, 20])
    upper_brown = np.array([30, 255, 200])
    
    # Create a mask to detect brown regions
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_area = 0
    
    for contour in contours:
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Filter for reasonable sizes
        area = cv2.contourArea(contour)
        if area > 500:  # Ignore small areas
            total_area += area
            
            # Draw detected areas on the image
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
    
    return image, total_area

def train_model():
    image, total_area = extract_cardboard_area("Images/test.png")
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img
        result_label.config(text=f"Total Surface Area: {total_area:.2f} pixels^2")

def classify_image():
    result_label.config(text="Coming soon")

# Create GUI
root = tk.Tk()
root.title("Cardboard Surface Area Detector")

train_button = Button(root, text="Training", command=train_model)
train_button.pack()

classify_button = Button(root, text="Classify Image", command=classify_image)
classify_button.pack()

img_label = Label(root)
img_label.pack()

result_label = Label(root, text="")
result_label.pack()

root.mainloop()
