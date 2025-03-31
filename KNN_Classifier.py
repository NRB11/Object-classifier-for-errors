import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.neighbors import KNeighborsClassifier
import pickle
import pandas as pd
from PIL import Image, ImageTk

# Global variables
current_image_index = 0
image_files = []
training_folder_path = ""
csv_filename = "training_data.csv"
training_data = pd.DataFrame(columns=["Surface Area", "Label"])

# Function to extract the cardboard surface area from an image
def extract_cardboard_area(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range for cardboard (brown) color in HSV
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([20, 255, 255])

    # Mask the image to isolate the cardboard color
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    result = cv2.bitwise_and(img, img, mask=mask)

    # Convert to grayscale and apply thresholding
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the surface area of each contour
    total_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Ignore small areas, adjust as needed
            total_area += area

    return img, total_area

# Function to save training data to CSV
def save_training_data(surface_area, label):
    global training_data
    new_data = pd.DataFrame({"Surface Area": [surface_area], "Label": [label]})
    
    # Append new data and save
    training_data = pd.concat([training_data, new_data], ignore_index=True)
    training_data.to_csv(csv_filename, index=False)

# Function to load training data from CSV
def load_training_data():
    global training_data
    if os.path.exists(csv_filename):
        training_data = pd.read_csv(csv_filename)
        print("Loaded existing training data.")

# Function to train the k-NN classifier
def train_knn():
    if training_data.empty:
        messagebox.showerror("Error", "No training data available!")
        return

    X = training_data["Surface Area"].values.reshape(-1, 1)
    y = training_data["Label"].values

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    # Save classifier
    with open('knn_classifier.pkl', 'wb') as f:
        pickle.dump(knn, f)

    messagebox.showinfo("Training Complete", "Classifier training is complete!")

# Function to display the next image for manual classification
def display_next_image():
    global current_image_index, image_files, training_folder_path

    if current_image_index >= len(image_files):
        messagebox.showinfo("Done", "All images have been classified!")
        train_knn()  # Train k-NN after manual classification
        return

    image_path = os.path.join(training_folder_path, image_files[current_image_index])
    img, surface_area = extract_cardboard_area(image_path)

    # Convert to Tkinter-compatible format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    # Update the GUI
    image_label.config(image=img_tk)
    image_label.image = img_tk
    text_label.config(text=f"Surface Area: {surface_area} px²")

    # Store surface area
    classify_button_slip_sheet.config(command=lambda: classify_image(surface_area, 1))
    classify_button_not_slip_sheet.config(command=lambda: classify_image(surface_area, 0))

# Function to classify an image manually
def classify_image(surface_area, label):
    global current_image_index
    save_training_data(surface_area, label)
    current_image_index += 1
    display_next_image()

# Function for the "Training" button
def on_train_button_click():
    global training_folder_path, image_files, current_image_index

    training_folder_path = filedialog.askdirectory(title="Select Folder for Manual Classification")
    if not training_folder_path:
        return

    # Load all image files
    image_files = [f for f in os.listdir(training_folder_path) if f.endswith((".jpg", ".png"))]
    if not image_files:
        messagebox.showerror("Error", "No images found in the selected folder!")
        return

    current_image_index = 0
    display_next_image()

# Function to classify an image using k-NN
def on_classify_button_click():
    classify_folder_path = filedialog.askdirectory(title="Select Folder for Image Classification")
    if not classify_folder_path:
        return

    if not os.path.exists('knn_classifier.pkl'):
        messagebox.showerror("Error", "Please train the model first!")
        return

    # Load classifier
    with open('knn_classifier.pkl', 'rb') as f:
        knn = pickle.load(f)

    for filename in os.listdir(classify_folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(classify_folder_path, filename)
            img, surface_area = extract_cardboard_area(image_path)

            # Predict
            prediction = knn.predict([[surface_area]])
            result = "Slip Sheet" if prediction[0] == 1 else "Not Slip Sheet"

            # Display result
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)

            image_label.config(image=img_tk)
            image_label.image = img_tk
            text_label.config(text=f"Prediction: {result}")

# Load existing training data on startup
load_training_data()

# Create the Tkinter window
root = tk.Tk()
root.title("Cardboard Surface Area Classifier")

# Image display label
image_label = tk.Label(root)
image_label.pack()

# Surface area display text
text_label = tk.Label(root, text="Surface Area: -- px²", font=("Arial", 14))
text_label.pack()

# Classification buttons
classify_button_slip_sheet = tk.Button(root, text="Slip Sheet", command=None, bg="green", fg="white")
classify_button_slip_sheet.pack(side="left", padx=10, pady=10)

classify_button_not_slip_sheet = tk.Button(root, text="Not Slip Sheet", command=None, bg="red", fg="white")
classify_button_not_slip_sheet.pack(side="right", padx=10, pady=10)

# Training and classification buttons
train_button = tk.Button(root, text="Training", command=on_train_button_click)
train_button.pack(pady=20)

classify_button = tk.Button(root, text="Classify Image", command=on_classify_button_click)
classify_button.pack(pady=20)

# Run the Tkinter main loop
root.mainloop()
