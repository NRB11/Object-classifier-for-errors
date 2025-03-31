import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
from sklearn.neighbors import KNeighborsClassifier
import pickle
import pandas as pd
from PIL import Image, ImageTk

# Global variables
training_folder_path = ""
csv_filename = "training_data.csv"
training_data = pd.DataFrame(columns=["Surface Area", "Label"])
knn = None  # Placeholder for classifier

# Function to extract cardboard surface area
def extract_cardboard_area(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range for cardboard color in HSV
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([20, 255, 255])

    # Mask and extract
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    result = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours and compute area
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 500)
    
    return img, total_area

# Function to save training data
def save_training_data(surface_area, label):
    global training_data
    new_data = pd.DataFrame({"Surface Area": [surface_area], "Label": [label]})
    training_data = pd.concat([training_data, new_data], ignore_index=True)
    training_data.to_csv(csv_filename, index=False)

# Function to load training data
def load_training_data():
    global training_data
    if os.path.exists(csv_filename):
        training_data = pd.read_csv(csv_filename)
        print("Loaded training data.")

# Function to train k-NN classifier
def train_knn():
    global knn
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

# Function to start manual classification in a new window
def start_training():
    global training_folder_path

    training_folder_path = filedialog.askdirectory(title="Select Training Folder")
    if not training_folder_path:
        return

    image_files = [f for f in os.listdir(training_folder_path) if f.endswith((".jpg", ".png"))]
    if not image_files:
        messagebox.showerror("Error", "No images found in the selected folder!")
        return

    # Open new training window
    training_window = Toplevel(root)
    training_window.title("Manual Classification")

    image_label = tk.Label(training_window)
    image_label.pack()

    text_label = tk.Label(training_window, text="Surface Area: -- px²", font=("Arial", 14))
    text_label.pack()

    # Function to classify an image and move to the next
    def classify_image(surface_area, label, idx):
        save_training_data(surface_area, label)
        if idx + 1 < len(image_files):
            display_image(idx + 1)
        else:
            messagebox.showinfo("Done", "All images classified! Training model now.")
            training_window.destroy()
            train_knn()

    # Function to display the next image
    def display_image(idx):
        image_path = os.path.join(training_folder_path, image_files[idx])
        img, surface_area = extract_cardboard_area(image_path)

        # Convert to Tkinter-compatible format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        # Update GUI
        image_label.config(image=img_tk)
        image_label.image = img_tk
        text_label.config(text=f"Surface Area: {surface_area} px²")

        # Update buttons
        classify_button_slip_sheet.config(command=lambda: classify_image(surface_area, "Slip Sheet", idx))
        classify_button_not_slip_sheet.config(command=lambda: classify_image(surface_area, "Not Slip Sheet", idx))

    # Classification buttons in new window
    classify_button_slip_sheet = tk.Button(training_window, text="Slip Sheet", bg="green", fg="white")
    classify_button_slip_sheet.pack(side="left", padx=10, pady=10)

    classify_button_not_slip_sheet = tk.Button(training_window, text="Not Slip Sheet", bg="red", fg="white")
    classify_button_not_slip_sheet.pack(side="right", padx=10, pady=10)

    # Start with first image
    display_image(0)

# Function to classify a new image
def classify_new_image():
    global knn

    classify_folder_path = filedialog.askdirectory(title="Select Classification Folder")
    if not classify_folder_path:
        return

    # Load training data
    load_training_data()
    if training_data.empty:
        messagebox.showerror("Error", "No training data found! Run training first.")
        return

    # Train k-NN if not trained
    if knn is None:
        train_knn()

    # Load classifier if exists
    if os.path.exists('knn_classifier.pkl'):
        with open('knn_classifier.pkl', 'rb') as f:
            knn = pickle.load(f)
    else:
        messagebox.showerror("Error", "No trained model found! Train first.")
        return

    # Process images
    for filename in os.listdir(classify_folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(classify_folder_path, filename)
            img, surface_area = extract_cardboard_area(image_path)

            # Predict using k-NN
            prediction = knn.predict([[surface_area]])
            result = "Slip Sheet" if prediction[0] == "Slip Sheet" else "Not Slip Sheet"

            # Display result
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)

            image_label.config(image=img_tk)
            image_label.image = img_tk
            text_label.config(text=f"Prediction: {result}")

# Load training data at startup
load_training_data()

# Create Tkinter GUI
root = tk.Tk()
root.title("Cardboard Surface Area Classifier")

# Image display
image_label = tk.Label(root)
image_label.pack()

# Surface area display
text_label = tk.Label(root, text="Surface Area: -- px²", font=("Arial", 14))
text_label.pack()

# Training and classification buttons
train_button = tk.Button(root, text="Training", command=start_training)
train_button.pack(pady=20)

classify_button = tk.Button(root, text="Classify Image", command=classify_new_image)
classify_button.pack(pady=20)

# Run Tkinter loop
root.mainloop()
