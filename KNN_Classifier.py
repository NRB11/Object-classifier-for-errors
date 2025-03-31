import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.neighbors import KNeighborsClassifier
import pickle
from PIL import Image, ImageTk

# Function to extract the cardboard surface area from an image
def extract_cardboard_area(image_path):
    # Read the image
    img = cv2.imread(image_path)
    # Convert to HSV
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
        if area > 500:  # Ignore small areas, adjust this threshold as needed
            total_area += area

    return img, total_area

# Function to train the classifier with images in a folder
def train_classifier(training_folder_path):
    surface_areas = []
    labels = []
    for filename in os.listdir(training_folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(training_folder_path, filename)
            image, surface_area = extract_cardboard_area(image_path)

            # Display the image and surface area for manual classification
            cv2.imshow("Image", image)
            cv2.putText(image, f"Surface Area: {surface_area}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Image with Surface Area", image)
            cv2.waitKey(0)

            # Manually classify the image (Slip Sheet or Not Slip Sheet)
            label = input(f"Classify the image '{filename}' as 'Slip Sheet' (1) or 'Not Slip Sheet' (0): ")
            labels.append(int(label))
            surface_areas.append(surface_area)

    # Create and train a k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(np.array(surface_areas).reshape(-1, 1), labels)

    # Save the classifier to a file
    with open('knn_classifier.pkl', 'wb') as f:
        pickle.dump(knn, f)
    
    messagebox.showinfo("Training Complete", "Classifier training is complete!")

# Function to classify a new image using the k-NN classifier
def classify_image(classify_folder_path):
    # Load the trained k-NN classifier
    with open('knn_classifier.pkl', 'rb') as f:
        knn = pickle.load(f)

    for filename in os.listdir(classify_folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(classify_folder_path, filename)
            image, surface_area = extract_cardboard_area(image_path)

            # Predict the class using k-NN classifier
            prediction = knn.predict([[surface_area]])

            # Display the result
            result = "Slip Sheet" if prediction[0] == 1 else "Not Slip Sheet"
            cv2.putText(image, f"Prediction: {result}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Classified Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Function for the "Training" button
def on_train_button_click():
    # Ask user to select the folder for training images
    training_folder_path = filedialog.askdirectory(title="Select Folder for Manual Classification")
    if training_folder_path:
        train_classifier(training_folder_path)

# Function for the "Classify Image" button
def on_classify_button_click():
    # Ask user to select the folder for images to classify
    classify_folder_path = filedialog.askdirectory(title="Select Folder for Image Classification")
    if classify_folder_path:
        classify_image(classify_folder_path)

# Create the Tkinter window
root = tk.Tk()
root.title("Cardboard Surface Area Classifier")

# Create and pack the "Training" button
train_button = tk.Button(root, text="Training", command=on_train_button_click)
train_button.pack(pady=20)

# Create and pack the "Classify Image" button
classify_button = tk.Button(root, text="Classify Image", command=on_classify_button_click)
classify_button.pack(pady=20)

# Run the Tkinter main loop
root.mainloop()
