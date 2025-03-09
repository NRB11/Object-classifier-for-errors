import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "Images/test.png"  # Update with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define brown color range in HSV
lower_brown = np.array([10, 50, 50])
upper_brown = np.array([30, 255, 255])

# Create a mask for brown regions
mask = cv2.inRange(hsv, lower_brown, upper_brown)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Function to find the largest contour
def find_largest_contour(contours):
    max_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    return max_contour

# Get the largest contour
largest_contour = find_largest_contour(contours)

# Approximate the contour to a quadrilateral
if largest_contour is not None:
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx_quad = cv2.approxPolyDP(largest_contour, epsilon, True)
else:
    approx_quad = None

# Draw the detected quadrilateral
quad_filtered_image = image_rgb.copy()
if approx_quad is not None and len(approx_quad) == 4:
    cv2.drawContours(quad_filtered_image, [approx_quad], -1, (0, 0, 255), 3)  # Red outline

# Display results
plt.figure(figsize=(7, 5))
plt.imshow(quad_filtered_image)
plt.title("Refined Quadrilateral (Cardboard Detection)")
plt.axis("off")
plt.show()
