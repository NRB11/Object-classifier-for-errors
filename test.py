import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "Images/test.png"  # Update with your actual image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define an adjusted brown color range (tuned for better detection)
lower_brown = np.array([5, 40, 40])   
upper_brown = np.array([40, 255, 255])

# Create a mask for brown regions
mask = cv2.inRange(hsv, lower_brown, upper_brown)

# Apply morphological operations to remove noise
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Apply Canny edge detection on the masked image
edges = cv2.Canny(mask, 50, 150)

# Find contours from the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
approx_quad = None
if largest_contour is not None:
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    if len(approx) == 4:  # Ensure it's a quadrilateral
        approx_quad = approx

# Draw the detected quadrilateral
quad_filtered_image = image_rgb.copy()
if approx_quad is not None:
    cv2.drawContours(quad_filtered_image, [approx_quad], -1, (0, 0, 255), 3)  # Red outline

# Display results
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Brown Color Mask")
axes[1].axis("off")

axes[2].imshow(edges, cmap="gray")
axes[2].set_title("Edge Detection")
axes[2].axis("off")

axes[3].imshow(quad_filtered_image)
axes[3].set_title("Final Quadrilateral")
axes[3].axis("off")

plt.show()
