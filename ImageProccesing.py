import cv2
import os
import time

# Set the folder path and video filename
folder_path = "Videos"
video_filename = "2025_02_09_16_13_21_502_8215_0E52ED78-78C6-4BF4-ADD2-08265FD22906.mp4"
video_path = os.path.join(folder_path, video_filename)

# Toggle this to True when running tests
TEST_MODE = True

# Function to limit FPS during testing
def limit_fps(target_fps):
    if TEST_MODE:
        time.sleep(1 / target_fps)

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit when video ends

    cv2.imshow("Video Player", frame)

    # Limit FPS to 10 when testing
    limit_fps(10)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
