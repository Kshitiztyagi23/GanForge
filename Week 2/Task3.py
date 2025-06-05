from yolov5 import detect
import os
import cv2
import numpy as np

def detect_flag(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (300, 200))  # Normalize size
    hue, _, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    # Masking to get only red part (red -> 160-179 in OpenCV) I am ignoring 0-10 as I get better results that way
    red_mask = ((hue > 160)) & (val > 50)

    # Split into top and bottom halves
    top_half = red_mask[:100, :]
    bottom_half = red_mask[100:, :]

    # Count red pixels in each half
    top_red_count = np.sum(top_half)
    bottom_red_count = np.sum(bottom_half)

    # Classification
    if top_red_count > bottom_red_count * 1.5:
        return "Indonesia"
    elif bottom_red_count > top_red_count * 1.5:
        return "Poland"
    else:
        return "Uncertain - Not clearly a flag of Indonesia or Poland"

for i in range(2):
    filename = os.path.join("task3_input", f"poland{i+1}.jpeg")
    print(filename, detect_flag(filename))
for i in range(2):
    filename = os.path.join("task3_input", f"indonesia{i+1}.jpeg")
    print(filename, detect_flag(filename))


