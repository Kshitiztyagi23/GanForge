import cv2
import numpy as np

def detect_flag(image_path):
    # Loading image
    img = cv2.imread(image_path)
    if img is None:
        print("Could not open image file.")
        return "Error loading image."

    # Resizing
    img = cv2.resize(img, (300, 200))

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Split into top and bottom halves
    top = hsv[:100, :]
    bottom = hsv[100:, :]

    # Color ranges (HSV)
    red1 = cv2.inRange(top, (0, 70, 50), (10, 255, 255))
    red2 = cv2.inRange(top, (160, 70, 50), (180, 255, 255))
    white = cv2.inRange(top, (0, 0, 200), (180, 30, 255))
    top_red = cv2.countNonZero(red1) + cv2.countNonZero(red2)
    top_white = cv2.countNonZero(white)

    red1_b = cv2.inRange(bottom, (0, 70, 50), (10, 255, 255))
    red2_b = cv2.inRange(bottom, (160, 70, 50), (180, 255, 255))
    white_b = cv2.inRange(bottom, (0, 0, 200), (180, 30, 255))
    bottom_red = cv2.countNonZero(red1_b) + cv2.countNonZero(red2_b)
    bottom_white = cv2.countNonZero(white_b)

    # Figure out which color is dominant in each half
    top_color = 'red' if top_red > top_white else 'white'
    bottom_color = 'red' if bottom_red > bottom_white else 'white'

    # Debug prints 
    print(f"Top: {top_color}, Bottom: {bottom_color}")

    if top_color == 'red' and bottom_color == 'white':
        return "this is Indonesia's flag "
    elif top_color == 'white' and bottom_color == 'red':
        return "this is Poland's flag"
    else:
        # TODO: Add more flags later
        return "Sorry, I couldn't recognize this flag."

# Example usage
result = detect_flag("flag1")
print(result)