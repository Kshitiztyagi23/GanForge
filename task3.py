import cv2
import numpy as np

def segment_flag_by_color(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (300, 200))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    combined_mask = cv2.bitwise_or(white_mask, red_mask)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found")
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    cropped = image[y:y+h, x:x+w]
    return cropped


def image_classifier(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_resized = cv2.resize(image_rgb, (200, 100))

    top_half = image_resized[:50, :, :]
    bottom_half = image_resized[50:, :, :]

    def is_red_region(region):
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (160, 100, 100), (179, 255, 255))
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_ratio = np.count_nonzero(red_mask) / red_mask.size
        return red_ratio > 0.5

    def is_white_region(region):
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        white_ratio = np.count_nonzero(white_mask) / white_mask.size
        return white_ratio > 0.5

    if is_red_region(top_half) and is_white_region(bottom_half):
        print("Indonesia")
    elif is_red_region(bottom_half) and is_white_region(top_half):
        print("Poland")
    else:
        print("Not sure")

img = 'Screenshot 2025-06-08 175424.png'
img2 = segment_flag_by_color(img)
image_classifier(img2)