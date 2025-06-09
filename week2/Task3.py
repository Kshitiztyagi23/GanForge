import cv2
import numpy as np
from ultralytics import YOLO
import os


def crop_flag(image_path):
    model = YOLO('yolov5s.pt')  

    
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model(img)

    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
    
            areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
            idx = int(areas.argmax())
            x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy().astype(int)
            cropped_flag = img[y1:y2, x1:x2]

    
            cv2.imwrite("cropped_flag.jpg", cropped_flag)
            print("[INFO] Saved cropped flag as cropped_flag.jpg")

            return cropped_flag

    print("[INFO] No flag detected, using full image")
    cv2.imwrite("cropped_flag.jpg", img)
    print("[INFO] Saved full image as fallback cropped_flag.jpg")
    return img


def identify_flag(flag_image):
    hsv = cv2.cvtColor(flag_image, cv2.COLOR_BGR2HSV)

 
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])

    
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 50, 255])

    height, _ = hsv.shape[:2]
    stripe_height = max(5, height // 20)  # Thin stripe height, min 5px

    for y in range(0, height, stripe_height):
        stripe = hsv[y:y + stripe_height]

        red_mask1 = cv2.inRange(stripe, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(stripe, red_lower2, red_upper2)
        red_pixels = cv2.countNonZero(red_mask1 | red_mask2)

        white_mask = cv2.inRange(stripe, white_lower, white_upper)
        white_pixels = cv2.countNonZero(white_mask)

        if red_pixels > white_pixels and red_pixels > 50:
            return "Indonesian Flag 🇮🇩"
        elif white_pixels > red_pixels and white_pixels > 50:
            return "Polish Flag 🇵🇱"
    else:
        return "Flag type could not be determined"


    


def Execute():
    image_path = input("Enter the path to the flag image: ").strip()

    if not os.path.exists(image_path):
        print("Error: File not found.")
        return

    cropped_flag = crop_flag(image_path)
    result = identify_flag(cropped_flag)

    print(f"[RESULT] Detected flag: {result}")


if __name__ == "__main__":
    Execute()
