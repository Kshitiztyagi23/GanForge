import cv2
import numpy as np
from ultralytics import YOLO

def crop_flag(image_path):
   
  
    model = YOLO('yolov5su.pt')  
    
    # Load image
    img = cv2.imread(image_path)
    
    # Run detection
    results = model(img)
    
    # Get detections
    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            # Get first detection
            box = boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cropped_flag = img[y1:y2, x1:x2]
            return cropped_flag
    print("No object detected, using full image")
    return img

def identify_flag(flag_image):
   
    # Convert to HSV
    hsv = cv2.cvtColor(flag_image, cv2.COLOR_BGR2HSV)
    
    # Define red and white color ranges in HSV
    red_lower = np.array([0, 50, 50])
    red_upper = np.array([10, 255, 255])

    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    
    
    # Create masks
    mask_red1 = cv2.inRange(hsv, red_lower, red_upper)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # Split image into top and bottom halves
    height = flag_image.shape[0]
    top_half_red = mask_red[:height//2]
    bottom_half_red = mask_red[height//2:]

    # Count red pixels in each half
    red_top = cv2.countNonZero(top_half_red)
    red_bottom = cv2.countNonZero(bottom_half_red)
    
    # Classify based on red position
    if red_top > red_bottom:
        return "Indonesian Flag"
    else:
        return "Polish Flag"

def implementation():
    image_path = input("Enter the path to the flag image: ")
    
    
    # Task 1: Crop flag using YOLOv5
    cropped_flag = crop_flag(image_path)
    
    # Task 2: Classify Indonesian vs Polish
    result = identify_flag(cropped_flag)
    
    print(f"Result: {result}")

    cv2.imwrite("cropped_flag.jpg", cropped_flag)

implementation()