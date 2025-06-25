import cv2

def checkFlag(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not read.")
    
    image = cv2.resize(image, (300, 200))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    top_half = hsv[:100, :]
    bottom_half = hsv[100:, :]

    def is_red(region):
        mask1 = cv2.inRange(region, (0, 70, 50), (10, 255, 255))
        mask2 = cv2.inRange(region, (170, 70, 50), (180, 255, 255))
        red_pixels = cv2.countNonZero(mask1 | mask2)
        return red_pixels / region.size > 0.2  

    def is_white(region):
        mask = cv2.inRange(region, (0, 0, 200), (180, 50, 255))
        white_pixels = cv2.countNonZero(mask)
        return white_pixels / region.size > 0.2

    if is_red(top_half) and is_white(bottom_half):
        return "Indonesia"
    elif is_white(top_half) and is_red(bottom_half):
        return "Poland"
    else:
        return "Uncertain"

