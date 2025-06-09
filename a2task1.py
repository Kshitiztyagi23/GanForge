# Image Processing Assignment
# Date: 07-06-2025

import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_to_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return h, s, v

def histogram_equalization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return gray, eq

def binary_inversion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return binary

def posterize_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bins = np.linspace(0, 256, 5)
    poster = np.digitize(gray, bins) * 64 - 64
    poster = np.clip(poster, 0, 255).astype(np.uint8)
    return poster

def edge_filters(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    scharr = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr = cv2.convertScaleAbs(scharr)
    return lap, scharr

def median_filter(img):
    noisy = img.copy()
    row, col, ch = noisy.shape
    s_vs_p = 0.5
    amount = 0.04

    num_salt = np.ceil(amount * img.size * s_vs_p / ch).astype(int)
    coords_salt = [np.random.randint(0, i - 1, num_salt) for i in noisy.shape[:2]]
    noisy[coords_salt[0], coords_salt[1], :] = 255

    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p) / ch).astype(int)
    coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in noisy.shape[:2]]
    noisy[coords_pepper[0], coords_pepper[1], :] = 0

    median = cv2.medianBlur(noisy, 3)
    return noisy, median

def unsharp_mask(img):
    blur = cv2.GaussianBlur(img, (9, 9), 10.0)
    unsharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return blur, unsharp

def convert_to_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return l, a, b

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    ret, img = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture image")
        return

    img = cv2.resize(img, (320, 240))

    # Processing steps
    h, s, v = convert_to_hsv(img)
    gray, eq = histogram_equalization(img)
    binary = binary_inversion(img)
    poster = posterize_gray(img)
    lap, scharr = edge_filters(img)
    noisy, median = median_filter(img)
    blur, unsharp = unsharp_mask(img)
    
    # Display all results
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 4, 1)
    plt.title("Hue (HSV)")
    plt.imshow(h, cmap='hsv')
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.title("Histogram Equalization")
    plt.imshow(eq, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.title("Binary Inversion")
    plt.imshow(binary, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plt.title("Posterized (4 levels)")
    plt.imshow(poster, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 4, 5)
    plt.title("Laplacian Edge")
    plt.imshow(lap, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.title("Scharr Edge")
    plt.imshow(scharr, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 4, 7)
    plt.title("Median Filter")
    plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.title("Unsharp Masking")
    plt.imshow(cv2.cvtColor(unsharp, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
