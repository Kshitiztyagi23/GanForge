import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def convert_to_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    return h, s, v

def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    return gray, equalized

def binary_inversion(image, threshold=128):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_inv = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary_inv


def reduce_gray_levels(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    reduced = (gray // 64) * 64
    return reduced

def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    scharr = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    return np.uint8(np.abs(laplacian)), np.uint8(np.abs(scharr))

def add_salt_pepper_noise(image, amount=0.01):
    noisy = image.copy()
    h, w, _ = noisy.shape
    num_salt = int(amount * h * w)
    for _ in range(num_salt):
        i = random.randint(0, h - 1)
        j = random.randint(0, w - 1)
        noisy[i, j] = [255, 255, 255] if random.random() > 0.5 else [0, 0, 0]
    return noisy

def denoise_median(image):
    return cv2.medianBlur(image, 5)

def unsharp_mask(image, ksize=(5, 5), amount=1.5):
    blurred = cv2.GaussianBlur(image, ksize, 0)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharpened

def convert_to_lab(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    return l, a, b

def display_all_outputs(image):
    hsv_h, hsv_s, hsv_v = convert_to_hsv(image)
    gray, equalized = histogram_equalization(image)
    binary_inv = binary_inversion(image)
    reduced = reduce_gray_levels(image)
    laplacian, scharr = edge_detection(image)
    noisy = add_salt_pepper_noise(image)
    denoised = denoise_median(noisy)
    sharpened = unsharp_mask(image)
    lab_l, lab_a, lab_b = convert_to_lab(image)

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    axs[0, 0].imshow(hsv_h, cmap='gray')
    axs[0, 0].set_title("Hue Channel")

    axs[0, 1].imshow(equalized, cmap='gray')
    axs[0, 1].set_title("Histogram Equalized")

    axs[0, 2].imshow(binary_inv, cmap='gray')
    axs[0, 2].set_title("Binary Inversion")

    axs[0, 3].imshow(reduced, cmap='gray')
    axs[0, 3].set_title("4-Level Grayscale")

    axs[1, 0].imshow(laplacian, cmap='gray')
    axs[1, 0].set_title("Laplacian Filter")

    axs[1, 1].imshow(denoised)
    axs[1, 1].set_title("Salt & Pepper Denoised")

    axs[1, 2].imshow(sharpened)
    axs[1, 2].set_title("Unsharp Masking")

    axs[1, 3].imshow(lab_l, cmap='gray')
    axs[1, 3].set_title("L Channel (Lightness)")

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image = capture_image()
    if image is not None:
        display_all_outputs(image)
    else:
        print("Image capture failed.")
