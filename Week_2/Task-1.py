import cv2
import numpy as np
import matplotlib.pyplot as plt

def capture_image():
    cap = cv2.VideoCapture(0)  
    ret, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def convert_to_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return hsv, cv2.split(hsv)

def equalize_histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    return gray, equalized

def apply_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return binary

def reduce_gray_levels(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return (gray // 64) * 64

def apply_edge_filters(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    scharr = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    return np.uint8(np.abs(laplacian)), np.uint8(np.abs(scharr))

def denoise_median(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    noisy = gray.copy()
    noisy[::10, ::10] = 0
    noisy[::15, ::15] = 255
    denoised = cv2.medianBlur(noisy, 3)
    return noisy, denoised

def unsharp_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return sharpened

def convert_to_lab(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    return cv2.split(lab)


img = capture_image()

hsv, (h, s, v) = convert_to_hsv(img)
gray, equalized = equalize_histogram(img)
binary = apply_threshold(img)
poster = reduce_gray_levels(img)
lap, sch = apply_edge_filters(img)
noisy, denoised = denoise_median(img)
sharpened = unsharp_mask(img)
L, A, B = convert_to_lab(img)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

axes[0].imshow(h, cmap='gray'); axes[0].set_title("Hue Channel")
axes[1].imshow(equalized, cmap='gray'); axes[1].set_title("Equalized Gray")
axes[2].imshow(binary, cmap='gray'); axes[2].set_title("Binary Threshold")
axes[3].imshow(poster, cmap='gray'); axes[3].set_title("4 Gray Levels")
axes[4].imshow(lap, cmap='gray'); axes[4].set_title("Laplacian Edge")
axes[5].imshow(denoised, cmap='gray'); axes[5].set_title("Denoised Median")
axes[6].imshow(sharpened, cmap='gray'); axes[6].set_title("Unsharp Masking")
axes[7].imshow(L, cmap='gray'); axes[7].set_title("L Channel (LAB)")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
