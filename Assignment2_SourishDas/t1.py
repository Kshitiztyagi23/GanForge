import cv2
import numpy as np
import matplotlib.pyplot as plt

# Capturing the image with webcamera
def capture_image(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError("Failed to capture image")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Convert to HSV and return image + HSV channels
def convert_to_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    return hsv, h, s, v

# Histogram equalization on grayscale
def equalize_histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    eq = cv2.equalizeHist(gray)
    return gray, eq

# Binary inversion thresholding
def binary_inversion(image, threshold=127):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary

# Posterization (reduce to 4 gray levels)
def posterize_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    poster = (gray // 64) * 64  # 256 / 4 = 64
    return poster

# Laplacian and Scharr edge detection
def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    scharr = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    return np.uint8(np.abs(laplacian)), np.uint8(np.abs(scharr))

# Salt and pepper noise + median filter
def denoise_median(image):
    noisy = image.copy()
    h, w, _ = noisy.shape
    num_noise = int(0.01 * h * w)

    for _ in range(num_noise):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        noisy[y, x] = np.random.choice([0, 255], 3)

    denoised = cv2.medianBlur(noisy, 3)
    return noisy, denoised

# Unsharp masking
def unsharp_mask(image, amount=1.5):
    blurred = cv2.GaussianBlur(image, (9, 9), 10)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharpened

# RGB to LAB conversion
def convert_to_lab(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    return lab, l, a, b



image = capture_image()

_, h, s, v = convert_to_hsv(image)
gray, eq = equalize_histogram(image)
binary = binary_inversion(image)
poster = posterize_gray(image)
laplacian, scharr = edge_detection(image)
noisy, denoised = denoise_median(image)
sharpened = unsharp_mask(image)
_, l, a, b = convert_to_lab(image)


fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Image Processing Tasks", fontsize=16)

axes[0, 0].imshow(h, cmap='hsv')
axes[0, 0].set_title("Hue Channel")
axes[0, 1].imshow(eq, cmap='gray')
axes[0, 1].set_title("Histogram Equalized")
axes[0, 2].imshow(binary, cmap='gray')
axes[0, 2].set_title("Binary Inversion")
axes[0, 3].imshow(poster, cmap='gray')
axes[0, 3].set_title("Posterized (4 levels)")

axes[1, 0].imshow(laplacian, cmap='gray')
axes[1, 0].set_title("Laplacian Edge")
axes[1, 1].imshow(denoised)
axes[1, 1].set_title("Denoised Median")
axes[1, 2].imshow(sharpened)
axes[1, 2].set_title("Unsharp Masking")
axes[1, 3].imshow(l, cmap='gray')
axes[1, 3].set_title("L Channel (LAB)")

for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
