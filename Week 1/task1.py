import cv2
import numpy as np
import matplotlib.pyplot as plt



def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Webcam not accessible")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise Exception("Failed to capture image")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def extract_hsv_channels(hsv_image):
    return cv2.split(hsv_image)

def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    return gray, equalized

def binary_inversion(image, threshold=128):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary

def posterize_4_levels(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return (gray // 64) * 85



def laplacian_scharr_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    scharr = cv2.Scharr(gray, cv2.CV_64F, 1, 0) + cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    return np.uint8(np.absolute(laplacian)), np.uint8(np.absolute(scharr))

def add_salt_pepper_noise(image, amount=0.02):
    noisy = np.copy(image)
    row, col, ch = noisy.shape
    num_salt = int(amount * row * col)
    
   
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 255

   
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 0

    return noisy

def median_filter(image):
    return cv2.medianBlur(image, 3)

def unsharp_mask(image, blur_ksize=5, amount=1.5):
    blurred = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharpened

def convert_to_lab(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    return cv2.split(lab)


image = capture_image()
hsv = convert_to_hsv(image)
h, s, v = extract_hsv_channels(hsv)
gray, equalized = histogram_equalization(image)
binary = binary_inversion(image)
posterized = posterize_4_levels(image)
laplacian, scharr = laplacian_scharr_edge(image)
noisy = add_salt_pepper_noise(image)
denoised = median_filter(noisy)
sharpened = unsharp_mask(image)
l, a, b = convert_to_lab(image)



fig, axes = plt.subplots(4, 4, figsize=(18, 12))
axes = axes.ravel()

images = [
    image, h, s, v,
    gray, equalized, binary, posterized,
    laplacian, scharr, noisy, denoised,
    sharpened, l, a, b
]

titles = [
    "Original Image", "Hue Channel", "Saturation", "Value",
    "Grayscale", "Histogram Equalized", "Binary Inversion", "4-level Posterized",
    "Laplacian Edge", "Scharr Edge", "Noisy Image", "Median Filtered",
    "Unsharp Masking", "L Channel", "A Channel", "B Channel"
]

for i in range(16):
    cmap = 'gray' if len(images[i].shape) == 2 else None
    axes[i].imshow(images[i], cmap=cmap)
    axes[i].set_title(titles[i])
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("image_processing_results.png", dpi=300) 
plt.show()
