import cv2 
import matplotlib.pyplot as plt
import numpy as np



# CAPTURING THE IMAGE

capture = cv2.VideoCapture(0)

ret, pic = capture.read() # frame will be a numpy array reperesenting pixel values
if ret :
    cv2.imshow('Captured Image', pic) # func to show the captured image
    cv2.imwrite('captured_image.jpg', pic) # function to save the captured image
    print("Image captured and saved as captured_image.jpg")

    cv2.waitKey(0) # destroy the window after user presses any key
    cv2.destroyAllWindows()

capture.release() # frees up all windows that were used




# CONVERT THE IMAGE TO HSV COLOUR SPACE

hsv_image = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
hue, saturation, value = cv2.split(hsv_image)
fig, axs = plt.subplots(1, 3, figsize = (15, 5))
axs[0].imshow(hue, cmap = 'hsv')
axs[1].imshow(saturation, cmap = 'grey')
axs[2].imshow(value, cmap = 'grey')
axs[0].set_title('Hue')
axs[1].set_title('Saturation')
axs[2].set_title('Value')
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
plt.show()


# CONVERT TO GRAYSCALE ->> HISTROGRAM EQUALISATION

gray_image = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
equalized_image = cv2.equalizeHist(gray_image)
pic_rgb = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

fig, axs = plt.subplots(1, 2, figsize = (10, 5))
axs[0].imshow(equalized_image, cmap = 'gray')
axs[1].imshow(pic_rgb)
axs[0].set_title('Equalized Image')
axs[1].set_title('Original Image')
axs[0].axis('off')
axs[1].axis('off')
plt.show()


# BINARY INVERSION THRESHOLDING

ret, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
fig, axs = plt.subplots(1, 2, figsize = (10, 5))
axs[0].imshow(binary_image, cmap ='gray')
axs[1].imshow(pic_rgb)
axs[0].set_title('Binary Inversion Thresholding')
axs[1].set_title('Original Image')
axs[0].axis('off')
axs[1].axis('off')
plt.show()


# REDUCING THE IMAGE TO 4 GRAY INTENSITY LEVELS

posterized_image = (gray_image // 64) * 85 # pixel intensity : [0, 255]
fig, axs = plt.subplots(1, 2, figsize = (10, 5))
axs[0].imshow(posterized_image, cmap ='gray')
axs[1].imshow(pic_rgb)
axs[0].set_title('Posterized Image')
axs[1].set_title('Original Image')
axs[0].axis('off')
axs[1].axis('off')
plt.show()


# LAPLACIAN FUNC AND SCHARR FUNC

laplacian = cv2.Laplacian(pic_rgb, ddepth=cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

scharr_x = cv2.Scharr(pic_rgb, ddepth=cv2.CV_64F, dx=1, dy=0)
scharr_x = cv2.convertScaleAbs(scharr_x)
scharr_y = cv2.Scharr(pic_rgb, ddepth=cv2.CV_64F, dx=0, dy=1)
scharr_y = cv2.convertScaleAbs(scharr_y)
scharr_combined = cv2.addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0)

fig, axs = plt.subplots(1, 2, figsize = (10, 5))
axs[0].imshow(laplacian)
axs[1].imshow(scharr_combined)
axs[0].set_title('laplacian filter')
axs[1].set_title('scarr filter')
axs[0].axis('off')
axs[1].axis('off')
plt.show()

# REDUCING NOISE

noisy_image = pic_rgb.copy()
noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
prob = 0.01 # Add salt-and-pepper noise
thresh = 1 - prob
for i in range(pic_rgb.shape[0]):
    for j in range(pic_rgb.shape[1]):
        rand = np.random.rand()
        if rand < prob:
            noisy_image_rgb[i][j] = 0
        elif rand > thresh:
            noisy_image_rgb[i][j] = 255

denoised_image = cv2.medianBlur(pic, 3) # Apply median filter
denoised_image_rgb = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)

fig, axs = plt.subplots(1, 3, figsize = (15, 5))
axs[0].imshow(pic_rgb)
axs[1].imshow(noisy_image_rgb, cmap = 'grey')
axs[2].imshow(denoised_image_rgb, cmap = 'grey')
axs[0].set_title('Original')
axs[1].set_title('Noisy Image')
axs[2].set_title('Denoised image')
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
plt.show()


# UNSHARP MASKING FOR SHARPENING

blurred = cv2.GaussianBlur(pic_rgb, (5, 5), 1.0)
mask = pic_rgb - blurred
amount = 1
sharpened_image = cv2.addWeighted(pic_rgb, 1.0, mask, amount, 0)
fig, axs = plt.subplots(1, 2, figsize = (10, 5))
axs[0].imshow(sharpened_image)
axs[1].imshow(pic_rgb)
axs[0].set_title('Sharpened Image')
axs[1].set_title('Original Image')
axs[0].axis('off')
axs[1].axis('off')
plt.show()


# RGB TO LAB COLOUR SPACE

lab_image = cv2.cvtColor(pic, cv2.COLOR_BGR2Lab)
lightness, a_red2greenAxis, b_yellow2blueAxis = cv2.split(lab_image)
fig, axs = plt.subplots(1, 3, figsize = (15, 5))
axs[0].imshow(lightness, cmap = 'gray')
axs[1].imshow(a_red2greenAxis, cmap = 'RdGy')
axs[2].imshow(b_yellow2blueAxis, cmap = 'bwr')
axs[0].set_title('Lightness')
axs[1].set_title('Red - Green Axis')
axs[2].set_title('Yellow - Blue Axis')
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
plt.show()