import cv2
import matplotlib.pyplot as plt 
import numpy as np

ind = cv2.imread('indonesia 2.jpeg')
pol = cv2.imread('poland 2.jpeg')

ind = cv2.cvtColor(ind, cv2.COLOR_BGR2GRAY)
pol = cv2.cvtColor(pol, cv2.COLOR_BGR2GRAY)
flag = ind.copy()
ret, binary_image = cv2.threshold(flag, 127, 255, cv2.THRESH_BINARY)

h, w = flag.shape
upper_half = flag[:h//2, :]
lower_half = flag[h//2:, :]

mean_up = np.mean(upper_half)
mean_down = np.mean(lower_half)

if mean_up > mean_down :
    result = 'Poland'
else : 
    result = 'Indonesia' 

print(result)      