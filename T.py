import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = './Temp/frame_pictures/      0.png'
image = cv2.imread(image_path)
plt.imshow(image, cmap='gray')
plt.show()