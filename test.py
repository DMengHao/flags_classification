import cv2
from matplotlib import pyplot as plt

image_path = './Pictures/train/    1963.png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()
blurred_frame = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=1, sigmaY=1)
plt.imshow(blurred_frame)
plt.show()
new_image = cv2.resize(blurred_frame, (400, 400))
plt.imshow(new_image)
plt.show()