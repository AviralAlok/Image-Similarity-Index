# 1. Import the necessary packages
from skimage.measure import compare_ssim
import imutils
import cv2
import numpy as numpy
import matplotlib as plt

# 2. Load the two input images
imageA = cv2.imread(r"C:\Users\AVIRAL\Desktop\Images\printer1.jpg")
imageB = cv2.imread(r"C:\Users\AVIRAL\Desktop\Images\Football2.jpg")
k=min(imageA.shape[0],imageB.shape[0])
r=min(imageA.shape[1],imageB.shape[1])

imageA=cv2.resize(imageA,(r,k))
imageB=cv2.resize(imageB,(r,k))

# 3. Convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
#print (imageA.shape[0])

# 4. Computing the Structural Similarity Index (SSIM) between the two
#    images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
window_name = 'image'

# 5. Print SSIM
print("SSIM: {}".format(score))