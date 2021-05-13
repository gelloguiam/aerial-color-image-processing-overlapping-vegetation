#Author: Angelo C. Guiam, CMSC 265 2S AY 2020-2021

import cv2
import imutils
import numpy as np
import skimage

from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed

plot_image = []
plot_title = []

#read the image
image = cv2.imread('input.png')
(ROW, COL, D) = image.shape

plot_image.append(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB))
plot_title.append("Original Image")

#convert the image into HSV
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#set HSV range for green color
green_hsv_min = np.array([30, 65, 20])
green_hsv_max = np.array([75, 255, 255])

#get a mask using the green range
mask = cv2.inRange(image_hsv, green_hsv_min, green_hsv_max)

#extract the vegetation using the mask
vegetation = cv2.bitwise_and(image, image, mask=mask)

#extract the non-vegetation area by subtracting the original image of the vegetation image
non_vegetation = image.copy() - vegetation

plot_image.append(cv2.cvtColor(vegetation.copy(), cv2.COLOR_BGR2RGB))
plot_title.append("Extracted Vegetation")

#convert image to grayscale
image_gray = cv2.cvtColor(vegetation, cv2.COLOR_BGR2GRAY)

#binarize image
ret, image_binary = cv2.threshold(image_gray, 40, 255, cv2.THRESH_BINARY)

plot_image.append(image_binary)
plot_title.append("After Binarization")

kernel = np.ones((3, 3), np.uint8)
image_erode = cv2.morphologyEx(image_binary, cv2.MORPH_ERODE, kernel)

kernel = np.ones((7, 7), np.uint8)
image_mclose = cv2.morphologyEx(image_erode, cv2.MORPH_CLOSE, kernel)

kernel = np.ones((5, 5), np.uint8)
image_mopen = cv2.morphologyEx(image_mclose, cv2.MORPH_OPEN, kernel)

plot_image.append(image_mopen)
plot_title.append("After Morphological Transformations")

#calculate distance of the centroid to the edges
image_dt = ndimage.distance_transform_edt(image_mopen)
#filter tha maximum distance based on some parameters
local_max = peak_local_max(image_dt, indices=False, min_distance=9, labels=image_binary)

#perform a connected component analysis on the local peaks,
#using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
labels = watershed(-image_dt, markers, mask=image_mopen)

print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

grass = image.copy()
count_tree = 0
count_non_tree = 0

for label in np.unique(labels):
	if label == 0:
		continue

	#create a max using the matrix of local maxima
	mask = np.zeros(image_gray.shape, dtype="uint8")
	mask[labels == label] = 255

	#generate contours from the mask
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	#get the bounds of the contour
	c = max(cnts, key=cv2.contourArea)
	((x, y), r) = cv2.minEnclosingCircle(c)

	#filter out assumed tree contours
	if r > 24 and r < 57:
		count_tree = count_tree + 1
		cv2.circle(image, (int(x), int(y)), int(r), (0, 0, 255), 1)
		cv2.putText(image, "{}".format(count_tree), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

	#else, categorize them as other / grass vegetation
	else:
		count_non_tree = count_non_tree + 1
		cv2.circle(grass, (int(x), int(y)), int(r), (255, 0, 0), 1)
		cv2.putText(grass, "{}".format(count_non_tree), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

#annotate total count
cv2.putText(image, "Total: {}".format(count_tree), (round(COL*0.8), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.putText(grass, "Total: {}".format(count_non_tree), (round(COL*0.8), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imwrite("result-trees-vegetation.png", image)
cv2.imwrite("result-non-trees-vegetation.png", grass)

plot_image.append(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB))
plot_title.append("Detected Tree Crowns")

plot_image.append(cv2.cvtColor(grass.copy(), cv2.COLOR_BGR2RGB))
plot_title.append("Detected Non-Tree Vegetation")

f, ax = plt.subplots(2,3)
idx = 0

for i in range(0, 2):
	for j in range(0, 3):
		ax[i,j].imshow(plot_image[idx], cmap='gray')
		ax[i,j].set_title(plot_title[idx], color='black')
		idx = idx + 1

plt.show()

#References
#Mordvintsev, A., & Abid, K. (2013). Image Segmentation with Watershed Algorithm — OpenCV-Python Tutorials 1 documentation. OpenCV. https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html
#Rosebrock, A. (2021, April 17). Watershed OpenCV. PyImageSearch. https://www.pyimagesearch.com/2015/11/02/watershed-opencv/
