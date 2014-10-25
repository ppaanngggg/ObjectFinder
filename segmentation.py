import cv2
import numpy as np
image=cv2.imread('train_pic/img.jpg')
image_height,image_width,image_channels=image.shape
num_superpixels=image_height*image_width/70/70
num_levels=4
seed=cv2.ximgproc.createSuperpixelSEEDS(
    image_width,
    image_height,
    image_channels,
    num_superpixels,
    num_levels
)
converted_img=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
num_iterations=5
seed.iterate(converted_img,num_iterations)
labels=seed.getLabels()
labels_show=labels/float(np.max(labels))
cv2.imshow('labels',labels_show)
label_contour=seed.getLabelContourMask()
for i in range(image_width):
    for j in range(image_height):
        if label_contour[i,j]:
            image[i,j]=(0,0,255)
cv2.imshow('contours',image)
cv2.waitKey()