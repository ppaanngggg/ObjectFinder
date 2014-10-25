import cv2
import numpy as np
import skimage.segmentation as skseg
import skimage.io as skio

img=cv2.imread('train_pic/img.jpg')
segments=skseg.slic(img,img.shape[0]*img.shape[1]/50/50,50)
value_set=set()
for row in segments:
    for pixel in row:
        value_set.add(pixel)
segment_list=[np.zeros((img.shape[0],img.shape[1]),np.uint8)for i in range(len(value_set))]
for row in range(segments.shape[0]):
    for col in range(segments.shape[1]):
        segment_list[segments[row,col]][row,col]=255
# for segment in segment_list:
#     skio.imshow(segment)
#     skio.show()
cv2.findContours(segment_list[0],cv2.)