import cv2
import numpy as np
import skimage.segmentation as skseg
import skimage.io as skio
import hog

img = cv2.imread('train_pic/img.jpg')
segments = skseg.slic(img, img.shape[0] * img.shape[1] / 50 / 50, 60)
skio.imshow(skseg.mark_boundaries(img,segments))
# skio.show()
value_set = set()
for row in segments:
    for pixel in row:
        value_set.add(pixel)
segment_list = [np.zeros((img.shape[0], img.shape[1]), np.uint8) for i in range(len(value_set))]
for row in range(segments.shape[0]):
    for col in range(segments.shape[1]):
        segment_list[segments[row, col]][row, col] = 255

contour_list = cv2.findContours(segment_list[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour_max = \
    [contour for contour in contour_list[1] if contour.shape[0] == max([tmp.shape[0] for tmp in contour_list[1]])][0]
tmp_img=np.zeros((img.shape[0], img.shape[1]), np.uint8)
for point in contour_max:
    loc=point[0]
    tmp_img[loc[1],loc[0]]=255
rect=cv2.boundingRect(contour_max)
print rect
hog_img=np.copy(img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]])
skio.imshow(hog_img)
# skio.show()

hog.hog(hog_img,8,2,4,4,9)