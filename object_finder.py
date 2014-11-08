import cv2
import train
from object_process import ObjectProcess
import numpy as np
import k_means_multi_layer
from pymongo import MongoClient


class ObjectFinder:
    def __init__(self,path):

# filename = '3.jpg'
# img = cv2.imread('test_pic/' + filename)
# img = cv2.resize(img, (400, 400))
# clf_fore = train.train_sample('fore')
# clf_shape = train.train_sample('shape')
# seg_s = segmentation.segmentation(img, 70)
# seg_s.set_classify_target_list(clf_fore.predict(seg_s.get_classify_vec_list()))
# seg_m = segmentation.segmentation(img, 100)
# seg_m.set_classify_target_list(clf_shape.predict(seg_m.get_classify_vec_list()))
# seg_l = segmentation.segmentation(img, 130)
# seg_l.set_classify_target_list(clf_shape.predict(seg_l.get_classify_vec_list()))
#
# seg_s.compute_foreground_mask()
# seg_s.compute_foreground_image()
# cv2.imshow('seg', seg_s.get_foreground_image())
# cv2.waitKey()
#
# for image in seg_m.hog_image_list + seg_l.hog_image_list:
#     cv2.imshow('hog', image)
#     cv2.waitKey()
#
# print '**** hist ****'
# result_color_dict = {}
#
# img_list = cv2.split(seg_s.get_foreground_image())
# hist_list = [
#     cv2.calcHist([img], [0], seg_s.get_foreground_mask(), [16], [0, 256])
#     for img in img_list
# ]
# hist = []
# for h in hist_list:
#     h /= np.sum(h)
#     hist += [float(num) for num in list(h)]
# result_list, best_result = k_means_multi_layer.find_k_means(hist, 'object_finder', 'k_means_color_hist', 'hist')
# # for result in result_list:
# for result in [best_result]:
#     if result['file'] in result_color_dict.keys():
#         result_color_dict[result['file'][:-4]] += 1
#     else:
#         result_color_dict[result['file'][:-4]] = 1
#
# print '**** ORB ****'
# result_visual_word_dict = {}
#
# orb = cv2.ORB_create()
# kp_list = orb.detect(seg_s.get_image(), seg_s.get_foreground_mask())
# kp_list, des_list = orb.compute(seg_s.get_image(), kp_list)
# for des in des_list:
#     result_list, best_result = k_means_multi_layer.find_k_means(list(des), 'object_finder',
#                                                                 'k_means_bag_of_visual_word', 'des')
#     # for result in result_list:
#     for result in [best_result]:
#         if result['file'] in result_visual_word_dict.keys():
#             result_visual_word_dict[result['file'][:-4]] += 1
#         else:
#             result_visual_word_dict[result['file'][:-4]] = 1
#
# print '**** hog ****'
# result_boundary_dict = {}
#
# for hog in seg_m.get_hog_list() + seg_l.get_hog_list():
#     result_list, best_result = k_means_multi_layer.find_k_means(list(hog), 'object_finder', 'k_means_bag_of_boundary',
#                                                                 'hog')
#     # for result in result_list:
#     for result in [best_result]:
#         if result['file'] in result_boundary_dict.keys():
#             result_boundary_dict[result['file'][:-4]] += 1
#         else:
#             result_boundary_dict[result['file'][:-4]] = 1
#
# print result_color_dict
# print result_visual_word_dict
# print result_boundary_dict
#
# client = MongoClient()
# db = client.object_finder
# coll = db.test_result
# coll.insert(
#     {
#         'file': filename,
#         'hist': result_color_dict,
#         'ORB': result_visual_word_dict,
#         'hog': result_boundary_dict
#     }
# )