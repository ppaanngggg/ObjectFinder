import sys

sys.path.append('..')
import cv2
import segmentation
from pymongo import MongoClient
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


def get_sample(path):
    image = cv2.imread(path)
    # cv2.imshow('image',image)
    # cv2.waitKey()
    seg_50 = segmentation.segmentation(image, 50)
    seg_50.show_mark_image(False)
    seg_50.label_classify_target_list(True)
    seg_75 = segmentation.segmentation(image, 75)
    seg_75.show_mark_image(False)
    seg_75.label_classify_target_list(True)
    seg_100 = segmentation.segmentation(image, 100)
    seg_100.show_mark_image(False)
    seg_100.label_classify_target_list(True)
    return seg_50.get_classify_vec_list() + seg_75.get_classify_vec_list() + seg_100.get_classify_vec_list(), \
           seg_50.get_classify_target_list() + seg_75.get_classify_target_list() + seg_100.get_classify_target_list()


def save_sample(filename, data_set):
    vec, target = get_sample('../train_pic/' + filename)

    client = MongoClient()
    db = client.object_finder
    collection = db['seg_' + data_set + '_db']
    collection.insert({'vec': vec, 'target': target, 'file': filename})
    tmp = collection.find_one()
    print tmp


def find_sample(data_set):
    client = MongoClient()
    db = client.object_finder
    collection = db['seg_' + data_set + '_db']
    vec = []
    target = []
    for sample in collection.find():
        vec += sample['vec']
        target += sample['target']
    vec = np.array(vec)
    target = np.array(target)
    return vec, target


def train_sample():
    vec, target = find_sample('train')
    clf = GradientBoostingClassifier().fit(vec, target)
    return clf


def test_sample(clf):
    vec, target = find_sample('test')
    print clf.score(vec, target)

# save_sample('90.jpg','train')
# test_sample(train_sample())
get_sample('../train_pic/90.jpg')