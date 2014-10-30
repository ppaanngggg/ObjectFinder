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
    seg_s = segmentation.segmentation(image, 60)
    seg_s.label_classify_target_list(True)
    seg_m = segmentation.segmentation(image, 90)
    seg_m.label_classify_target_list(True)
    seg_l = segmentation.segmentation(image, 120)
    seg_l.label_classify_target_list(True)
    return seg_s.get_classify_vec_list() + seg_m.get_classify_vec_list() + seg_l.get_classify_vec_list(), \
           seg_s.get_classify_target_list() + seg_m.get_classify_target_list() + seg_l.get_classify_target_list()


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
    print clf.score(vec, target)
    return clf


def test_sample(clf):
    vec, target = find_sample('test')
    print clf.score(vec, target)

for i in range(640,750,10):
    print '********  file',i,' ********'
    save_sample(str(i)+'.jpg','train')
# save_sample('640.jpg','train')
# test_sample(train_sample())