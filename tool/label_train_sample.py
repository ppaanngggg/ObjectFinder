import sys

sys.path.append('..')
import cv2
import segmentation
from pymongo import MongoClient
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


def get_sample(path,mode):
    image = cv2.imread(path)
    print mode
    if mode=='fore':
        seg_s = segmentation.segmentation(image, 70)
        seg_s.label_classify_target_list(True)
        return seg_s.get_classify_vec_list(),seg_s.get_classify_target_list()
    else:
        seg_m = segmentation.segmentation(image, 100)
        seg_m.label_classify_target_list(True)
        seg_l = segmentation.segmentation(image, 130)
        seg_l.label_classify_target_list(True)
    return seg_m.get_classify_vec_list() + seg_l.get_classify_vec_list(), \
           seg_m.get_classify_target_list() + seg_l.get_classify_target_list()


def save_sample(filename, data_set):
    if data_set[0:4]=='fore':
        vec, target = get_sample('../train_pic/' + filename,'fore')
    else:
        vec, target = get_sample('../train_pic/' + filename,'shape')

    client = MongoClient()
    db = client.object_finder
    collection = db[data_set]
    collection.insert({'vec': vec, 'target': target, 'file': filename})
    tmp = collection.find_one()
    print tmp


def find_sample(data_set):
    client = MongoClient()
    db = client.object_finder
    collection = db[data_set]
    vec = []
    target = []
    for sample in collection.find():
        vec += sample['vec']
        target += sample['target']
    vec = np.array(vec)
    target = np.array(target)
    return vec, target


def train_sample(mode):
    vec, target = find_sample(mode+'_train')
    clf = GradientBoostingClassifier().fit(vec, target)
    print clf.score(vec, target)
    return clf


def test_sample(clf,mode):
    vec, target = find_sample(mode+'_test')
    print clf.score(vec, target)

for i in range(340,450,10):
    print '********  file',i,' ********'
    save_sample(str(i)+'.jpg','fore_train')

for i in range(500,550,10):
    print '********  file',i,' ********'
    save_sample(str(i)+'.jpg','fore_test')

test_sample(train_sample('fore'),'fore')