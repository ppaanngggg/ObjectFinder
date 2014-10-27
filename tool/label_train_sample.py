import sys
sys.path.append('..')
import cv2
import segmentation
from pymongo import MongoClient
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

def get_sample(path):
    image=cv2.imread(path)
    # cv2.imshow('image',image)
    # cv2.waitKey()
    a=segmentation.segmentation(image)
    a.show_mark_image()
    a.label_classify_target_list(True)
    return a.get_classify_vec_list(),a.get_classify_target_list()

def save_sample():
    vec,target=get_sample('../train_pic/img.jpg')

    client=MongoClient()
    db=client.object_finder
    collection=db.seg_train_db
    collection.insert({'vec':vec,'target':target})
    tmp=collection.find_one()
    print tmp

def find_sample():
    client=MongoClient()
    db=client.object_finder
    collection=db.seg_train_db
    vec=[]
    target=[]
    for sample in collection.find():
        vec+=sample['vec']
        target+=sample['target']
    vec=np.array(vec)
    target=np.array(target)
    return vec,target

def train_sample():
    vec,target=find_sample()
    clf=GradientBoostingClassifier().fit(vec,target)
