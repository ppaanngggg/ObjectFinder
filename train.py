from pymongo import MongoClient
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

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

# test_sample(train_sample('fore'),'fore')