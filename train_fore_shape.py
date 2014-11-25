from pymongo import MongoClient
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

def find_sample(data_set):
    client = MongoClient()
    db = client.object_finder
    collection = db[data_set]
    vec_list = []
    target_list = []
    for sample in collection.find():
        vec_list.append(sample['vec'])
        target_list.append(sample['target'])
    vec = np.array(vec_list)
    target = np.array(target_list)
    return vec, target


def train_sample(mode):
    vec_list, target_list = find_sample('train_'+mode)
    clf = GradientBoostingClassifier().fit(vec_list, target_list)
    print clf.score(vec_list, target_list)
    return clf


def test_sample(clf,mode):
    vec_list, target_list = find_sample('test_'+mode)
    print clf.score(vec_list, target_list)


def main():
    import pickle
    clf_fore = train_sample('fore')
    f = open('cache/clf_fore', 'w')
    pickle.dump(clf_fore, f)
    f.close()
    test_sample(clf_fore,'fore')
    clf_shape = train_sample('shape')
    f = open('cache/clf_shape', 'w')
    pickle.dump(clf_shape, f)
    f.close()
    test_sample(clf_shape,'shape')

if __name__=='__main__':
    main()