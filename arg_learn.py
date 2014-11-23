from path_process import PathProcess
from object_process import ObjectProcess
import train
from pymongo import MongoClient
import pickle
import BPNN


def store_train_find_sample():
    import pickle
    f = open('cache/clf_fore', 'r')
    clf_fore = pickle.load(f)
    f.close()
    f = open('cache/clf_shape', 'r')
    clf_shape = pickle.load(f)
    f.close()

    client = MongoClient()
    db = client.object_finder
    coll = db.train_find

    path_proc = PathProcess('train_pic_arg')
    for kind in path_proc.get_kind_list():
        for path in path_proc.get_file_path_list(kind):
            print path
            obj_proc = ObjectProcess(path, clf_fore, clf_shape)
            coll.insert({
                'kind': obj_proc.kind,
                'name': obj_proc.name,
                'color': obj_proc.get_fit_color_dict(),
                'best_color': obj_proc.get_best_fit_color_dict(),
                'sift': obj_proc.get_fit_sift_dict(),
                'best_sift': obj_proc.get_best_fit_sift_dict(),
                'hog': obj_proc.get_fit_hog_dict(),
                'best_hog': obj_proc.get_best_fit_hog_dict()
            })


def load_train_find_sample():
    client = MongoClient()
    db = client.object_finder
    coll = db.train_find

    train_find_list = []
    for find in coll.find():
        train_find_list.append(find)
    return train_find_list

def to_vec(train_find):
    vec = []
    for i in ('color', 'best_color', 'sift', 'best_sift', 'hog', 'best_hog'):
        for j in ('cloth', 'cup', 'shore'):
            try:
                vec.append(sum(train_find[i][j].values()))
            except:
                vec.append(0)
    for i in range(0, len(vec), 3):
        s = sum(vec[i:i + 3])
        for j in range(i, i + 3):
            vec[j] /= (float(s)+10 ** -10)
    return vec

def train_arg():
    train_find_list = load_train_find_sample()
    vec_list = []
    target_list = []
    for train_find in train_find_list:
        vec=to_vec(train_find)
        vec_list.append(vec)
        target = []
        for i in ('cloth', 'cup', 'shore'):
            if train_find['kind'] == i:
                target.append(1)
            else:
                target.append(0)
        target_list.append(target)

    bpnn = BPNN.Bpnn(len(vec_list[0]), [6,3])
    sample_list = []
    for i in range(len(vec_list)):
        sample_list.append([vec_list[i], target_list[i]])
    bpnn.train(sample_list, 0.005)
    # for vec in vec_list:
    #     bpnn.compute(vec)
    #     print bpnn.output()
    print sum(bpnn.error(sample_list))
    return bpnn


if __name__ == '__main__':
    # store_train_find_sample()
    bpnn=train_arg()
    f = open('cache/bpnn', 'w')
    pickle.dump(bpnn, f)
    f.close()