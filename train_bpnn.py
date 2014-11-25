from path_process import PathProcess
from object_process import ObjectProcess
from pymongo import MongoClient
import pickle
import BPNN


def store_sample():
    import pickle
    f = open('cache/clf_fore', 'r')
    clf_fore = pickle.load(f)
    f.close()
    f = open('cache/clf_shape', 'r')
    clf_shape = pickle.load(f)
    f.close()

    client = MongoClient()
    db = client.object_finder
    coll = db.train_bpnn

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


def load_sample():
    client = MongoClient()
    db = client.object_finder
    coll = db.train_bpnn

    find_list = []
    for find in coll.find():
        find_list.append(find)
    return find_list

def to_vec(one):
    vec = []
    for i in ('color', 'best_color', 'sift', 'best_sift', 'hog', 'best_hog'):
        for j in ('cloth', 'cup', 'shore'):
            try:
                vec.append(sum(one[i][j].values()))
            except:
                vec.append(0)
    for i in range(0, len(vec), 3):
        s = sum(vec[i:i + 3])
        for j in range(i, i + 3):
            vec[j] /= (float(s)+10 ** -10)
    return vec

def train_bpnn():
    find_list = load_sample()
    vec_list = []
    target_list = []
    for one in find_list:
        vec=to_vec(one)
        vec_list.append(vec)
        target = []
        for i in ('cloth', 'cup', 'shore'):
            if one['kind'] == i:
                target.append(1)
            else:
                target.append(0)
        target_list.append(target)

    bpnn = BPNN.Bpnn(len(vec_list[0]), [6,3])
    sample_list = []
    for i in range(len(vec_list)):
        sample_list.append([vec_list[i], target_list[i]])
    bpnn.train(sample_list, 0.01)
    # for vec in vec_list:
    #     bpnn.compute(vec)
    #     print bpnn.output()
    print sum(bpnn.error(sample_list))
    return bpnn


if __name__ == '__main__':
    # store_sample()

    bpnn=train_bpnn()
    f = open('cache/bpnn', 'w')
    pickle.dump(bpnn, f)
    f.close()