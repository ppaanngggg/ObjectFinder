from sklearn.cluster import KMeans
from pymongo import MongoClient
import numpy as np


def find_data(db_name, coll_name):
    client = MongoClient()
    db = client[db_name]
    coll = db[coll_name]
    data = []
    for one in coll.find():
        data.append(one)
    return data


def k_means_one_layer(data, key, n_clusters, parent, db_name, coll_name):
    X = [np.array(d[key]) for d in data]
    # print X
    k_means = KMeans(n_clusters=n_clusters)
    predict = k_means.fit_predict(X)
    # print predict
    mean = [np.zeros(X[0].shape) for i in range(n_clusters)]
    count = [0] * n_clusters
    for index in range(len(X)):
        mean[predict[index]] += X[index]
        count[predict[index]] += 1
    for index in range(len(mean)):
        mean[index] /= float(count[index])
    client = MongoClient()
    db = client[db_name]
    coll = db[coll_name]
    for m in mean:
        coll.insert({key: list(m), 'parent': parent})
    id_list = []
    for m in mean:
        one = coll.find_one({key: list(m)})
        id_list.append(one['_id'])
    cluster_list = [[] for i in range(n_clusters)]
    # print id_list
    for index in range(len(predict)):
        data[index]['parent'] = id_list[predict[index]]
        cluster_list[predict[index]].append(data[index])
    # print cluster_list
    return cluster_list


def k_means_multi_layer(data, key, n_clusters, parent, db_name, coll_name, num):
    if len(data) < num * n_clusters:
        client = MongoClient()
        db = client[db_name]
        coll = db[coll_name]
        for d in data:
            coll.insert(d)
        return
    cluster_list = k_means_one_layer(data, key, n_clusters, parent, db_name, coll_name)
    for cluster in cluster_list:
        k_means_multi_layer(cluster, key, n_clusters, cluster[0]['parent'], db_name, coll_name, num)


def print_k_means(db_name, coll_name, key, parent='none', tab_num=0):
    client = MongoClient()
    db = client[db_name]
    coll = db[coll_name]
    for one in coll.find({'parent': parent}):
        print '\t' * tab_num, one[key]
        print_k_means(db_name, coll_name, key, one['_id'], tab_num + 1)


def find_k_means(vec, db_name, coll_name, key):
    client = MongoClient()
    db = client[db_name]
    coll = db[coll_name]
    parent = 'none'
    result = []
    best_result=None
    while True:
        data = []
        for one in coll.find({'parent': parent}):
            data.append(one)
        if not data:
            break
        result = data
        X = [np.array(d[key]) for d in data]
        min_index = 0
        min_distance = np.sum((np.array(vec) - X[0]) ** 2)
        for index in range(1, len(X)):
            distance = np.sum((np.array(vec) - X[index]) ** 2)
            if distance < min_distance:
                min_index = index
                min_distance = distance
        best_result=data[min_index]
        parent = data[min_index]['_id']

    return result,best_result


def test():
    k_means_multi_layer(find_data('test', 'test'), 'des', 2, 'none', 'test', 'test_2', 2)
    print_k_means('test', 'test_2', 'des')
    print find_k_means([9.8, 8.5, 10.6], 'test', 'test_2', 'des')


if __name__ == '__main__':
    test()