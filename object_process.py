from image_process import ImageProcess
import cv2
import copy
from pymongo import MongoClient
import train
import threading
import numpy as np
from k_means_multi_layer import *


class ObjectProcess:
    def __init__(self, path, clf_fore, clf_shape):
        str_list = path.split('/')
        self.kind = str_list[-2]
        self.name = str_list[-1]

        self.clf_fore = clf_fore
        self.clf_shape = clf_shape

        self.image = cv2.imread(path)
        self.image = cv2.resize(self.image, (400, 400))
        # cv2.imshow('img',self.image)
        # cv2.waitKey()

        thread_s = threading.Thread(target=self.init_image_proc_s)
        thread_m = threading.Thread(target=self.init_image_proc_m)
        thread_l = threading.Thread(target=self.init_image_proc_l)
        thread_s.start()
        thread_m.start()
        thread_l.start()
        thread_s.join()
        thread_m.join()
        thread_l.join()

        self.hog_list = []
        hog_list = self.image_proc_m.get_hog_list() + \
                   self.image_proc_l.get_hog_list()
        target_list = list(self.image_proc_m.get_classify_target_list()) + \
                      list(self.image_proc_l.get_classify_target_list())
        for index in range(len(hog_list)):
            if target_list[index]:
                self.hog_list.append(hog_list[index])

        self.client = MongoClient()
        self.db = self.client.object_finder

    def init_image_proc_s(self):
        self.image_proc_s = ImageProcess(
            cv2.medianBlur(self.image, 5), 70
        )
        self.image_proc_s.set_classify_target_list(
            self.clf_fore.predict(
                self.image_proc_s.get_classify_vec_list()
            )
        )
        self.image_proc_s.image = cv2.bilateralFilter(self.image, 5, 50, 50)
        self.image_proc_s.compute_foreground_mask()
        self.image_proc_s.compute_foreground_image()
        self.image_proc_s.compute_color_list()
        self.image_proc_s.compute_sift_list()

    def init_image_proc_m(self):
        self.image_proc_m = ImageProcess(
            cv2.bilateralFilter(self.image, 5, 50, 50), 100
        )
        self.image_proc_m.set_classify_target_list(
            self.clf_shape.predict(
                self.image_proc_m.get_classify_vec_list()
            )
        )

    def init_image_proc_l(self):
        self.image_proc_l = ImageProcess(
            cv2.bilateralFilter(self.image, 5, 50, 50), 130
        )
        self.image_proc_l.set_classify_target_list(
            self.clf_shape.predict(
                self.image_proc_l.get_classify_vec_list()
            )
        )

    def get_image(self):
        return copy.deepcopy(self.image)

    def get_fore(self):
        return copy.deepcopy(self.image_proc_s.get_foreground_image())

    def get_seg_image_list(self):
        return [
            np.array(self.image_proc_s.get_mark_image() * 255, dtype=np.uint8),
            np.array(self.image_proc_m.get_mark_image() * 255, dtype=np.uint8),
            np.array(self.image_proc_l.get_mark_image() * 255, dtype=np.uint8)
        ]

    def get_pos_hog_image_list(self):
        try:
            return copy.deepcopy(self.hog_image_list)
        except:
            self.hog_image_list = []
            tmp_list = self.image_proc_m.get_hog_image_list() + \
                       self.image_proc_l.get_hog_image_list()
            target_list = list(self.image_proc_m.get_classify_target_list()) + \
                          list(self.image_proc_l.get_classify_target_list())
            for i in range(len(target_list)):
                if target_list[i]:
                    self.hog_image_list.append(tmp_list[i])
            return copy.deepcopy(self.hog_image_list)


    def get_color_list(self):
        return copy.deepcopy(self.image_proc_s.get_color_list())

    def get_sift_list(self):
        return copy.deepcopy(self.image_proc_s.get_sift_list())

    def get_hog_list(self):
        return copy.deepcopy(self.hog_list)

    def store_color_list(self):
        coll = self.db.color_list
        color_list=self.get_color_list()
        try:
            for color in color_list:
                coll.insert({
                    'vec': color,
                    'kind': self.kind,
                    'name': self.name
                })
        except:
            pass
        return self

    def store_sift_list(self):
        coll = self.db.sift_list
        sift_list = self.get_sift_list()
        try:
            for sift in sift_list:
                coll.insert({
                    'vec': [int(num) for num in sift],
                    'kind': self.kind,
                    'name': self.name
                })
        except:
            pass
        return self

    def store_hog_list(self):
        coll = self.db.hog_list
        hog_list = self.get_hog_list()
        try:
            for hog in hog_list:
                coll.insert({
                    'vec': hog,
                    'kind': self.kind,
                    'name': self.name
                })
        except:
            pass
        return self

    def write_fore_image(self, img_base):
        cv2.imwrite(
            img_base + '/' + self.kind + '/' + self.name,
            self.image_proc_s.get_foreground_image()
        )

    def insert_into_dict(self, fit, arg_dict):
        if fit['kind'] in arg_dict.keys():
            if fit['name'][:-4] in arg_dict[fit['kind']].keys():
                arg_dict[fit['kind']][fit['name'][:-4]] += 1
            else:
                arg_dict[fit['kind']][fit['name'][:-4]] = 1
        else:
            arg_dict[fit['kind']] = {}
            arg_dict[fit['kind']][fit['name'][:-4]] = 1

    def find_k_means_color_list(self):
        fit_list = []
        best_fit_list = []
        print len(self.get_color_list())
        for color in self.get_color_list():
            fit, best = find_k_means(
                color,
                'object_finder',
                'k_means_color_list',
                'vec'
            )
            fit_list += fit
            best_fit_list.append(best)
        self.fit_color_dict = {}
        for fit in fit_list:
            self.insert_into_dict(fit, self.fit_color_dict)
        self.best_fit_color_dict = {}
        for best_fit in best_fit_list:
            self.insert_into_dict(best_fit, self.best_fit_color_dict)

    def get_fit_color_dict(self):
        try:
            return copy.deepcopy(self.fit_color_dict)
        except:
            self.find_k_means_color_list()
            return copy.deepcopy(self.fit_color_dict)

    def get_best_fit_color_dict(self):
        try:
            return copy.deepcopy(self.best_fit_color_dict)
        except:
            self.find_k_means_color_list()
            return copy.deepcopy(self.best_fit_color_dict)

    def find_k_means_sift_list(self):
        fit_list = []
        best_fit_list = []
        try:
            print len(self.get_sift_list())
            for sift in self.get_sift_list():
                fit, best = find_k_means(
                    sift,
                    'object_finder',
                    'k_means_sift_list',
                    'vec'
                )
                fit_list += fit
                best_fit_list.append(best)
        except:
            print 0
        self.fit_sift_dict = {}
        for fit in fit_list:
            self.insert_into_dict(fit, self.fit_sift_dict)
        self.best_fit_sift_dict = {}
        for best_fit in best_fit_list:
            self.insert_into_dict(best_fit, self.best_fit_sift_dict)

    def get_fit_sift_dict(self):
        try:
            return copy.deepcopy(self.fit_sift_dict)
        except:
            self.find_k_means_sift_list()
            return copy.deepcopy(self.fit_sift_dict)

    def get_best_fit_sift_dict(self):
        try:
            return copy.deepcopy(self.best_fit_sift_dict)
        except:
            self.find_k_means_sift_list()
            return copy.deepcopy(self.best_fit_sift_dict)

    def find_k_means_hog_list(self):
        fit_list = []
        best_fit_list = []
        for hog in self.get_hog_list():
            fit, best = find_k_means(
                hog,
                'object_finder',
                'k_means_hog_list',
                'vec'
            )
            fit_list += fit
            best_fit_list.append(best)
        self.fit_hog_dict = {}
        for fit in fit_list:
            self.insert_into_dict(fit, self.fit_hog_dict)
        self.best_fit_hog_dict = {}
        for best_fit in best_fit_list:
            self.insert_into_dict(best_fit, self.best_fit_hog_dict)

    def get_fit_hog_dict(self):
        try:
            return copy.deepcopy(self.fit_hog_dict)
        except:
            self.find_k_means_hog_list()
            return copy.deepcopy(self.fit_hog_dict)

    def get_best_fit_hog_dict(self):
        try:
            return copy.deepcopy(self.best_fit_hog_dict)
        except:
            self.find_k_means_hog_list()
            return copy.deepcopy(self.best_fit_hog_dict)


def test():
    import pickle
    f = open('cache/clf_fore', 'r')
    clf_fore = pickle.load(f)
    f.close()
    f = open('cache/clf_shape', 'r')
    clf_shape = pickle.load(f)
    f.close()
    obj = ObjectProcess('test_pic/0.jpg', clf_fore, clf_shape)
    print obj.get_fit_color_dict(), obj.get_best_fit_color_dict()
    # print obj.get_fit_sift_dict(), obj.get_best_fit_sift_dict()
    # print obj.get_fit_hog_dict(), obj.get_best_fit_hog_dict()


if __name__ == '__main__':
    test()