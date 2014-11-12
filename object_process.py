from image_process import ImageProcess
import cv2
import copy
from pymongo import MongoClient
import train
import threading
import numpy as np


class ObjectProcess:
    def __init__(self, path, clf_fore, clf_shape):
        str_list = path.split('/')
        self.kind = str_list[-2]
        self.name = str_list[-1]

        self.clf_fore = clf_fore
        self.clf_shape = clf_shape

        self.image = cv2.imread(path)
        self.image = cv2.resize(self.image, (400, 400))

        thread_s = threading.Thread(target=self.init_image_proc_s())
        thread_m = threading.Thread(target=self.init_image_proc_m())
        thread_l = threading.Thread(target=self.init_image_proc_l())
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
        self.image_proc_s = ImageProcess(self.image, 70)
        self.image_proc_s.set_classify_target_list(
            self.clf_fore.predict(
                self.image_proc_s.get_classify_vec_list()
            )
        )
        self.image_proc_s.compute_foreground_mask()
        self.image_proc_s.compute_foreground_image()
        self.image_proc_s.compute_color_hist()
        self.image_proc_s.compute_ORB_list()

    def init_image_proc_m(self):
        self.image_proc_m = ImageProcess(self.image, 100)
        self.image_proc_m.set_classify_target_list(
            self.clf_shape.predict(
                self.image_proc_m.get_classify_vec_list()
            )
        )

    def init_image_proc_l(self):
        self.image_proc_l = ImageProcess(self.image, 130)
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

    def get_color_hist(self):
        return copy.deepcopy(self.image_proc_s.get_color_hist())

    def get_ORB_list(self):
        return copy.deepcopy(self.image_proc_s.get_ORB_list())

    def get_hog_list(self):
        return copy.deepcopy(self.hog_list)

    def store_color_hist(self):
        coll = self.db.color_hist
        coll.insert({
            'vec': self.get_color_hist(),
            'kind': self.kind,
            'name': self.name
        })
        return self

    def store_ORB_list(self):
        coll = self.db.ORB_list
        ORB_list = self.get_ORB_list()
        try:
            for ORB in ORB_list:
                coll.insert({
                    'vec': [int(num) for num in ORB],
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


def test():
    clf_fore = train.train_sample('fore')
    clf_shape = train.train_sample('shape')
    obj = ObjectProcess('train_pic/cup/69.jpg', clf_fore, clf_shape)

    obj.store_color_hist()
    obj.store_ORB_list()
    obj.store_hog_list()


if __name__ == '__main__':
    test()