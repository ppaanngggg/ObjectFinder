from image_process import ImageProcess
import cv2
import copy
from pymongo import MongoClient
import train


class ObjectProcess:
    def __init__(self, path, clf_fore, clf_shape):
        str_list = path.split('/')
        self.kind = str_list[1]
        self.name = str_list[2]

        self.image = cv2.imread(path)
        self.image = cv2.resize(self.image,(400,400))

        self.image_proc_s = ImageProcess(self.image, 70)
        self.image_proc_s.set_classify_target_list(
            clf_fore.predict(
                self.image_proc_s.get_classify_vec_list()
            )
        )
        self.image_proc_s.compute_foreground_mask()
        self.image_proc_s.compute_foreground_image()
        self.image_proc_s.compute_color_hist()
        self.image_proc_s.compute_ORB_list()

        self.image_proc_m = ImageProcess(self.image, 100)
        self.image_proc_m.set_classify_target_list(
            clf_shape.predict(
                self.image_proc_m.get_classify_vec_list()
            )
        )

        self.image_proc_l = ImageProcess(self.image, 130)
        self.image_proc_l.set_classify_target_list(
            clf_shape.predict(
                self.image_proc_l.get_classify_vec_list()
            )
        )

        self.hog_list = []
        hog_list = \
            self.image_proc_m.get_hog_list() + self.image_proc_l.get_hog_list()
        target_list = \
            list(self.image_proc_m.get_classify_target_list()) + list(self.image_proc_l.get_classify_target_list())
        for index in range(len(hog_list)):
            if target_list[index]:
                self.hog_list.append(hog_list[index])

        self.client = MongoClient()
        self.db = self.client.object_finder

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
        ORB_list=self.get_ORB_list()
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
        hog_list=self.get_hog_list()
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

    def write_fore_image(self,img_base):
        cv2.imwrite(
            img_base+'/'+self.kind+'/'+self.name,
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