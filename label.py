import cv2
from image_process import ImageProcess
from pymongo import MongoClient
import copy

class Label:
    def __init__(self, path, mode):
        str_list = path.split('/')
        self.kind = str_list[1]
        self.name = str_list[2]
        self.mode = copy.deepcopy(mode)

        self.image = cv2.imread(path)

        self.image_proc_s = ImageProcess(self.image, 70)
        self.image_proc_m = ImageProcess(self.image, 100)
        self.image_proc_l = ImageProcess(self.image, 130)

        self.client = MongoClient()
        self.db = self.client.object_finder

    def label_fore(self):
        self.image_proc_s.label_classify_target_list(True)
        coll = self.db[self.mode+'_fore']
        vec_list = self.image_proc_s.get_classify_vec_list()
        target_list = self.image_proc_s.get_classify_target_list()
        for index in range(len(vec_list)):
            coll.insert({
                'vec': vec_list[index],
                'target': target_list[index],
                'kind': self.kind,
                'name': self.name
            })

    def label_shape(self):
        self.image_proc_m.label_classify_target_list(True)
        self.image_proc_l.label_classify_target_list(True)
        coll = self.db[self.mode+'_shape']
        vec_list = \
            self.image_proc_m.get_classify_vec_list() + self.image_proc_l.get_classify_vec_list()
        target_list = \
            self.image_proc_m.get_classify_target_list() + self.image_proc_l.get_classify_target_list()
        for index in range(len(vec_list)):
            coll.insert({
                'vec': vec_list[index],
                'target': target_list[index],
                'kind': self.kind,
                'name': self.name
            })

def label_target():
    from path_process import PathProcess
    paths=PathProcess('train_pic')
    kind_list=paths.get_kind_list()

    train_list=[]
    for kind in kind_list:
        path_list=paths.get_file_path_list(kind)
        for index in range(0,len(path_list),20):
            l=Label(path_list[index],'train')
            print l.kind,l.name
            train_list.append(l)
            l.label_fore()

    for train in train_list:
        print train.kind,train.name
        train.label_shape()

    test_list=[]
    for kind in kind_list:
        path_list=paths.get_file_path_list(kind)
        for index in range(1,len(path_list),50):
            l=Label(path_list[index],'test')
            print l.kind,l.name
            test_list.append(l)
            l.label_fore()

    for test in test_list:
        print test.kind,test.name
        test.label_shape()

def test():
    l=Label('','test')
    l.label_fore()

if __name__=='__main__':
    label_target()