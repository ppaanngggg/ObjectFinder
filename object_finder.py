import cv2
import train
import arg_learn
from object_process import ObjectProcess
from k_means_multi_layer import *
from pymongo import MongoClient
import threading
import copy
from PyQt4 import QtGui, QtCore
import sys
import pickle
import numpy as np


class Note(QtGui.QWidget):
    def __init__(self, show_info):
        super(Note, self).__init__()
        self.setWindowTitle('ObjectFinder')
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
        )
        self.layout = QtGui.QVBoxLayout()
        self.label = QtGui.QLabel(show_info)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)


class ImageView(QtGui.QDialog):
    def __init__(self, parent, title, image_list, num):
        super(ImageView, self).__init__(parent)
        self.setWindowTitle(title)
        self.cv_image_list = copy.deepcopy(image_list)
        self.layout = QtGui.QGridLayout()
        row = 0
        col = 0

        for cv_image in self.cv_image_list:
            if len(cv_image.shape) < 3:
                if cv_image.dtype != np.uint8:
                    cv_image = np.array(cv_image * 255, dtype=np.uint8)
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            height, width, bytes_per_component = cv_image.shape
            bytes_per_line = bytes_per_component * width
            qt_image = QtGui.QImage(
                cv_image.data,
                width, height, bytes_per_line,
                QtGui.QImage.Format_RGB888
            )
            label = QtGui.QLabel()
            label.setPixmap(QtGui.QPixmap.fromImage(qt_image))
            self.layout.addWidget(label, row, col)
            col += 1
            if col == num:
                col = 0
                row += 1
        self.setLayout(self.layout)
        # self.show()


class ObjectFinder(QtGui.QWidget):
    def __init__(self):
        super(ObjectFinder, self).__init__()
        self.setWindowTitle('ObjectFinder')
        self.setMinimumWidth(200)
        self.setMinimumHeight(300)

        note = Note('training...')
        note.show()

        # thread_fore = threading.Thread(target=self.train_fore)
        # thread_shape = threading.Thread(target=self.train_shape)
        # thread_bpnn = threading.Thread(target=self.train_bpnn)
        # thread_fore.start()
        # thread_shape.start()
        # thread_bpnn.start()
        # thread_fore.join()
        # thread_shape.join()
        # thread_bpnn.join()

        self.read_clf_fore()
        self.read_clf_shape()
        self.read_bpnn()

        note.close()

        self.btm_read = QtGui.QPushButton('Read')
        self.btm_read.clicked.connect(self.btm_read_clicked)
        self.btm_img = QtGui.QPushButton('Image')
        self.btm_img.setEnabled(False)
        self.btm_img.clicked.connect(self.btm_img_clicked)
        self.btm_fore = QtGui.QPushButton('Fore')
        self.btm_fore.setEnabled(False)
        self.btm_fore.clicked.connect(self.btm_fore_clicked)
        self.btm_seg = QtGui.QPushButton('Segmentation')
        self.btm_seg.setEnabled(False)
        self.btm_seg.clicked.connect(self.btm_seg_clicked)
        self.btm_shape = QtGui.QPushButton('Shape')
        self.btm_shape.setEnabled(False)
        self.btm_shape.clicked.connect(self.btm_shape_clicked)
        self.btm_ORB = QtGui.QPushButton('ORB')
        self.btm_ORB.setEnabled(False)
        self.btm_ORB.clicked.connect(self.btm_ORB_clicked)
        self.btm_result = QtGui.QPushButton('result')
        self.btm_result.setEnabled(False)
        self.btm_result.clicked.connect(self.btm_result_clicked)
        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.btm_read)
        self.layout.addWidget(self.btm_img)
        self.layout.addWidget(self.btm_fore)
        self.layout.addWidget(self.btm_seg)
        self.layout.addWidget(self.btm_shape)
        self.layout.addWidget(self.btm_ORB)
        self.layout.addWidget(self.btm_result)
        self.setLayout(self.layout)

    def train_fore(self):
        self.clf_fore = train.train_sample('fore')
        f = open('cache/clf_fore', 'w')
        pickle.dump(self.clf_fore, f)
        f.close()

    def read_clf_fore(self):
        f = open('cache/clf_fore', 'r')
        self.clf_fore = pickle.load(f)
        f.close()

    def train_shape(self):
        self.clf_shape = train.train_sample('shape')
        f = open('cache/clf_shape', 'w')
        pickle.dump(self.clf_shape, f)
        f.close()

    def read_clf_shape(self):
        f = open('cache/clf_shape', 'r')
        self.clf_shape = pickle.load(f)
        f.close()

    def train_bpnn(self):
        self.bpnn = arg_learn.train_arg()
        f = open('cache/bpnn', 'w')
        pickle.dump(self.bpnn, f)
        f.close()

    def read_bpnn(self):
        f = open('cache/bpnn', 'r')
        self.bpnn = pickle.load(f)
        f.close()

    def btm_read_clicked(self):
        file_dialog = QtGui.QFileDialog()
        path = file_dialog.getOpenFileName()
        print path
        if path:
            note = Note('processing...')
            note.show()
            self.object = ObjectProcess(str(path), self.clf_fore, self.clf_shape)
            self.detect_kind()
            self.detect_name()
            note.close()
            self.btm_img.setEnabled(True)
            self.btm_fore.setEnabled(True)
            self.btm_seg.setEnabled(True)
            self.btm_shape.setEnabled(True)
            self.btm_ORB.setEnabled(True)
            self.btm_result.setEnabled(True)

    def btm_img_clicked(self):
        image_view = ImageView(
            self, 'image',
            [self.object.get_image()],
            1
        )
        image_view.show()

    def btm_fore_clicked(self):
        image_view = ImageView(
            self, 'fore',
            [self.object.get_fore()],
            1
        )
        image_view.show()

    def btm_seg_clicked(self):
        image_view = ImageView(
            self, 'seg',
            self.object.get_seg_image_list(),
            3
        )
        image_view.show()

    def btm_shape_clicked(self):
        image_view = ImageView(
            self, 'shape',
            self.object.get_pos_hog_image_list(),
            10
        )
        image_view.show()

    def btm_ORB_clicked(self):
        image_view = ImageView(
            self, 'ORB',
            [self.object.image_proc_s.ORB_image],
            1
        )
        image_view.show()

    def btm_result_clicked(self):
        image_view = ImageView(
            self, 'result',
            self.find_result_img_list,
            3
        )
        image_view.show()

    def detect_kind(self):
        self.obj_dict = {
            'color': self.object.get_fit_color_dict(),
            'best_color': self.object.get_best_fit_color_dict(),
            'ORB': self.object.get_fit_ORB_dict(),
            'best_ORB': self.object.get_best_fit_ORB_dict(),
            'hog': self.object.get_fit_hog_dict(),
            'best_hog': self.object.get_best_fit_hog_dict()
        }
        print self.obj_dict['color']
        print self.obj_dict['best_color']
        vec = arg_learn.to_vec(self.obj_dict)
        self.bpnn.compute(vec)
        output = self.bpnn.output()
        print output
        kind_table = ['cloth', 'cup', 'shore']
        k_max = output[0]
        index_max = 0
        for index in range(1, len(output)):
            if output[index] > k_max:
                k_max = output[index]
                index_max = index
        self.kind = kind_table[index_max]
        print self.kind


    def detect_name(self):
        name_dict = {}
        weight_table = {
            'cloth':
                {'color': 3, 'best_color': 2, 'ORB': 0.5, 'best_ORB': 0.5, 'hog': 1, 'best_hog': 1},
            'cup':
                {'color': 2, 'best_color': 2, 'ORB': 1, 'best_ORB': 1, 'hog': 1, 'best_hog': 1},
            'shore':
                {'color': 4, 'best_color': 3, 'ORB': 0.5, 'best_ORB': 0.5, 'hog': 3, 'best_hog': 2}
        }
        for t in ['color', 'best_color','ORB','best_ORB','hog','best_hog']:
            try:
                weight = weight_table[self.kind][t]
                for key, value in self.obj_dict[t][self.kind].items():
                    # print key,value
                    if key in name_dict.keys():
                        name_dict[key] += weight * value
                    else:
                        name_dict[key] = weight * value
            except:
                pass
        import operator

        sorted_name = sorted(name_dict.items(), key=operator.itemgetter(1), reverse=True)
        print sorted_name
        img_list = []
        if len(sorted_name) > 6:
            for i in range(6):
                img = cv2.imread(
                    'train_pic/' + str(self.kind) + '/' + str(sorted_name[i][0]) + '.jpg'
                )
                img_list.append(img)
        else:
            for name in sorted_name:
                img = cv2.imread(
                    'train_pic/' + str(self.kind) + '/' + str(name[0]) + '.jpg'
                )
                img_list.append(img)
        self.find_result_img_list = img_list


def main():
    app = QtGui.QApplication(sys.argv)
    object_finder = ObjectFinder()
    object_finder.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()