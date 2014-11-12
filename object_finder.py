import cv2
import train
from object_process import ObjectProcess
import numpy as np
from k_means_multi_layer import *
from pymongo import MongoClient
import threading
import copy
from PyQt4 import QtGui, QtCore
import sys


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
            print cv_image
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
        thread_fore = threading.Thread(target=self.train_fore())
        thread_shape = threading.Thread(target=self.train_shape())
        thread_fore.start()
        thread_shape.start()
        thread_fore.join()
        thread_shape.join()
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
        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.btm_read)
        self.layout.addWidget(self.btm_img)
        self.layout.addWidget(self.btm_fore)
        self.layout.addWidget(self.btm_seg)
        self.layout.addWidget(self.btm_shape)
        self.layout.addWidget(self.btm_ORB)
        self.setLayout(self.layout)

    def train_fore(self):
        self.clf_fore = train.train_sample('fore')

    def train_shape(self):
        self.clf_shape = train.train_sample('shape')

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

    def detect_kind(self):
        client = MongoClient()
        db = client.object_finder
        coll = db.find_result
        coll.insert({
            'name': self.object.name,
            'fit_color': self.object.get_fit_color_dict(),
            'best_fit_color': self.object.get_best_fit_color_dict(),
            'fit_ORB': self.object.get_fit_ORB_dict(),
            'best_fit_ORB': self.object.get_best_fit_ORB_dict(),
            'fit_hog': self.object.get_fit_hog_dict(),
            'best_fit_hog': self.object.get_best_fit_hog_dict()
        })

    def detect_name(self):
        pass


def main():
    app = QtGui.QApplication(sys.argv)
    object_finder = ObjectFinder()
    object_finder.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()