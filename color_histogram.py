import segmentation
import train
import os
import cv2
from pymongo import MongoClient
import numpy as np
import k_means_multi_layer


def store_color_histogram():
    files = os.listdir('train_pic')
    files = [filename for filename in files if filename[-4:] == '.jpg']
    clf = train.train_sample('fore')

    client = MongoClient()
    db = client.object_finder
    collection = db.color_hist

    total = 0

    for filename in files:
        print filename
        image = cv2.imread('train_pic/' + filename)
        seg = segmentation.segmentation(image, 70)
        seg.set_classify_target_list(clf.predict(seg.get_classify_vec_list()))
        seg.compute_foreground_mask()
        seg.compute_foreground_image()
        cv2.imwrite('train_pic_fore/' + filename, seg.get_foreground_image())
        img_list = cv2.split(seg.get_foreground_image())
        hist_list = [
            cv2.calcHist([img], [0], seg.get_foreground_mask(), [16], [0, 256])
            for img in img_list
        ]
        hist = []
        for h in hist_list:
            h /= np.sum(h)
            hist += [float(num) for num in list(h)]
        # print hist
        # cv2.imshow('img',seg.get_foreground_image())
        # cv2.waitKey()
        collection.insert(
            {
                'hist': hist,
                'file': filename
            }
        )
        total += 1

    print '*** sum *** :', total


def k_means_color_histogram():
    k_means_multi_layer.k_means_multi_layer(
        k_means_multi_layer.find_data('object_finder', 'color_hist'),
        'hist',
        8,
        'none',
        'object_finder',
        'k_means_color_hist',
        2
    )


if __name__ == '__main__':
    store_color_histogram()
    k_means_color_histogram()