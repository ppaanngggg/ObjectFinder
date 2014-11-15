from object_process import ObjectProcess
from k_means_multi_layer import *
import threading
from path_process import PathProcess
import train


def build_k_means_color_hist():
    k_means_multi_layer(
        find_data('object_finder', 'color_hist'),
        'vec',
        8,
        'none',
        'object_finder',
        'k_means_color_hist',
        2
    )


def build_k_means_ORB_list():
    k_means_multi_layer(
        find_data('object_finder', 'ORB_list'),
        'vec',
        8,
        'none',
        'object_finder',
        'k_means_ORB_list',
        2
    )


def build_k_means_hog_list():
    k_means_multi_layer(
        find_data('object_finder', 'hog_list'),
        'vec',
        8,
        'none',
        'object_finder',
        'k_means_hog_list',
        2
    )


def store_by_kind(paths, kind, clf_fore, clf_shape):
    path_list = paths.get_file_path_list(kind)
    for path in path_list:
        print path.split('/')
        ObjectProcess(path, clf_fore, clf_shape) \
            .store_color_hist() \
            .store_ORB_list() \
            .store_hog_list() \
            .write_fore_image('train_pic_fore')


def main():
    paths = PathProcess('train_pic')
    kind_list = paths.get_kind_list()

    clf_fore = train.train_sample('fore')
    clf_shape = train.train_sample('shape')

    thread_list = [
        threading.Thread(target=store_by_kind, args=(paths, kind, clf_fore, clf_shape))
        for kind in kind_list
    ]

    for thread in thread_list:
        thread.start()

    for thread in thread_list:
        thread.join()

    t_color_hist = threading.Thread(target=build_k_means_color_hist)
    t_ORB = threading.Thread(target=build_k_means_ORB_list)
    t_hog = threading.Thread(target=build_k_means_hog_list)

    t_color_hist.start()
    # t_ORB.start()
    t_hog.start()

    t_color_hist.join()
    t_ORB.join()
    t_hog.join()


if __name__ == '__main__':
    main()