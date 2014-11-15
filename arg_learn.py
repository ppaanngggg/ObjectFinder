from path_process import PathProcess
from object_process import ObjectProcess
import train

def main():
    clf_fore = train.train_sample('fore')
    clf_shape = train.train_sample('shape')

    path_proc = PathProcess('train_pic_arg')
    for kind in path_proc.get_kind_list():
        for path in path_proc.get_file_path_list(kind):
            print path
            obj_proc=ObjectProcess(path,clf_fore,clf_shape)
            print obj_proc.get_fit_color_dict()
            print obj_proc.get_best_fit_color_dict()
            # obj_proc.get_fit_ORB_dict()
            # obj_proc.get_best_fit_ORB_dict()
            # obj_proc.get_fit_hog_dict()
            # obj_proc.get_best_fit_hog_dict()


if __name__ == '__main__':
    main()