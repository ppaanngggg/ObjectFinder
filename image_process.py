import cv2
import numpy as np
import skimage.segmentation as skseg
import hog
import copy
from cv2.xfeatures2d import *


class ImageProcess:
    def __init__(self, image, super_pixel_size, compactness=80,
                 hog_cell_size=8, hog_block_size=2, hog_row_num=4, hog_col_num=4, hog_angle_bin=9):
        # copy paras into class
        self.image = np.copy(image)
        # cv2.imshow('img',self.image)
        # cv2.waitKey()
        self.super_pixel_size = super_pixel_size
        self.compactness = compactness
        self.hog_cell_size = hog_cell_size
        self.hog_block_size = hog_block_size
        self.hog_row_num = hog_row_num
        self.hog_col_num = hog_col_num
        self.hog_angle_bin = hog_angle_bin

        # compute info
        self.compute_segments()
        self.compute_point_list_list()
        # self.compute_segment_image_list()
        self.compute_rect_list()
        self.compute_hog_list()
        self.compute_boundary_vec_list()
        self.compute_segment_vec_list()
        self.compute_classify_vec_list()
        # print len(self.point_list_list)
        # print len(self.rect_list)
        # print len(self.hog_list)
        # print len(self.boundary_vec_list)
        # print len(self.classify_vec_list)


    def get_image(self):
        return copy.deepcopy(self.image)

    def show_image(self):
        cv2.imshow('image', self.image)
        cv2.waitKey()

    def compute_segments(self):
        tmp_b, tmp_g, tmp_r = cv2.split(self.image)
        tmp_b = cv2.equalizeHist(tmp_b)
        tmp_g = cv2.equalizeHist(tmp_g)
        tmp_r = cv2.equalizeHist(tmp_r)
        tmp_image = cv2.merge([tmp_b, tmp_g, tmp_r])
        # compute super pixel segments
        self.segments = skseg.slic(
            tmp_image,
            self.image.shape[0] * self.image.shape[1] / self.super_pixel_size / self.super_pixel_size,
            self.compactness,
            enforce_connectivity=True
        )

    def get_segments(self):
        return copy.deepcopy(self.segments)

    def show_segments(self):
        cv2.imshow('segments', self.segments)
        cv2.waitKey()

    def compute_point_list_list(self):
        # depart segments into different point list
        segment_value_set = set()
        for row in self.segments:
            for pixel in row:
                segment_value_set.add(pixel)
        self.point_list_list = [[] for i in range(len(segment_value_set))]
        for row in range(self.segments.shape[0]):
            for col in range(self.segments.shape[1]):
                self.point_list_list[self.segments[row, col]].append((col, row))
                # self.tmp_point_list_list = []
                # for point_list in self.point_list_list:
                # if len(point_list) < self.super_pixel_size ** 2 * 0.5:
                # continue
                # self.tmp_point_list_list.append(point_list)
                # self.point_list_list = self.tmp_point_list_list

    def get_point_list_list(self):
        return copy.deepcopy(self.point_list_list)

    def compute_segment_image_list(self):
        self.segment_image_list = [np.zeros(self.segments.shape, np.uint8) for i in range(len(self.point_list_list))]
        for i in range(len(self.point_list_list)):
            for point in self.point_list_list[i]:
                self.segment_image_list[i][point[1], point[0]] = 255
                # for image in self.segment_image_list:
                # cv2.imshow('segment_image',image)
                # cv2.waitKey()

    def compute_rect_list(self):
        # get bounding rect of each segment
        self.rect_list = []
        for point_list in self.point_list_list:
            # if len(point_list) < self.super_pixel_size ** 2 * 0.5:
            # continue
            x = min([point[0] for point in point_list])
            y = min([point[1] for point in point_list])
            width = max([point[0] for point in point_list]) - x
            height = max([point[1] for point in point_list]) - y
            # if width*height<super_pixel_size**2:
            # continue
            self.rect_list.append((x, y, width, height))
            # for rect in self.rect_list:
            # print rect

    def get_rect_list(self):
        return copy.deepcopy(self.rect_list)

    def show_rect_image(self, mark=False):
        if mark:
            tmp_image = self.get_mark_image()
        else:
            tmp_image = self.get_image()
        for rect in self.rect_list:
            rect_img = np.copy(tmp_image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]])
            cv2.imshow('rect_image', rect_img)
            cv2.waitKey()

    def compute_hog_list(self):
        # compute hog of each segment
        self.hog_list = []
        self.hog_image_list = []
        for rect in self.rect_list:
            hog_img = np.copy(self.image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]])
            vec, image = hog.hog(
                hog_img, self.hog_cell_size, self.hog_block_size,
                self.hog_row_num, self.hog_col_num, self.hog_angle_bin
            )
            self.hog_list.append(vec)
            self.hog_image_list.append(image)

    def get_hog_list(self):
        return copy.deepcopy(self.hog_list)

    def get_hog_image_list(self):
        return copy.deepcopy(self.hog_image_list)

    def get_mark_image(self):
        try:
            return copy.deepcopy(self.mark_image)
        except:
            self.mark_image = skseg.mark_boundaries(self.image, self.segments)
            return copy.deepcopy(self.mark_image)

    def show_mark_image(self, wait=True):
        cv2.imshow('mark_image', self.get_mark_image())
        if wait:
            cv2.waitKey()

    def compute_boundary_vec_list(self):
        self.boundary_vec_list = []
        for rect in self.rect_list:
            boundary_vec = [0, 0, 0, 0]
            if rect[0] <= 1:
                boundary_vec[0] = 1
            if rect[1] <= 1:
                boundary_vec[1] = 1
            if rect[0] + rect[2] >= self.segments.shape[1] - 2:
                boundary_vec[2] = 1
            if rect[1] + rect[3] >= self.segments.shape[0] - 2:
                boundary_vec[3] = 1
            # print boundary_vec
            self.boundary_vec_list.append(boundary_vec)

    def get_boundary_vec_list(self):
        return copy.deepcopy(self.boundary_vec_list)

    def compute_segment_vec_list(self):
        self.segment_vec_list = []
        if len(self.rect_list) != len(self.point_list_list):
            raise ValueError('len(self.rect_list) != len(self.point_list_list) !')
        for i in range(len(self.rect_list)):
            segment_vec = np.zeros((4, 4), np.float64)
            for point in self.point_list_list[i]:
                col = int((point[0] - self.rect_list[i][0]) / (self.rect_list[i][2] / 4 + 1))
                row = int((point[1] - self.rect_list[i][1]) / (self.rect_list[i][3] / 4 + 1))
                segment_vec[row, col] += 1
            segment_vec /= len(self.point_list_list[i])
            segment_vec = segment_vec.reshape((16))
            # cv2.imshow('segment',self.segment_image_list[i])
            # print segment_vec
            # cv2.waitKey()
            # print list(segment_vec)
            self.segment_vec_list.append(list(segment_vec))

    def get_segment_vec_list(self):
        return copy.deepcopy(self.segment_vec_list)

    def compute_classify_vec_list(self):
        if not len(self.hog_list) == len(self.boundary_vec_list) == len(self.segment_vec_list):
            raise ValueError('not len(self.hog_list) == len(self.boundary_vec_list) == len(self.segment_vec_list) !')
        self.classify_vec_list = []
        for i in range(len(self.hog_list)):
            self.classify_vec_list.append(
                list(self.rect_list[i]) + self.boundary_vec_list[i] + self.segment_vec_list[i] + self.hog_list[i]
            )
            # for vec in self.classify_vec_list:
            # print vec

    def get_classify_vec_list(self):
        return copy.deepcopy(self.classify_vec_list)

    def set_classify_target_list(self, target_list):
        self.classify_target_list = target_list

    def label_classify_target_list(self, mark=False):
        self.classify_target_list = []
        for i in range(len(self.rect_list)):
            if mark:
                tmp_image = np.copy(self.get_mark_image())
            else:
                tmp_image = np.copy(self.get_image())
            for point in self.point_list_list[i]:
                tmp_image[point[1], point[0]] = tmp_image[point[1], point[0]] * 0.95 + np.array([0, 0, 255]) * 0.05
            print self.rect_list[i]
            cv2.rectangle(
                tmp_image,
                (self.rect_list[i][0], self.rect_list[i][1]),
                (
                    self.rect_list[i][0] + self.rect_list[i][2],
                    self.rect_list[i][1] + self.rect_list[i][3]
                ),
                (0, 255, 0)
            )
            cv2.namedWindow('rect_image')
            cv2.moveWindow('rect_image', 50, 70)
            cv2.imshow('rect_image', tmp_image)
            cv2.namedWindow('hog')
            cv2.moveWindow('hog', 50, 0)
            cv2.imshow('hog', self.hog_image_list[i])
            while 1:
                try:
                    target = int(raw_input())
                    break
                except:
                    pass
            self.classify_target_list.append(target)
            cv2.destroyAllWindows()

    def get_classify_target_list(self):
        return copy.deepcopy(self.classify_target_list)

    def compute_foreground_mask(self):
        self.fore_mask = np.zeros(self.image.shape[0:2], np.uint8)
        if len(self.classify_target_list) != len(self.point_list_list):
            raise ValueError('len(self.classify_target_list)!=len(self.point_list_list) !')
        for i in range(len(self.classify_target_list)):
            if self.classify_target_list[i]:
                for point in self.point_list_list[i]:
                    self.fore_mask[point[1], point[0]] = 1

    def get_foreground_mask(self):
        return copy.deepcopy(self.fore_mask)

    def compute_foreground_image(self):
        self.fore_image = np.zeros(self.image.shape, np.uint8)
        for row in range(self.fore_mask.shape[0]):
            for col in range(self.fore_mask.shape[1]):
                if self.fore_mask[row, col]:
                    self.fore_image[row, col] = self.image[row, col]
                    # self.fore_image = cv2.copyTo(self.image , self.fore_mask)

    def get_foreground_image(self):
        return copy.deepcopy(self.fore_image)

    def compute_color_list(self):
        # img=self.get_image()
        # cv2.imshow('img',img)
        # cv2.waitKey()
        lab = cv2.cvtColor(self.get_image(), cv2.COLOR_BGR2Lab)
        segment_value_set = set()
        for row in self.segments:
            for pixel in row:
                segment_value_set.add(pixel)
        pixel_list_list = [[] for i in range(len(segment_value_set))]
        for row in range(self.segments.shape[0]):
            for col in range(self.segments.shape[1]):
                pixel_list_list[self.segments[row, col]].append(lab[row, col])
        # print pixel_list_list[0]
        # print pixel_list_list[1]
        pixel_array_list = []
        target_list = self.get_classify_target_list()
        for i in range(len(target_list)):
            if target_list[i]:
                pixel_array_list.append(np.array(pixel_list_list[i]))
        self.color_list = []
        for pixel_array in pixel_array_list:
            # print pixel_array
            l = pixel_array[:, 0]
            a = pixel_array[:, 1]
            b = pixel_array[:, 2]
            color = []
            for item in [l, a, b]:
                # print item
                from scipy.stats import skew

                mean = np.mean(item)
                # print mean
                var = np.sqrt(np.var(item))
                # print var
                s = skew(item)
                # print s
                color.append(mean)
                color.append(var)
                color.append(s)
            self.color_list.append(color)


    def get_color_list(self):
        return copy.deepcopy(self.color_list)

    def compute_sift_list(self):
        sift = SIFT_create(nfeatures=100)
        kp_list, self.sift_list = sift.detectAndCompute(self.get_image(), self.get_foreground_mask())
        self.sift_image = cv2.drawKeypoints(self.get_foreground_image(), kp_list, None,
                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow('tmp', self.sift_image)
        # cv2.waitKey()

    def get_sift_list(self):
        return copy.deepcopy(self.sift_list)


def test():
    img = cv2.imread('train_pic/cloth/0.jpg')
    img_proc = ImageProcess(
        cv2.medianBlur(img, 5), 70
    )
    # img_proc.label_classify_target_list(True)
    # import pickle
    #
    # f = open('cache/clf_fore', 'r')
    # clf_fore = pickle.load(f)
    # f.close()
    # img_proc.set_classify_target_list(
    #     clf_fore.predict(
    #         img_proc.get_classify_vec_list()
    #     )
    # )
    # img_proc.image = cv2.bilateralFilter(img, 5, 50, 50)
    # img_proc.compute_foreground_mask()
    # img_proc.compute_foreground_image()
    # img_proc.compute_sift_list()


if __name__ == '__main__':
    test()