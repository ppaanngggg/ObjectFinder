import cv2
import numpy as np
import skimage.segmentation as skseg
import hog
import copy


class segmentation:
    def __init__(self, image, super_pixel_size=50, compactness=40,
                 hog_cell_size=8, hog_block_size=2, hog_row_num=4, hog_col_num=4, hog_angle_bin=9):
        # copy paras into class
        self.image = np.copy(image)
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
        self.compute_rect_list()
        self.compute_hog_list()
        self.compute_boundary_vec_list()
        self.compute_classify_vec_list()


    def get_image(self):
        return copy.deepcopy(self.image)

    def show_image(self):
        cv2.imshow('image', self.image)
        cv2.waitKey()

    def compute_segments(self):
        # compute super pixel segments
        self.segments = skseg.slic(
            self.image,
            self.image.shape[0] * self.image.shape[1] / self.super_pixel_size / self.super_pixel_size,
            self.compactness
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

    def get_point_list_list(self):
        return copy.deepcopy(self.point_list_list)

    def compute_rect_list(self):
        # get bounding rect of each segment
        self.rect_list = []
        for point_list in self.point_list_list:
            if len(point_list) < self.super_pixel_size ** 2 * 0.6:
                continue
            x = min([point[0] for point in point_list])
            y = min([point[1] for point in point_list])
            width = max([point[0] for point in point_list]) - x
            height = max([point[1] for point in point_list]) - y
            # if width*height<super_pixel_size**2:
            # continue
            self.rect_list.append((x, y, width, height))

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
        for rect in self.rect_list:
            hog_img = np.copy(self.image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]])
            # cv2.imshow('hog_image',hog_img)
            # cv2.waitKey()
            self.hog_list.append(
                hog.hog(hog_img, self.hog_cell_size, self.hog_block_size,
                        self.hog_row_num, self.hog_col_num, self.hog_angle_bin)
            )

    def get_hog_list(self):
        return copy.deepcopy(self.hog_list)

    def get_mark_image(self):
        try:
            return copy.deepcopy(self.mark_image)
        except:
            self.mark_image = skseg.mark_boundaries(self.image, self.segments)
            return copy.deepcopy(self.mark_image)

    def show_mark_image(self):
        cv2.imshow('mark_image', self.get_mark_image())
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
            self.boundary_vec_list.append(boundary_vec)

    def get_boundary_vec_list(self):
        return copy.deepcopy(self.boundary_vec_list)

    def compute_classify_vec_list(self):
        if len(self.hog_list) != len(self.boundary_vec_list):
            raise ValueError('len(self.hog_list)!=len(self.boundary_vec_list) !')
        self.classify_vec_list = []
        for i in range(len(self.hog_list)):
            self.classify_vec_list.append(
                self.boundary_vec_list[i] + self.hog_list[i]
            )

    def get_classify_vec_list(self):
        return copy.deepcopy(self.classify_vec_list)

    def compute_classify_target_list(self, classifier):
        pass

    def label_classify_target_list(self, mark=False):
        self.classify_target_list = []
        if mark:
            tmp_image = self.get_mark_image()
        else:
            tmp_image = self.get_image()
        for rect in self.rect_list:
            rect_img = np.copy(tmp_image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]])
            cv2.imshow('rect_image', rect_img)
            while 1:
                try:
                    target = int(raw_input())
                    break
                except:
                    pass
            self.classify_target_list.append(target)

    def get_classify_target_list(self):
        return copy.deepcopy(self.classify_target_list)


        # def test():
        # img = cv2.imread('train_pic/img.jpg')
        #     segments = skseg.slic(img, img.shape[0] * img.shape[1] / 50 / 50,40 )
        #     mark_img=skseg.mark_boundaries(img, segments)
        #     cv2.imshow('mark_img',mark_img)
        #     cv2.waitKey()
        #     value_set = set()
        #     for row in segments:
        #         for pixel in row:
        #             value_set.add(pixel)
        #     segment_list = [np.zeros((img.shape[0], img.shape[1]), np.uint8) for i in range(len(value_set))]
        #     for row in range(segments.shape[0]):
        #         for col in range(segments.shape[1]):
        #             segment_list[segments[row, col]][row, col] = 255
        #
        #     for segment in segment_list:
        #         contour_list = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #         contour_max = \
        #             [contour for contour in contour_list[1] if contour.shape[0] == max([tmp.shape[0] for tmp in contour_list[1]])][
        #                 0]
        #         # tmp_img=np.zeros((img.shape[0], img.shape[1]), np.uint8)
        #         # for point in contour_max:
        #         # loc=point[0]
        #         #     tmp_img[loc[1],loc[0]]=255
        #         rect = cv2.boundingRect(contour_max)
        #         if rect[2]*rect[3]<50*50:
        #             continue
        #         boundary_vec=[0,0,0,0]
        #         if rect[0]<=1:
        #             boundary_vec[0]=1
        #         if rect[1]<=1:
        #             boundary_vec[1]=1
        #         if rect[0]+rect[2]>=img.shape[1]-2:
        #             boundary_vec[2]=1
        #         if rect[1]+rect[3]>=img.shape[0]-2:
        #             boundary_vec[3]=1
        #         print boundary_vec
        #         hog_img = np.copy(img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]])
        #         show_img=np.copy(mark_img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]])
        #         cv2.imshow('hog_img', show_img)
        #         # cv2.waitKey()
        #         hog.hog(hog_img, 8, 2, 4, 4, 9)
        #         target=raw_input()