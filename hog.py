import cv2
import numpy as np


def hog(image, cell_size, block_size, row_num, col_num, angle_bin):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.medianBlur(image, 3)
    image = cv2.equalizeHist(image)
    # cv2.imshow('image',image)
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    grad_x = cv2.resize(grad_x, (col_num * cell_size, row_num * cell_size))
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    grad_y = cv2.resize(grad_y, (col_num * cell_size, row_num * cell_size))
    norm_vec = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # cv2.imshow('norm_vec',norm_vec/np.max(norm_vec))
    # cv2.waitKey()
    norm_angle = np.arctan(grad_y / (grad_x + 10 ** -10))
    for row in range(norm_angle.shape[0]):
        for col in range(norm_angle.shape[1]):
            if norm_angle[row, col] < 0:
                norm_angle[row, col] += np.pi
    norm_angle = np.uint16(norm_angle / (np.pi / angle_bin))
    cell_table = np.zeros((row_num, col_num, angle_bin))
    for row in range(row_num):
        for col in range(col_num):
            for cell_row in range(row * cell_size, (row + 1) * cell_size):
                for cell_col in range(col * cell_size, (col + 1) * cell_size):
                    cell_table[row, col, norm_angle[cell_row, cell_col]] += norm_vec[cell_row, cell_col]
            cell_table[row, col, 0] = cell_table[row, col, 8] = (cell_table[row, col, 0] + cell_table[row, col, 8]) / 2
    # print cell_table
    # cv2.imshow('image',norm_vec/np.max(norm_vec))
    # cv2.waitKey()
    hog_vec = []
    for row in range(row_num - block_size + 1):
        for col in range(col_num - block_size + 1):
            block_vec = []
            for block_row in range(row, row + block_size):
                for block_col in range(col, col + block_size):
                    block_vec.append(cell_table[block_row, block_col])
            block_vec = np.hstack(block_vec)
            block_vec = block_vec / np.sqrt(np.sum(block_vec ** 2) + 10 ** -10)
            hog_vec.append(block_vec)
    hog_vec = list(np.hstack(hog_vec))
    return hog_vec