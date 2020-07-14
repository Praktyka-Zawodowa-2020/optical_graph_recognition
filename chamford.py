import cv2 as cv
import numpy as np

import math


def array_scope_control(img: np.ndarray, i: int, j: int) -> bool:
    """
    check if (i,j) is in the table
    :param img: input image
    :param i:
    :param j:
    :return:
    """
    width, height = img.shape[:2]
    if 0 <= i < width and 0 <= j < height:
        return True
    else:
        return False


def pixel_value(img: np.ndarray, i: int, j: int, direction: int) -> int:
    """
    Calculate the pixel(i,j) value for the chamford function
    Forward: use    [*, *, *] mask
                    [*, 0, -]
                    [-, -, -]

    Backward: use   [-, -, -] mask
                    [-, 0, *]
                    [*, *, *]

    where '*' value is uses, '0' is center of mask (the center is located in pixel(i,j)), '-' value is non used

    :param img: input image
    :param i:
    :param j:
    :param direction: determines the phase forward or backward
    :return: minimum pixel distance from the background
    """
    mask = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]  # determines the pixel distance from the center
    maximum = []
    maximum.append(0)
    if direction == 1:  # forward
        i -= 1
        j -= 1
        for x in range(0, 3):
            for y in range(0, 3):
                if x == 1 and y == 1:
                    return max(maximum)
                if array_scope_control(img, i + x, j + y):
                    if (img[i + x][j + y] - mask[x][y]) == -1:
                        maximum.append(255)
                    else:
                        maximum.append(img[i + x][j + y] - mask[x][y])
    else:  # backward
        i += 1
        j += 1
        for x in range(0, 3):
            for y in range(0, 3):
                if x == 1 and y == 1:
                    return max(maximum)
                if array_scope_control(img, i - x, j - y):
                    if (img[i - x][j - y] - mask[x][y]) == -1:
                        maximum.append(255)
                    else:
                        maximum.append(img[i - x][j - y] - mask[x][y])


def chamford(img: np.ndarray) -> np.ndarray:
    """
    Calculate the distance of the pixel from the background

    Chamford algorithm have two steps:
    Forward: goes right and down
    Backward: goes left up

    :param img: binary image. 0 pixel is background
    :return: image with the distance of the image pixels from the background
    """
    width, height = img.shape[:2]
    size = 3  # size mask in pixel_value
    chamford_image = np.zeros((width, height), np.uint8)
    # forward
    for i in range(math.floor((size + 1) / 2), width):
        for j in range(math.floor((size + 1) / 2), height):
            if img[i][j] > 0:
                chamford_image[i][j] = pixel_value(chamford_image, i, j, 1)
            else:
                chamford_image[i][j] = img[i][j]
    # backward
    for i in range(math.floor(width - (size - 1) / 2), 0, -1):
        for j in range(math.floor(height - (size - 1) / 2), 0, -1):
            if img[i][j] > 0:
                val = pixel_value(chamford_image, i, j, 2)
                if val > chamford_image[i][j]:
                    chamford_image[i][j] = val

    return chamford_image


def spr_local_extreme(img: np.ndarray, i: int, j: int, val: int) -> bool:
    """
    Determine whether pixel (i, j) is a local extreme

    :param img: Input image
    :param i:
    :param j:
    :param val: pixel distance from the background
    :return: True if pixel[i][j] is local extreme, otherwaise False
    """
    i -= 1
    j -= 1
    for x in range(0, 3):
        for y in range(0, 3):
            if x == 1 and y == 1:
                continue
            if array_scope_control(img, i - x, j - y):
                if img[i - x][j - y] > val and img[i - x][j - y] != 0:  # if the pixel is 255, its a background
                    return False
    return True


def extreme(img: np.ndarray) -> (int, int):
    """
    Image must be after chamford function, background pixel value must be 0
    Finds min and max object extremum in image
    Object extremum is the furthest pixel distance from the background


    :param img: Image
    :return: minimum,maximum
    """
    width, height = img.shape[:2]
    max_extreme = 255
    # find max extreme value
    for i in range(0, width):
        for j in range(0, height):
            if max_extreme > img[i][j] > 0:
                max_extreme = img[i][j]

    # specifies the maximum edge thickness(in the image, the max edge thickness is (255-edge_thickness) * 2 pixels)
    if max_extreme < 245:
        edge_thickness = 245
    else:
        edge_thickness = max_extreme

    # find minimum extreme value
    for val in range(254, edge_thickness, -1):
        for i in range(0, width):
            for j in range(0, height):
                if img[i][j] == val:
                    if spr_local_extreme(img, i, j, val):
                        return 255-val, 255-max_extreme

    return 255-245, 255-max_extreme