import numpy as np


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


def extreme(img: np.ndarray) -> (int, int, int):
    """
    Image must be after chamford function, background pixel value must be 0
    Finds min and max object extremum in image
    Object extremum is the furthest pixel distance from the background

    :param img: Image
    :return: minimum,sr_extreme, maximum
    """
    width, height = img.shape[:2]
    extremes = []
    max_extreme = 0
    # find max extreme value
    for i in range(0, width):
        for j in range(0, height):
            if max_extreme < img[i][j] > 0:
                max_extreme = int(img[i][j])

    # specifies the maximum edge thickness(in the image, the max edge thickness is (255-edge_thickness) * 2 pixels)
    if max_extreme > 15:
        edge_thickness = 15
    else:
        edge_thickness = max_extreme

    extremes.append(max_extreme)
    # find minimum extreme value
    for val in range(0, edge_thickness, 1):
        for i in range(0, width):
            for j in range(0, height):
                if img[i][j] == val:
                    if spr_local_extreme(img, i, j, val):
                        return int((val+max_extreme)/2)

    return int(max_extreme/2)