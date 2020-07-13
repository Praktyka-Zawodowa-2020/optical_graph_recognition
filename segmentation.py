"""Module with operations that are used to segment a graph picture"""
import cv2 as cv
import numpy as np

from math import ceil, floor, sqrt
from Vertex import Vertex

# Below are constants for HoughCircles function
MAX_R_FACTOR = 0.035
MIN_R_FACTOR = 0.005
DIST_FACTOR = 0.06
INNER_CANNY = 200
CIRCLE_THRESHOLD = 13

# Below are other constants
K = 4  # Consider 2 or 3 for lower and bigger K for higher resolutions
KERNEL_SIZE = 3  # Must be an odd number
COLOR_R_FACTOR = 0.5  # Should be < 1.0


def segment(source: np.ndarray, binary: np.ndarray, preprocessed: np.ndarray, imshow_enabled: bool) -> list:
    """
    Detect vertices in preprocessed image and return them in a list
    :param source: resized input image
    :param binary: binarized image from preprocessing phase
    :param preprocessed: fully preprocessed image
    :param imshow_enabled: flag determining to display (or not) segmentation steps with imshow function
    :return vertices_list: list of detected Vertices (objects of Vertex class)
    """
    # fill unfilled vertices
    filled = fill_vertices(preprocessed)

    # remove edges
    edgeless = remove_edges(filled)

    # detect vertices
    vertices_list, visualised  = find_vertices(source, binary, edgeless)

    # display results of certain steps
    if imshow_enabled:
        cv.imshow("filled", filled)
        cv.imshow("edgeless", edgeless)
        cv.imshow(str(len(vertices_list))+" detected vertices", visualised)

    return vertices_list


def fill_vertices(image: np.ndarray) -> np.ndarray:
    """
    Detect unfilled vertices in preprocessed image. Return image with vertices filled with object color (white)
    :param image: preprocessed image
    :return image: image with filled vertices
    """
    input_width = image.shape[1]
    # detect unfilled circles with Hough circles transform (parametrized based on image width)
    unfilled_v = cv.HoughCircles(
        image, cv.HOUGH_GRADIENT, 1,
        minDist=floor(input_width * DIST_FACTOR),
        param1=INNER_CANNY,
        param2=CIRCLE_THRESHOLD,
        minRadius=floor(input_width * MIN_R_FACTOR),
        maxRadius=ceil(input_width * MAX_R_FACTOR)
    )
    # Filling detected areas (circles)
    if unfilled_v is not None:
        circles = np.uint16(np.around(unfilled_v))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv.circle(image, center, round(radius), 255, thickness=cv.FILLED, lineType=8, shift=0)

    return image


def remove_edges(image: np.ndarray) -> np.ndarray:
    """
    Remove graph edges by performing erosion and dilation K times (also removes noise if some remained)
    :param image: preprocessed image with filled vertices
    :return dilated: image without edges (only vertices pixels)
    """
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    # eroding k times
    eroded = cv.erode(image, kernel, iterations=K)
    # dilating k times
    dilated = cv.dilate(eroded, kernel, iterations=K)
    return dilated


def find_vertices(source: np.ndarray, binary: np.ndarray, edgeless: np.ndarray) -> (list, np.ndarray):
    """
    Finds vertices based on detected contours in the edgeless image. Return list of those vertices.
    :param source: input image
    :param binary: binarized image from preprocessing phase
    :param edgeless: preprocessed image with filled vertices and without edges
    :return: list of detected vertices and their visualisation drawn on source image
    """
    # finding contours of vertices
    contours, _ = cv.findContours(edgeless, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # creating vertices from contours
    vertices_list = []
    visualized = np.copy(source)
    for c in contours:
        # calculating x and y
        x = round(np.average(c[:, :, 0]))  # center x is average of x coordinates of a contour
        y = round(np.average(c[:, :, 1]))  # similarly for y

        # calculating r
        dist = np.sqrt(np.sum(np.power(c - (x, y), 2), 1))  # distance from the center pixel for each pixel in contour
        r_original = (3.0*np.max(dist) + np.average(dist))/4.0
        r_final = round(r_original * 1.25)  # we take a bit bigger r to ensure that vertex is inside circle area

        # determining vertex color
        color = determine_binary_color(binary, x, y, r_original, COLOR_R_FACTOR)

        # creating vertex from calculated data and appending it to the list
        vertices_list.append(Vertex(x, y, r_original, color))

        # creating visual representation of detected vertices
        thickness = cv.FILLED if color == 255 else 2
        cv.circle(visualized, (x, y), r_final, (0, 255, 0), thickness, 8, 0)

    return vertices_list, visualized


def determine_binary_color(binary: np.ndarray, x: int, y: int, r_original: float, r_factor: float) -> int:
    """
    Determine vertex color by finding dominant color in vertex inner circle (excluding border pixels - r_factor < 1.0).
    :param binary: binarized image from preprocessing phase
    :param x: coordinate of vertex center
    :param y: coordinate of vertex center
    :param r_original: vertex radius
    :param r_factor: factor used to reduce original radius to exclude borders of unfilled vertex.
    :return: Dominant color in vertex area
    """
    if x >= 0 and y >= 0 and r_factor < 1.0:
        r = round(r_original * r_factor)  # calculate new, smaller, radius
        # calculate square area boundaries that contains circle area
        top = y - r if ((y - r) >= 0) else 0
        bottom = y + r if ((y + r) <= binary.shape[0]) else binary.shape[0]
        left = x - r if ((x - r) >= 0) else 0
        right = x + r if ((x + r) <= binary.shape[1]) else binary.shape[1]

        # in inner circle area count black and white pixels to find dominant color
        white_count = 0
        black_count = 0
        for y_iter in range(top, bottom):
            for x_iter in range(left, right):
                distance = round(sqrt((x-x_iter)**2+(y-y_iter)**2))
                if distance <= r:
                    if binary[y_iter, x_iter] == 255:
                        white_count += 1
                    else:
                        black_count += 1

        return 255 if white_count > black_count else 0
    else:  # bad input
        return -1


def negativ(img: np.ndarray) -> np.ndarray:
    """
    :param img: input image
    :return: negative image
    """
    width, height = img.shape[:2]
    negativ_image = np.zeros((width, height), np.uint8)
    for i in range(0, width):
        for j in range(0, height):
            negativ_image[i][j] = 255 - img[i][j]
    return negativ_image


def array_scope_control(img: np.ndarray, i: int, j: int) -> bool:
    """
    checks if (i,j) is in the table
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
    Calculates the pixel(i,j) value for the chamford function
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
    minimum = []
    if direction == 1:  # forward
        i -= 1
        j -= 1
        for x in range(0, 3):
            for y in range(0, 3):
                if x == 1 and y == 1:
                    return min(minimum)
                if array_scope_control(img, i + x, j + y):
                    if (img[i + x][j + y] + mask[x][y]) == 256:
                        minimum.append(0)
                    else:
                        minimum.append(img[i + x][j + y] + mask[x][y])
    else:  # backward
        i += 1
        j += 1
        for x in range(0, 3):
            for y in range(0, 3):
                if x == 1 and y == 1:
                    return min(minimum)
                if array_scope_control(img, i - x, j - y):
                    if (img[i - x][j - y] + mask[x][y]) == 256:
                        minimum.append(0)
                    else:
                        minimum.append(img[i - x][j - y] + mask[x][y])


def chamford(img: np.ndarray) -> np.ndarray:
    """
    The function implements a chamford algorithm that calculates the distance of the pixel from the background

    Chamford algorithm have two steps:
    Forward: goes right and down
    Backward: goes left up

    :param img: binary image. 255 pixel is background
    :return: image with the distance of the image pixels from the background
    """
    width, height = img.shape[:2]
    size = 3  # size mask in pixel_value
    chamford_image = np.zeros((width, height), np.uint8)
    # forward
    for i in range(math.floor((size + 1) / 2), width):
        for j in range(math.floor((size + 1) / 2), height):
            if img[i][j] < 255:
                chamford_image[i][j] = pixel_value(chamford_image, i, j, 1)
            else:
                chamford_image[i][j] = img[i][j]
    # backward
    for i in range(math.floor(width - (size - 1) / 2), 0, -1):
        for j in range(math.floor(height - (size - 1) / 2), 0, -1):
            if img[i][j] < 255:
                val = pixel_value(chamford_image, i, j, 2)
                if val < chamford_image[i][j]:
                    chamford_image[i][j] = val

    return chamford_image


def spr_local_extreme(img: np.ndarray, i: int, j: int, val: int) -> bool:
    """
    The function determines whether pixel (i, j) is a local extreme

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
                if img[i - x][j - y] > val and img[i - x][j - y] != 255:  # if the pixel is 255, its a background
                    return False
    return True


def extreme(img: np.ndarray) -> (int,int) :
    """
    Image must be after chamford function, background pixel value must be 255
    Finds min and max object extremum in image
    Object extremum is the furthest pixel distance from the background


    :param img: Image
    :return: minimum,maximum
    """
    width, height = img.shape[:2]
    extremes = []
    max_extreme = 0
    # find max extreme value
    for i in range(0, width):
        for j in range(0, height):
            if max_extreme < img[i][j] < 255:
                max_extreme = img[i][j]
    # find all local extreme value
    for val in range(1, max_extreme):
        for i in range(0, width):
            for j in range(0, height):
                if img[i][j] == val:
                    if spr_local_extreme(img, i, j, val):
                        extremes.append(val)

    return min(extremes), max_extreme