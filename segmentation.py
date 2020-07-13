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
K = 4  # Consider 2 or 3 for lower resolutions
KERNEL_SIZE = 3  # Must be an odd number
COLOR_R_FACTOR = 0.5  # Should be < 1.0


# performs all steps in segmentation phase to detect vertices
def segment(source: np.ndarray, binary: np.ndarray, preprocessed: np.ndarray, imshow_enabled: bool) -> list:
    # fill unfilled vertices
    filled = fill_vertices(preprocessed)

    # remove edges
    edgeless = remove_edges(filled)

    # detect vertices
    visualised, vertices_list = find_vertices(source, binary, edgeless)

    # display results of certain steps
    if imshow_enabled:
        cv.imshow("filled", filled)
        cv.imshow("edgeless", edgeless)
        cv.imshow(str(len(vertices_list))+" detected vertices", visualised)

    return vertices_list


# detects unfilled vertices and fills them in given image
def fill_vertices(image: np.ndarray) -> np.ndarray:
    input_width = image.shape[1]
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


# removes edges by performing erosion and dilation K times (also removes some noise remaining after preprocessing)
def remove_edges(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    # eroding k times
    eroded = cv.erode(image, kernel, iterations=K)
    # dilating k times
    dilated = cv.dilate(eroded, kernel, iterations=K)
    return dilated


# finds vertices based on detected contours in the edgeless image
def find_vertices(source: np.ndarray, binary: np.ndarray, edgeless: np.ndarray) -> (np.ndarray, list):
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

    return visualized, vertices_list


# determines vertex color by finding dominant color in vertex inner circle excluding border pixels (r_factor < 1.0)
def determine_binary_color(binary: np.ndarray, x: int, y: int, r_original: float, r_factor: float) -> int:
    """

    :param binary:
    :param x:
    :param y:
    :param r_original:
    :param r_factor:
    :return:
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
