"""Module with operations that are used to segment a graph picture"""
import cv2 as cv
import numpy as np

from math import ceil, floor, sqrt
from Vertex import Vertex
# Below are constants for HoughCircles function
from shared import Color, Kernel, Mode

MAX_R_FACTOR: float = 0.035
MIN_R_FACTOR: float = 0.005
DIST_FACTOR: float = 0.06
INNER_CANNY: int = 200
CIRCLE_THRESHOLD: int = 13

# Below are other constants
COLOR_R_FACTOR: float = 0.5  # Should be < 1.0


def segment(source: np.ndarray, binary: np.ndarray, preprocessed: np.ndarray, imshow_enabled: bool, mode: int) -> [list, np.ndarray]:
    """
    Detect vertices in preprocessed image and return them in a list

    :param source: resized input image
    :param binary: binarized image from preprocessing phase
    :param preprocessed: fully preprocessed image
    :param imshow_enabled: flag determining to display (or not) segmentation steps
   :param mode: GRID_BG, CLEAN_BG, PRINTED
    :return vertices_list: list of detected Vertices (objects of Vertex class) and visualised results of detection

    """
    # fill unfilled vertices
    filled = fill_vertices(preprocessed, mode)

    # remove edges
    edgeless = remove_edges(filled)

    # detect vertices
    vertices_list, visualised = find_vertices(source, binary, edgeless)

    # display results of certain steps
    if imshow_enabled:
        cv.imshow("filled", filled)
        cv.imshow("edgeless", edgeless)
        cv.imshow(str(len(vertices_list))+" detected vertices", visualised)

    return vertices_list, visualised


def fill_vertices(image: np.ndarray, mode: int) -> np.ndarray:
    """
    Detect unfilled vertices in preprocessed image
    Return image with vertices filled with object color

    :param image: preprocessed image
    :param mode: input mode, see shared.py for more info
    :return image: image with filled vertices
    """

    if mode == Mode.PRINTED:
        image = fill_elliptical_contours(image, 0.6, 1.25)
    elif mode == Mode.CLEAN_BG:
        image = fill_elliptical_contours(image, 0.3)
    elif mode == Mode.GRID_BG:
        image = fill_elliptical_contours(image, 0.35)

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
            cv.circle(image, center, round(radius), Color.OBJECT, thickness=cv.FILLED, lineType=8, shift=0)

    # Vertices are not perfect circles so after circle fill we fill small gaps inside vertices with closing operation
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, Kernel.k7)

    return image


def fill_elliptical_contours(image: np.ndarray, threshold: float = 0.5, round_ratio: float = 4.0) -> np.ndarray:
    """
    Fill elliptical inner contours which are treated as vertices

    :param image: input image
    :param threshold: value from 0 to 1 - the bigger the more elliptical vertices should be to be filled
    :param round_ratio: major axis to minor axis ratio (1; inf) -  the less the more round vertices should be
    :return: image with filled elliptical vertices
    """
    processed = cv.morphologyEx(image, cv.MORPH_CLOSE, Kernel.k3, iterations=4)  # fill small gaps and close contours
    # find contours in 2 level hierarchy: inner and outer contours - inner contours in parent field have non -1 value
    contours, hierarchy = cv.findContours(processed, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    original = image.copy()
    img_area = image.shape[0] * image.shape[1]

    for i in range(0, len(contours)):
        # check for a contour parent (indicates being inner contour), also filter to big and to small contours
        if hierarchy[0][i][3] != -1 and img_area * 0.1 >= cv.contourArea(contours[i]) >= img_area * 0.0001:
            (x, y), (a, b), angle = cv.minAreaRect(contours[i])  # rotated bounding rect describe fitted ellipse
            if round_ratio >= a/b >= 1.0/round_ratio:  # check if fitted ellipse is round enough to be a vertex
                ellipse_cnt = cv.ellipse2Poly((int(x), int(y)), (int(a / 2.0), int(b / 2.0)), int(angle), 0, 360, 1)
                overlap_level = contours_overlap_level(ellipse_cnt, contours[i])
                if overlap_level >= threshold:  # if ellipse and inner contour overlap enough then fill vertex (contour)
                    cv.drawContours(original, contours, i, Color.OBJECT, thickness=cv.FILLED)
    return original


def contours_overlap_level(contour1, contour2):
    """
    Calculate overlap level of two contours

    :param contour1: for each pixel of this contour minimal distance to second contour will be calculated
    :param contour2: second contour
    :return: overlap level in range (0;1) - 0 indicating no overlapping and 1 indicating full overlapping
    """
    # if contour is big enough then overlaying limit is 1 pixel, otherwise for small contours it is 0 (exact overlay)
    dist_limit = 1 if cv.contourArea(contour1) >= 150 else 0
    overlaying_pixels = 0
    for i in range(0, len(contour1)):
        x, y = contour1[i]
        dist = abs(cv.pointPolygonTest(contour2, (x, y), True))
        if dist <= dist_limit:  # contour pixels are considered overlaying if they are distant not more than limit
            overlaying_pixels += 1
    overlay_level = overlaying_pixels / float(len(contour1))  # calculate overlaying to all pixels in contour ratio

    return overlay_level


def remove_edges(image: np.ndarray) -> np.ndarray:
    """
    Remove graph edges by performing erosion and dilation K times (also removes noise if some remained)

    :param image: preprocessed image with filled vertices
    :return dilated: image without edges (only vertices pixels)
    """
    kernel = Kernel.k3
    eroded = image.copy()
    contours, _ = cv.findContours(eroded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    start, before = len(contours), len(contours)
    K, counter = 0, 0

    # eroding k times
    while True:
        i = len(contours)
        if i == before and before != start:
            counter = counter+1
        else:
            counter = 0

        if start != i:
            start = 0
        if counter == 1:
            break
        eroded = cv.erode(eroded, kernel, iterations=1)
        K = K + 1
        contours, _ = cv.findContours(eroded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        before = i

        if len(contours) == 0:
            break

    eroded = cv.erode(eroded, kernel, iterations=1)
    K = K+1

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
    for i in range(0, len(contours)):
        cnt = contours[i]
        # calculating x and y
        x = round(np.average(cnt[:, :, 0]))  # center x is average of x coordinates of a contour
        y = round(np.average(cnt[:, :, 1]))  # similarly for y

        # calculating r
        dist = np.sqrt(np.sum(np.power(cnt - (x, y), 2), 1))  # distance from the center pixel for each pixel in contour
        r_original = (3*np.max(dist) + np.average(dist))/4.0
        r_final = round(r_original * 1.25)  # we take a bit bigger r to ensure that vertex is inside circle area

        # determining vertex color
        color = determine_binary_color(binary, x, y, r_original, COLOR_R_FACTOR)

        # creating vertex from calculated data and appending it to the list
        vertices_list.append(Vertex(x, y, r_original, color))

        # creating visual representation of detected vertices
        thickness = cv.FILLED if color == Color.OBJECT else 2
        cv.circle(visualized, (x, y), r_final, Color.GREEN, thickness, 8, 0)
        # cv.putText(visualized, str(i), (x, y), cv.QT_FONT_NORMAL, 0.75, Color.WHITE)

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
        object_count = 0
        bg_count = 0
        for y_iter in range(top, bottom):
            for x_iter in range(left, right):
                distance = round(sqrt((x-x_iter)**2+(y-y_iter)**2))
                if distance <= r:
                    if binary[y_iter, x_iter] == Color.OBJECT:
                        object_count += 1
                    else:
                        bg_count += 1

        return Color.OBJECT if object_count >= bg_count else Color.BG
    else:  # bad input
        return -1
