"""Module with operations that are used to segment a graph picture"""
import cv2 as cv
import numpy as np

from math import ceil, floor, sqrt
from Vertex import Vertex
from shared import Color, Kernel, Mode

# Constants for HoughCircles function
MAX_R_FACTOR: float = 0.04
MIN_R_FACTOR: float = 0.005
DIST_FACTOR: float = 0.06
INNER_CANNY: int = 200

# Other constants
ROUND_RATIO: float = 3.0
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
        round_factor = 1.25
        image = fill_elliptical_contours(image, 0.6, round_factor)
        image = cv.morphologyEx(image, cv.MORPH_CLOSE, Kernel.k3, iterations=1)
        image = fill_circular_shapes(image, 18, round_factor)
    elif mode == Mode.CLEAN_BG:
        round_factor = 2
        image = fill_elliptical_contours(image, 0.3, round_factor)
        image = fill_circular_shapes(image, 13, round_factor)
    elif mode == Mode.GRID_BG:
        round_factor = 2.5
        image = fill_elliptical_contours(image, 0.35, round_factor)
        image = fill_circular_shapes(image, 13, round_factor)

    # fill small gaps that remained in the picture with closing operator, and with contours operations
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, Kernel.k3, iterations=1)
    image = fill_small_contours(image, 0.001)

    return image


def fill_elliptical_contours(image: np.ndarray, threshold: float = 0.5, round_ratio: float = ROUND_RATIO) -> np.ndarray:
    """
    Fill elliptical inner contours which are treated as vertices
    Remove inner contours that are less round than round_ratio

    :param image: input image
    :param threshold: value from 0 to 1 - the bigger the more elliptical vertices should be to be filled
    :param round_ratio: vertex (ellipsis) major axis to minor axis ratio (1; inf)
    The bigger value, the less circular vertices have to be, to be filled
    :return: image with filled elliptical vertices
    """
    processed = cv.morphologyEx(image, cv.MORPH_CLOSE, Kernel.k3, iterations=4)  # fill small gaps and close contours
    # find contours in 2 level hierarchy: inner and outer contours - inner contours in parent field have non -1 value
    contours, hierarchy = cv.findContours(processed, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    original = image.copy()
    img_area = image.shape[0] * image.shape[1]

    for i in range(0, len(contours)):
        # check for a contour parent (indicates being inner contour), also filter to big and to small contours
        if hierarchy[0][i][3] != -1 and img_area * 0.1 >= cv.contourArea(contours[i]) >= img_area * 0.0003:
            (x, y), (a, b), angle = cv.minAreaRect(contours[i])  # rotated bounding rect describe fitted ellipse
            if round_ratio >= a/b >= 1.0/round_ratio:  # check if fitted ellipse is round enough to be a vertex
                ellipse_cnt = cv.ellipse2Poly((int(x), int(y)), (int(a / 2.0), int(b / 2.0)), int(angle), 0, 360, 1)
                overlap_level = contours_overlap_level(ellipse_cnt, contours[i])
                if overlap_level >= threshold:  # if ellipse and inner contour overlap enough then fill vertex (contour)
                    cv.drawContours(original, contours, i, Color.OBJECT, thickness=cv.FILLED)
            else:
                cv.drawContours(original, contours, i, Color.BG, thickness=6)

    return original


def contours_overlap_level(contour1, contour2):
    """
    Calculate overlap level of two contours

    :param contour1: for each pixel of this contour minimal distance to second contour will be calculated
    :param contour2: second contour
    :return: overlap level in range (0;1) - 0 indicating no overlapping and 1 indicating full overlapping
    """
    # if contour is big enough then overlapping limit is 1 pixel, otherwise for small contours it is 0 (exact overlap)
    dist_limit = 1 if cv.contourArea(contour1) >= 150 else 0
    overlapping_pixels = 0
    for i in range(0, len(contour1)):
        x, y = contour1[i]
        dist = abs(cv.pointPolygonTest(contour2, (x, y), True))  # minimal distance for point x, y with contour2 points
        if dist <= dist_limit:  # contour pixels are considered overlapping if they are distant not more than limit
            overlapping_pixels += 1
    overlay_level = overlapping_pixels / float(len(contour1))  # calculate overlapping to all pixels in contour ratio

    return overlay_level


def fill_circular_shapes(image: np.ndarray, circle_threshold: int, round_ratio: float = ROUND_RATIO) -> np.ndarray:
    """
    Fill circle-like shapes, which are considered vertices.

    :param image: input image (with elliptical contours filled)
    :param circle_threshold: threshold value for HouhgCircles function
    :param round_ratio: see get_hough_params function
    :return: image with filled circular regions in the image
    """
    # get optimal parameters for HoughCircles function based on already filled vertices
    r_min, r_max, min_dist = get_hough_param(image, round_ratio)

    # detect unfilled circles
    circles = cv.HoughCircles(
        image, cv.HOUGH_GRADIENT, 1,
        minDist=min_dist,
        param1=INNER_CANNY,
        param2=circle_threshold,
        minRadius=r_min,
        maxRadius=r_max
    )
    # Fill detected circular areas
    if circles is not None:
        print(circles)
        for circle in circles[0]:
            x, y, r = circle
            cv.circle(image, (int(x), int(y)), int(r), Color.OBJECT, thickness=cv.FILLED)

    return image


def get_hough_param(image: np.ndarray, round_ratio: float = ROUND_RATIO) -> (int, int, int):
    """
    Find optimal radius range and minimal distance between vertices based on already filled vertices.
    If there are no filled vertices, estimate parameters based on image width.

    :param image: input input image (with elliptical contours filled)
    :param round_ratio: vertex (ellipsis) major axis to minor axis ratio (1; inf)
        the bigger value, the less circular filled vertices have to be to determine radius for unfilled ones
    :return: minimal radius, maximal radius and minimal distance for HoughCircles function
    """
    edgeless = image.copy()

    edgeless = remove_edges(edgeless)
    contours, hierarchy = cv.findContours(edgeless, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    radius_list = []
    if contours is not None:
        for i in range(0, len(contours)):
            if hierarchy[0][i][3] == -1:  # outer contours
                x, y, w, h = cv.boundingRect(contours[i])
                if 1.0/round_ratio <= h/w <= round_ratio:  # consider only round enough contours (vertices)
                    (x, y), r = cv.minEnclosingCircle(contours[i])
                    radius_list.append(r)
                    cv.circle(edgeless, (int(x), int(y)), int(r), 127, thickness=3)
        if radius_list:
            r_avg = np.average(radius_list)
            r_min = floor(r_avg*0.5)
            r_max = ceil(r_avg*1.2)
            min_dist = r_avg*3

    if (contours is None) or (not radius_list):  # no filled vertices in the image
        r_min = floor(image.shape[1] * MIN_R_FACTOR)
        r_max = ceil(image.shape[1] * MAX_R_FACTOR)
        min_dist = floor(image.shape[1] * DIST_FACTOR)

    return r_min, r_max, min_dist


def fill_small_contours(image: np.ndarray, max_area_factor: float) -> np.ndarray:
    """
    Fill small inner closed contours in image
    :param image: input image
    :param max_area_factor: factor indicating what part of image area is upper limit for contour area to be filled
    :return: image with small inner contours filled
    """
    contours, hierarchy = cv.findContours(image, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    for i in range(0, len(contours)):
        # fill only small inner contours
        if hierarchy[0][i][3] != -1 and cv.contourArea(contours[i]) <= image.shape[0]*image.shape[1]*max_area_factor:
            cv.drawContours(image, contours, i, Color.OBJECT, thickness=cv.FILLED)
    return image


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
        if color == Color.OBJECT:
            thickness = 8
            # cv.putText(visualized, "F", (abs(x-10), abs(y+10)), cv.QT_FONT_NORMAL, 1.0, Color.GREEN)
        else:
            thickness = 2

        cv.circle(visualized, (x, y), r_final, Color.GREEN, thickness, 8, 0)

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
