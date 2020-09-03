"""Module with operations that are used to recognize topology of a graph"""
import cv2 as cv
import numpy as np
import math
import functools

from Vertex import Vertex
from shared import Color, Debug, Kernel, Mode, distance_L2

# constants
MIN_EDGE_LEN: int = 10  # px
VERTEX_AREA_FACTOR: float = 1.3
SEARCH_RADIUS_CONST = 15

# debugging windows title prefix
DBG_TITLE = "topology recognition: "


def recognize_topology(vertices_list: list, preprocessed: np.ndarray, visualised: np.ndarray, edge_thickness: int,
                       mode: Mode, debug: Debug) -> list:
    """
    Remove vertices from image and based on remaining contours detect edges that connect those vertices.
    Result of detection is a list of vertices with adjacency (connections) list for each vertex.

    :param vertices_list: list of detected vertices in segmentation phase
    :param preprocessed: image after preprocessing phase
    :param visualised: copy of source image with vertices drawn
    :param edge_thickness: value indicating how much erosion iterations has been performed in segmentation in order to
        remove edges from binary image
    :param mode: input type, see shared.py for more detailed description
    :param debug: indicates how much debugging windows will be displayed
    :return: list where each vertex has list of adjacent vertices (those that are connected)
    """
    preprocessed = remove_vertices(vertices_list, preprocessed, VERTEX_AREA_FACTOR)
    preprocessed = remove_lines_intersections(preprocessed, edge_thickness)

    lines_list, backend = lines_from_contours(preprocessed, visualised.copy())

    search_radius = int(np.average(np.array([v.r for v in vertices_list])))  # calculate optimal search area radius
    search_radius = search_radius if search_radius > SEARCH_RADIUS_CONST else SEARCH_RADIUS_CONST
    if mode == Mode.PRINTED:
        linked_lines, backend = link_nearby_endpoints(lines_list, backend, 1.5 * search_radius, 15)
        vertices_list, backend, visualised = edges_from_lines(linked_lines, vertices_list, backend, visualised, 2.4)
    else:
        linked_lines, backend = link_nearby_endpoints(lines_list, backend, 1.5 * search_radius, 20)
        vertices_list, backend, visualised = edges_from_lines(linked_lines, vertices_list, backend, visualised, 3.1)

    # Display intermediate results of topology recognition
    if debug == Debug.FULL:
        cv.imshow(DBG_TITLE+"removed vertices and lines intersections", preprocessed)
        cv.imshow(DBG_TITLE+"\"backend\" - colors description in topology.py", backend)
        # backend colors description: yellow - lines contours, cyan - lines segments endpoints, orange - approximated
        # edges, purple - vertex search area for edges endpoints, pink - areas for linking endpoints
    # Display final results of topology recognition
    if debug == Debug.GENERAL or debug == Debug.FULL:
        cv.imshow(DBG_TITLE+"final results - green: vertices, red: edges", visualised)

    return vertices_list


def remove_vertices(vertices_list: list, preprocessed: np.ndarray, vertex_area_factor: float) -> np.ndarray:
    """
    Remove vertices areas from binary preprocessed image by filling those areas with background color

    :param vertices_list: list of detected vertices in segmentation phase
    :param preprocessed: image after preprocessing phase
    :param vertex_area_factor: radius factor indicating how much area will be removed around vertices centers
    :return: processed image without vertices
    """
    for vertex in vertices_list:
        # remove vertices
        cv.circle(preprocessed, (vertex.x, vertex.y), round(vertex.r * vertex_area_factor), Color.BG, cv.FILLED)
    return preprocessed


def remove_lines_intersections(preprocessed: np.ndarray, edge_thickness: int) -> np.ndarray:
    """
    Remove line intersections in binary image.

    :param preprocessed: preprocessed image with vertices removed
    :param edge_thickness: value indicating how much erosion iterations has been performed in segmentation in order to
        remove edges from binary image
    :return: image with line intersections removed
    """
    ret_image = preprocessed.copy()
    skeleton = cv.ximgproc.thinning(preprocessed)  # skeletonize image
    # create kernels that will extract intersection points (junction points to be more specific)
    kernels_list = [
        np.array([[1, 0, 1],
                  [0, 1, 0],
                  [0, 1, 0]]),
        np.array([[0, 1, 0],
                  [0, 1, 1],
                  [1, 0, 0]]),
        np.array([[1, 0, 0],
                  [0, 1, 0],
                  [1, 0, 1]])
    ]

    for kernel in kernels_list:
        for i in range(0, 4):  # for each orientation of kernel in list
            intersections = cv.morphologyEx(skeleton, cv.MORPH_HITMISS, kernel)  # get intersection pixels
            # dilate these pixels to the thickness of an edge to create intersection regions
            regions = cv.dilate(intersections, Kernel.k3, iterations=edge_thickness+1)
            ret_image = cv.subtract(ret_image, regions)  # subtract intersection regions from image
            kernel = np.rot90(kernel)  # rotate kernel to create another intersection pattern for next interation

    return ret_image


def lines_from_contours(preprocessed: np.ndarray, backend: np.ndarray, min_line_length: float = 10) \
        -> (list, np.ndarray):
    """
    From image with removed vertices approximate each contour with straight line

    :param preprocessed: input preprocessed image with removed vertices
    :param backend: image with visualisation of topology recognition backend
    :param min_line_length: all approximated lines of smaller length than this value will be considered noise and not
        added to the list of lines
    :return: List of lines (tuples of 2 endpoints) and image with lines visualised
    """
    lines_list = []
    contours, hierarchy = cv.findContours(preprocessed, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(backend, contours, -1, Color.YELLOW, 1)
    for i in range(0, len(contours)):
        if hierarchy[0][i][3] == -1:  # outer contours only
            cnt = contours[i]
            pt1, pt2 = fit_line(cnt)
            # if line has been fitted take lines that are long enough to be an edge
            if pt1 is not None and pt2 is not None and distance_L2(pt1, pt2) >= min_line_length:
                cv.circle(backend, (pt1[0], pt1[1]), 4, Color.CYAN, cv.FILLED)
                cv.circle(backend, (pt2[0], pt2[1]), 4, Color.CYAN, cv.FILLED)
                lines_list.append([pt1, pt2])
    return lines_list, backend


def fit_line(edge_contour: list, epsilon: float = 0.01, delta: float = 0.01) -> ([int, int], [int, int]):
    """
    Approximate edge contour with a straight line (2 endpoints) using contour approximation operation
    This is achieved by approximating contour until 2 points remain and are treated as line endpoints

    :param edge_contour: set of points that describe edge contour
    :param epsilon: initial part of contour perimieter that will be used for contour approximation
    :param delta: step of epsilon in each loop
    :return: two endpoints of an approximating line
    """

    perimeter = cv.arcLength(edge_contour, True)
    while True:
        approx = cv.approxPolyDP(edge_contour, perimeter * epsilon, True)
        if len(approx) <= 2:
            break
        epsilon += delta
    if len(approx) == 2:  # exactly 2 (end)points remained
        return approx[0][0].astype(int), approx[1][0].astype(int)

    # approximation failure - left one or less points
    return None, None


def link_nearby_endpoints(lines_list: list, backend: np.ndarray, search_radius: float, angle_threshold: float) \
        -> (list, np.ndarray):
    """
    Link edge segments (lines) that are close to each other and go at a similar angle. This operation is performed
    because sometimes, one edge is separated into multiple lines (e.g. when intersections occur)

    :param lines_list: list of lines (tuples of 2 endpoints)
    :param backend: source image with visualised topology recognition backend
    :param search_radius: radius determining area around endpoint that other endpoint of other line has to be in to
        be considered a "nearby" endpoint
    :param angle_threshold: threshold that describes upper limit for angle (in degrees)
        that lines of 2 nearby endpoints have to be at to be linked
    :return: list of lines, where nearby line segments are connected into single lines also visualised results
    """
    # sort lines by descending length - longer lines have higher priority (occur closer to the beginning of the list)
    lines_list = sorted(lines_list, key=functools.cmp_to_key(lines_lengths_compare), reverse=True)
    i = 0
    while i < len(lines_list):
        if lines_list[i] is None:  # skip already linked lines
            i += 1
            continue
        line = lines_list[i]
        line_len = distance_L2(line[0], line[1])
        search_radius = search_radius if search_radius <= line_len*1.2 else line_len*1.2
        for j in range(0, len(line)):  # for each one of 2 endpoints
            main_point = line[j]  # search around this endpoint for endpoints from different lines
            other_point = line[(j+1) % 2]  # other endpoint from the same line
            cv.circle(backend, (main_point[0], main_point[1]), int(search_radius), Color.PINK, 1)
            cv.line(backend, (main_point[0]-int(search_radius), main_point[1]),
                    (main_point[0]+int(search_radius), main_point[1]), Color.PINK, 1)
            if main_point is None or other_point is None:
                break
            else:
                main_angle = vector_angle(other_point, main_point)
                in_area_list = find_endpoints_in_area(lines_list, i+1, main_point[0], main_point[1], search_radius,
                                                      main_angle)  # find all endpoints in area of main point
                if in_area_list is not None:
                    deltas = in_area_list[:, 2]  # extract angle differences into separate array
                    min_delta = np.min(deltas)
                    if min_delta <= angle_threshold:  # check if lines are going at a similar angle
                        min_index = np.argmin(deltas)
                        k, l = (int(in_area_list[min_index][0]), int(in_area_list[min_index][1]))
                        # create linked line by assigning new value to main_point place in list
                        lines_list[i][j] = lines_list[k][(l + 1) % 2]  # find new endpoint for main line
                        lines_list[k] = None  # mark line as linked
                        i -= 1  # take another iteration over current line since it has just changed
                        break  # start new iteration with new linked line
        i += 1
    final_lines_list = []
    i = 0
    for line in lines_list:
        i += 1
        if line is None:  # remove fields in list that remained after they were linked to other lines
            continue
        else:  # print lines
            final_lines_list.append(line)
            pt1, pt2 = line
            cv.line(backend, (pt1[0], pt1[1]), (pt2[0], pt2[1]), Color.ORANGE, 2)

    return final_lines_list, backend


def lines_lengths_compare(line1: ([int, int], [int, int]), line2: ([int, int], [int, int])) -> int:
    """
    Compare 2 lines in terms of length

    :param line1: first line (2 endpoints)
    :param line2: second line --||--
    :return:
         1 - first line is longer than second one
         0 - lines lengths are equal
        -1 - second line is longer than first one
    """
    len1 = distance_L2(line1[0], line1[1])
    len2 = distance_L2(line2[0], line2[1])
    if len1 > len2:
        return 1
    elif len1 == len2:
        return 0
    else:
        return -1


def vector_angle(start_pt: [int, int], end_pt: [int, int]) -> float:
    """
    Calculate vector angle with Y axis in cartesian coordinate system

    :param start_pt: starting point of vector
    :param end_pt: ending point of vector
    :return: angle in degrees that vector forms with Y axis going clockwise
    """
    tmp_vec = start_pt - end_pt  # calculate vector value from 2 endpoints
    angle = np.arctan2(tmp_vec[0], tmp_vec[1])  # calculate angle in radians
    angle = angle * 180 / math.pi + 180  # convert to degrees
    return angle


def find_endpoints_in_area(lines_list: list, start_index: int, x: int, y: int, radius: float, main_angle: float) \
        -> np.ndarray:
    """
    Find all endpoints in circle area (center and radius given)
    Also calculate angle difference between each line containing endpoint in area and main endpoint line

    :param lines_list: list of lines (tuples of 2 endpoints)
    :param start_index: starting index for search in lines list
    :param x: center coordinate
    :param y: center coordinate
    :param radius: determines area of search
    :param main_angle: angle of line that search point is in
    :return: numpy array of endpoint descriptions containing: (if no endpoints found empty list is returned)
        line_index - line index in list
        point_index - endpoint index in line (0 or 1)
        delta - difference of angles between main line, and lines containing in_area endpoints
    """
    in_area_list = []
    endpoint = np.array([x, y])
    for i in range(start_index, len(lines_list)):
        if lines_list[i] is None:
            continue
        for j in range(0, 2):
            tmp_endpoint = np.array([lines_list[i][j][0], lines_list[i][j][1]])
            if distance_L2(endpoint, tmp_endpoint) <= radius:  # calculate distance
                other_endpoint = np.array([lines_list[i][(j+1) % 2][0], lines_list[i][(j+1) % 2][1]])
                tmp_angle = vector_angle(tmp_endpoint, other_endpoint)
                diff = abs(main_angle - tmp_angle)
                delta = diff if diff <= 180 else 360 - diff
                in_area_list.append([i, j, delta])
    ret_arr = np.array(in_area_list) if in_area_list else None
    return ret_arr


def edges_from_lines(lines_list: list, vertices_list: list, backend: np.ndarray, final_results: np.ndarray,
                     within_r_factor: float) -> (list, np.ndarray, np.ndarray):
    """
    From extracted lines create edges (connections between adjacent vertices).

    :param lines_list: list of lines extracted from the image
    :param vertices_list: list of detected vertices in segmentation phase
    :param backend: source image with visualised topology recognition backend
    :param final_results: copy of source image used to display final results of OGR algorithm
    :param within_r_factor: factor to increase/decrease radius and therefore area of accepting lines endpoints as edges
    :return: list where each vertex has list of adjacent (connected) vertices,
        images with visualised intermediate recognition steps and final results,
    """
    for pt1, pt2 in lines_list:
        index1 = find_nearest_vertex(pt1, vertices_list)
        index2 = find_nearest_vertex(pt2, vertices_list)
        v1, v2 = (vertices_list[index1], vertices_list[index2])
        # visualise "within areas"
        cv.circle(backend, (v1.x, v1.y), round(v1.r * within_r_factor), Color.PURPLE, thickness=2)
        cv.circle(backend, (v2.x, v2.y), round(v2.r * within_r_factor), Color.PURPLE, thickness=2)
        # check if:
        # 1. points are within vertex area,
        # 2. edge endpoints are assigned to different vertices,
        # 3. connection doesn't already exists
        if point_within_radius(pt1, v1, within_r_factor) and point_within_radius(pt2, v2, within_r_factor)\
                and index1 != index2\
                and index2 not in v1.adjacency_list and index1 not in v2.adjacency_list:
            v1.adjacency_list.append(index2)  # create adjacency
            v2.adjacency_list.append(index1)  # ---||---
            cv.line(final_results, (v1.x, v1.y), (v2.x, v2.y), Color.RED, thickness=2)  # final edge
            cv.circle(final_results, (v1.x, v1.y), 4, Color.BLACK, cv.FILLED)           # edge endpoint
            cv.circle(final_results, (v2.x, v2.y), 4, Color.BLACK, cv.FILLED)           # --||--

    return vertices_list, backend, final_results


def find_nearest_vertex(point: np.ndarray, vertices_list: list) -> int:
    """
    Find vertex nearest to a given point (based on euclidean distance - L2)

    :param point: x and y coordinates from which distance to vertices is measured
    :param vertices_list: list of detected vertices in segmentation phase
    :return nearest_index: index in vertices list of vertex nearest to the given point
    """
    # Initialise values with first on list
    nearest_index = 0
    current_vertex = vertices_list[nearest_index]
    current_center = np.array([current_vertex.x, current_vertex.y])
    min_distance = distance_L2(point, current_center)
    for i in range(0, len(vertices_list)):
        current_vertex = vertices_list[i]
        current_center = np.array([current_vertex.x, current_vertex.y])
        distance = distance_L2(point, current_center)
        if distance < min_distance:     # Found vertex closer to a point
            nearest_index = i
            min_distance = distance
    return nearest_index


def point_within_radius(point: np.ndarray, vertex: Vertex, radius_factor: float) -> bool:
    """
    Check if point is within vertex area with radius modified by factor (based on euclidean distance - L2)

    :param point: x and y coordinates
    :param vertex: vertex which area is considered
    :param radius_factor: factor to increase/decrease radius and therefore area
    :return: True if point is within radius, and False if it is not
    """
    radius = vertex.r * radius_factor
    return True if distance_L2(point, [vertex.x, vertex.y]) <= radius else False
