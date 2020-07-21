"""Module with operations that are used to recognize topology of a graph"""
import cv2 as cv
import numpy as np
import math

from Vertex import Vertex
from shared import Color

# constants
MIN_EDGE_LEN: int = 10  # px
VERTEX_AREA_FACTOR: float = 1.6
WITHIN_R_FACTOR: float = 3.4


def recognize_topology(vertices_list: list, preprocessed: np.ndarray, visualised: np.ndarray, imshow_enabled: bool) \
        -> list:
    """
    Remove vertices from image, and based on remaining contours detect edges that connect those vertices.
    Result of detection is a list of vertices with connection (neighbour) list for each vertex.

    :param vertices_list: list of detected vertices in segmentation phase
    :param preprocessed: image after preprocessing phase
    :param visualised: copy of source image with vertices drawn
    :param imshow_enabled: flag determining to display (or not) topology recognition steps
    :return: list where each vertex has list of connected vertices (its neighbours)
    """
    preprocessed, topology_backend = remove_vertices(vertices_list, preprocessed, visualised)

    vertices_list, topology_backend, visualised = find_edges(vertices_list, preprocessed, topology_backend, visualised)

    if imshow_enabled:
        cv.imshow("removed vertices", preprocessed)
        cv.imshow("topology backend: search areas and approx. edges", topology_backend)
        cv.imshow("visualised", visualised)

    return vertices_list


def remove_vertices(vertices_list: list, preprocessed: np.ndarray, visualised: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    # Remove vertices areas from binary preprocessed image by setting those areas as background
    # Also mark areas for each vertex within which endpoints of edges will can be assigned to vertex

    :param vertices_list: list of detected vertices in segmentation phase
    :param preprocessed: image after preprocessing phase
    :param visualised: copy of source image with vertices drawn
    :return: processed image without vertices and image with visualised vertices areas
    """
    within_areas = np.copy(visualised)
    for vrtx in vertices_list:
        # remove vertices
        cv.circle(
            preprocessed, (vrtx.x, vrtx.y), round(vrtx.r * VERTEX_AREA_FACTOR),
            color=Color.BG, thickness=cv.FILLED, lineType=8
        )
        # visualise "within area"
        cv.circle(within_areas, (vrtx.x, vrtx.y), round(vrtx.r * WITHIN_R_FACTOR), Color.BLUE, thickness=cv.FILLED)
    return preprocessed, within_areas


def find_edges(vertices_list: list, preprocessed: np.ndarray, topology_backend: np.ndarray, visualised: np.ndarray)\
        -> (list, np.ndarray, np.ndarray):
    """
    Find vertices edges from contours.

    :param vertices_list: list of detected vertices in segmentation phase
    :param preprocessed: image after preprocessing phase
    :param topology_backend: source image with visualised vertices areas
    :param visualised: copy of source image with vertices drawn
    :return: list where each vertex has list of connected vertices (its neighbours),
    image with visualised intermediate recognition steps and shapes, image with visualised final topology
    """
    contours, _ = cv.findContours(preprocessed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(topology_backend, contours, -1, Color.YELLOW, 1)
    for cnt in contours:
        pt1, pt2 = fit_line(cnt)
        if np.linalg.norm(pt1 - pt2) > MIN_EDGE_LEN:    # considering lines that are long enough to be an edge
            cv.line(
                topology_backend, (round(pt1[0]), round(pt1[1])), (round(pt2[0]), round(pt2[1])),
                Color.ORANGE, thickness=3
            )
            index1 = find_nearest_vertex(pt1, vertices_list)
            vertex1 = vertices_list[index1]
            index2 = find_nearest_vertex(pt2, vertices_list)
            vertex2 = vertices_list[index2]
            # check if:
            # points are within vertex area,
            # edge endpoints are assigned to different vertices,
            # connection doesn't already exists
            if point_within_radius(pt1, vertex1, WITHIN_R_FACTOR)\
                    and point_within_radius(pt2, vertex2, WITHIN_R_FACTOR)\
                    and index1 != index2\
                    and index2 not in vertex1.neighbour_list and index1 not in vertex2.neighbour_list:
                vertex1.neighbour_list.append(index2)
                vertex2.neighbour_list.append(index1)
                cv.line(visualised, (vertex1.x, vertex1.y), (vertex2.x, vertex2.y), Color.RED, thickness=2)
                cv.circle(visualised, (vertex1.x, vertex1.y), 4, Color.BLACK, thickness=cv.FILLED, lineType=8)
                cv.circle(visualised, (vertex2.x, vertex2.y), 4, Color.BLACK, thickness=cv.FILLED, lineType=8)

    return vertices_list, topology_backend, visualised


def fit_line(edge_contour: list) -> ([float, float], [float, float]):
    """
    Approximate edge contour with a straight line using contour operations.
    This is achieved by finding intersections of bounding rectangle and fitted line.
    x and y coordinates are calculated by solving system of linear equations that describe line and rectangle

    :param edge_contour: set of points that describe edge contour
    :return: two endpoints of an approximating line
    """
    rect_x, rect_y, width, height = cv.boundingRect(edge_contour)
    vec_x, vec_y, line_x, line_y = cv.fitLine(edge_contour, cv.DIST_L2, 0, 0.01, 0.01)
    points = []
    if abs(vec_x) <= 0.01:  # Detected line is vertical
        points.append((line_x, rect_y))
        points.append((line_x, rect_y + height))
    elif abs(vec_y) <= 0.01:    # Detected line is horizontal
        points.append((rect_x, line_y))
        points.append((rect_x + width, line_y))
    else:   # Detected line is oblique
        # Calculate intersection points on horizontal lines (x coordinates):
        # y = rect_y (top rectangle line)
        # y = rect_y + height (bottom rectangle line)
        top_intersect = ((rect_y - line_y) * vec_x / vec_y + line_x)
        bottom_intersect = ((rect_y + height - line_y) * vec_x / vec_y + line_x)
        # and vertical lines (y coordinates):
        # x = rect_x (left rectangle line)
        # x = rect_x + width (right rectangle line)
        left_intersect = ((rect_x - line_x) * vec_y / vec_x + line_y)
        right_intersect = ((rect_x + width - line_x) * vec_y / vec_x + line_y)

        # Find those 2 intersections that occur on rectangle line segments
        # horizontal:
        if rect_x <= top_intersect <= rect_x + width:
            points.append([top_intersect, rect_y])
        if rect_x <= bottom_intersect <= rect_x + width:
            points.append([bottom_intersect, rect_y + height])
        # vertical
        if rect_y <= left_intersect <= rect_y + height:
            points.append([rect_x, left_intersect])
        if rect_y <= right_intersect <= rect_y + height:
            points.append([rect_x + width, right_intersect])

    return np.array(points[0], dtype=float), np.array(points[1], dtype=float)


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
    min_distance = np.linalg.norm(point - current_center)
    for i in range(0, len(vertices_list)):
        current_vertex = vertices_list[i]
        current_center = np.array([current_vertex.x, current_vertex.y])
        distance = np.linalg.norm(point - current_center)
        if distance < min_distance:     # Found vertex closer to a point
            nearest_index = i
            min_distance = distance
    return nearest_index


def point_within_radius(point: np.ndarray, vertex: Vertex, radius_factor: float) -> bool:
    """
    Check if point is within vertex area with radius modified by factor (based on euclidean distance - L2)

    :param point: x and y coordinates
    :param vertex: vertex which area is considered
    :param radius_factor: factor to increase/decrease radius and therefor area
    :return: True if point is within radius, and False if it is not
    """
    radius = vertex.r * radius_factor
    center = np.array([vertex.x, vertex.y])
    if np.linalg.norm(point-center) <= radius:
        return True
    else:
        return False

