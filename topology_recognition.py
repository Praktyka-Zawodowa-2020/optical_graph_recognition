"""Module with operations that are used to recognize topology of a graph"""
import cv2 as cv
import numpy as np
import math

from Vertex import Vertex

# constants
MIN_EDGE_LEN = 5  # px
RADIUS_FACTOR: float = 3.0


def recognize_topology(vertices_list: list, preprocessed: np.ndarray, visualised: np.ndarray):
    for vertex in vertices_list:
        cv.circle(preprocessed, (vertex.x, vertex.y), round(vertex.r*1.6), color=0, thickness=cv.FILLED, lineType=8)
        # cv.circle(visualised, (vertex.x, vertex.y), round(vertex.r*RADIUS_FACTOR), (255, 0, 0), thickness=cv.FILLED, lineType=8)

    cv.imshow("deleted vertices", preprocessed)

    contours, _ = cv.findContours(preprocessed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        pt1, pt2 = fit_line(cnt)
        # cv.line(visualised, pt1, pt2, (0, 0, 255), thickness=2)
        np_pt1 = np.array([pt1[0], pt1[1]])
        np_pt2 = np.array([pt2[0], pt2[1]])
        if np.linalg.norm(np_pt1 - np_pt2) > MIN_EDGE_LEN:
            index1 = find_nearest_vertex(pt1, vertices_list)
            vertex1 = vertices_list[index1]
            index2 = find_nearest_vertex(pt2, vertices_list)
            vertex2 = vertices_list[index2]
            if point_within_radius(pt1, vertex1, RADIUS_FACTOR) and point_within_radius(pt2, vertex2, RADIUS_FACTOR):
                cv.line(visualised, (vertex1.x, vertex1.y), (vertex2.x, vertex2.y), (0, 0, 255), thickness=2)
                cv.circle(visualised, (vertex1.x, vertex1.y), 4, (0, 0, 0), thickness=cv.FILLED,
                          lineType=8)
                cv.circle(visualised, (vertex2.x, vertex2.y), 4, (0, 0, 0), thickness=cv.FILLED,
                          lineType=8)
    # cv.drawContours(visualised, contours, -1, (127, 0, 127), 2)

    cv. imshow("visualised", visualised)


def fit_line(edge_contour):
    left_bound = math.inf
    right_bound = -1
    top_bound = math.inf
    bottom_bound = -1
    for i in range(0, len(edge_contour)):
        # x axis
        if edge_contour[i][0][0] <= left_bound:
            left_bound = edge_contour[i][0][0]
            left_point = (edge_contour[i][0][0], edge_contour[i][0][1])
        if edge_contour[i][0][0] >= right_bound:
            right_bound = edge_contour[i][0][0]
            right_point = (edge_contour[i][0][0], edge_contour[i][0][1])
        # y axis
        if edge_contour[i][0][1] <= top_bound:
            top_bound = edge_contour[i][0][1]
            top_point = (edge_contour[i][0][0], edge_contour[i][0][1])
        if edge_contour[i][0][1] >= bottom_bound:
            bottom_bound = edge_contour[i][0][1]
            bottom_point = (edge_contour[i][0][0], edge_contour[i][0][1])

    x_distance = np.linalg.norm(np.array([left_point[0], left_point[1]]) - np.array(right_point[0], right_point[1]))
    y_distance = np.linalg.norm(np.array([top_point[0], top_point[1]]) - np.array(bottom_point[0], bottom_point[1]))

    return (left_point, right_point) if x_distance > y_distance else (top_point, bottom_point)


def find_nearest_vertex(point: (int, int), vertices_list: list):
    point = np.array([point[0], point[1]])
    nearest_index = 0
    current_vertex = vertices_list[nearest_index]
    current_center = np.array([current_vertex.x, current_vertex.y])
    min_distance = np.linalg.norm(point - current_center)
    for i in range(0, len(vertices_list)):
        current_vertex = vertices_list[i]
        current_center = np.array([current_vertex.x, current_vertex.y])
        distance = np.linalg.norm(point - current_center)
        if distance < min_distance:
            nearest_index = i
            min_distance = distance
    return nearest_index

def point_within_radius(point: (int, int), vertex: Vertex, radius_factor):
    radius = vertex.r * radius_factor
    point = np.array([point[0], point[1]])
    center = np.array([vertex.x, vertex.y])
    if np.linalg.norm(point-center) <= radius:
        return True
    else:
        return False

