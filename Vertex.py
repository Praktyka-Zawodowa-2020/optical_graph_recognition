"""
Module contains class representing vertex
"""


class Vertex:
    x = -1
    y = -1
    r = -1
    color = -1
    neighbour_list = []

    def __init__(self, x, y, r, color):
        self.x = x
        self.y = y
        self.r = r
        self.color = color
        self.neighbour_list = []
