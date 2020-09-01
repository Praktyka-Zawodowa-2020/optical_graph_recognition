"""Module contains class representing vertex"""


class Vertex:
    """
    Represent vertex recognised from the image.

    Attributes:
        x, y (int): coordinates of the center
        r (float): radius
        is_filled (bool): flag indicating if vertex is filled
        color (int, int, int): bgr color
        adjacency_list (list): list of adjacent (connected) vertices
    """
    id = -1
    x = -1
    y = -1
    r = -1.0
    is_filled = False
    color = (-1, -1, -1)
    adjacency_list = []

    def __init__(self, x, y, r, is_filled, color):
        self.x = x
        self.y = y
        self.r = r
        self.is_filled = is_filled
        self.color = color
        self.adjacency_list = []
