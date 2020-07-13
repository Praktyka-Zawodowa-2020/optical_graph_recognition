"""Module contains class representing vertex"""


class Vertex:
    """
    Represent vertex recognised from the image.
    Attributes:
        x, y (int): coordinates of the center
        r (int): radius
        color (int): color in grayscale
        neighbour_list (list): list of connected vertices
    """
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
