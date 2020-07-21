"""Module containing global constants, functions, ..."""

import numpy as np


class Mode:
    # Input mode indicates visual properties of given graph photo
    GRID_BG = 1     # Hand drawn on grid/lined piece of paper (grid/lined notebook etc.)
    CLEAN_BG = 2    # Hand drawn on empty uniform color background (on board, empty piece of paper, editor (paint))
    PRINTED = 3     # Printed (e.g. from paper, publication, book...)


class Color:
    # Logical colors
    OBJECT = 255
    BG = 0

    # Physical colors - BGR
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLACK = (0, 0, 0)
    GRAY = (127, 127, 127)
    WHITE = (255, 255, 255)
    YELLOW = (0, 255, 255)
    ORANGE = (0, 140, 255)


class Kernel:
    k3 = np.ones((3, 3), dtype=np.uint8)
    k5 = np.ones((5, 5), dtype=np.uint8)
    k7 = np.ones((7, 7), dtype=np.uint8)
