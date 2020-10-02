"""Module containing global constants, functions, ..."""
import numpy as np
from math import sqrt


class Mode:
    """Input mode indicates visual properties of given graph photo"""
    HELP = '''
        Input mode indicates visual properties of given graph photo:
        grid_bg - Hand drawn graph on grid/lined piece of paper (grid/lined notebook etc.)
        clean_bg - Hand drawn graph on empty uniform color background (on board, empty piece of paper, editor (paint))
        printed - Graph from a printed source (e.g. from a paper, a publication, a book, etc.)
        auto - Mode is chosen automatically between grid_bg and clean_bg modes
    '''
    CHOICES = ['grid_bg', 'clean_bg', 'printed', 'auto']
    DEFAULT = 'auto'

    GRID_BG = CHOICES.index('grid_bg')
    CLEAN_BG = CHOICES.index('clean_bg')
    PRINTED = CHOICES.index('printed')
    AUTO = CHOICES.index('auto')

    @staticmethod
    def get_mode(cli_arg: str):
        """
        Resolves Mode code from command line input string
        :param cli_arg: command line argument indicating Mode for processing
        :return: Mode for processing
        """
        for i in range(0, len(Mode.CHOICES)):
            if cli_arg == Mode.CHOICES[i]:
                return i

        # invalid cli_arg value
        print("1: Mode \""+cli_arg+"\" is not a viable mode.")
        return -1


class Debug:
    """Debug mode indicates how much debugging information will be displayed"""
    HELP = '''
        Debug mode indicates how much debugging information will be displayed:
        - no - no windows with debugging information are displayed
        - general - only windows with general debugging information are displayed
        - full - all windows with debugging information are displayed
    '''
    CHOICES = ['no', 'general', 'full']
    DEFAULT = 'no'

    NO = CHOICES.index('no')
    GENERAL = CHOICES.index('general')
    FULL = CHOICES.index('full')

    @staticmethod
    def get_debug(cli_arg: str):
        """
        Resolves debugging mode code from command line input string
        :param cli_arg: command line argument indicating debugging mode
        :return: debugging mode code
        """
        for i in range(0, len(Debug.CHOICES)):
            if cli_arg == Debug.CHOICES[i]:
                return i
        # invalid cli_arg value
        print("1: Debugging mode \""+cli_arg+"\" is not a viable one.")
        return -1


class Color:
    """Logical colors used for processing and physical used for debugging purposes"""
    # Logical
    OBJECT = 255
    BG = 0

    # Physical (BGR)
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLACK = (0, 0, 0)
    GRAY = (127, 127, 127)
    WHITE = (255, 255, 255)
    YELLOW = (0, 255, 255)
    ORANGE = (0, 140, 255)
    PURPLE = (127, 0, 127)
    PINK = (255, 0, 255)
    CYAN = (255, 255, 0)


class Kernel:
    """Simple kernels (square matrices of ones) with different sizes used throughout processing"""
    k3 = np.ones((3, 3), dtype=np.uint8)
    k5 = np.ones((5, 5), dtype=np.uint8)
    k7 = np.ones((7, 7), dtype=np.uint8)


def distance_L2(point1: [float, float], point2: [float, float]) -> float:
    """
    Calculate euclidean (L2) distance between 2 points given in cartesian coordinate system

    :param point1:
    :param point2:
    :return: distance (euclidean - L2) between points'
    """
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
