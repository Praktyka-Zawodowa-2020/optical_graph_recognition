"""Module containing global constants, functions, ..."""
import numpy as np


class Mode:
    """Input mode indicates visual properties of given graph photo, see HELP below for details"""
    HELP = '''
        grid_bg - Hand drawn graph on grid/lined piece of paper (grid/lined notebook etc.) 
        clean_bg - Hand drawn graph on empty uniform color background (on board, empty piece of paper, editor (paint))
        printed - Printed graph (e.g. from paper, publication, book...)
        auto - Mode is chosen automatically between GRID_BG and CLEAN_BG    
    '''
    CHOICES = ['grid_bg', 'clean_bg', 'printed', 'auto']
    DEFAULT = 'clean_bg'

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
        no - no windows with debugging information are displayed
        general - only windows with general debugging information are displayed
        full - all windows with debugging information are displayed
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
