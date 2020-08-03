import os
import sys

import argparse
import cv2 as cv
from argsparser import parse_argument
from preprocessing import preprocess
from segmentation import segment
from topology_recognition import recognize_topology
from postprocesing import graph6_format, graphml_format

parser = argparse.ArgumentParser("Optical graph recognition")
parser.add_argument("-p", "--path", help="Path to file")
parser.add_argument("-b", "--background",
                    help='''
                            GRID_BG - Hand drawn on grid/lined piece of paper (grid/lined notebook etc.) 
                            CLEAN_BG - Hand drawn on empty uniform color background 
                            PRINTED - Printed 
                        ''',
                    default='CLEAN_BG',
                    choices=['CLEAN_BG', 'GRID_BG', 'PRINTED']
                    )


def main(args=None):
    args = parser.parse_args()
    mode, file_path, save_path = parse_argument(args)

    if mode == -1 or len(save_path) == 0:
        return -1

    source = cv.imread(file_path)
    if source is not None:  # read successful, process image

        source, binary, preprocessed = preprocess(source, False, mode)

        vertices_list, visualised = segment(source, binary, preprocessed, False, mode)
        if len(vertices_list) == 0:
            print("1: No vertices found")
            return -1

        vertices_list = recognize_topology(vertices_list, preprocessed, visualised, False)
        graphml_format(vertices_list, save_path)
        graph6_format(vertices_list, save_path)

        # display all windows until key is pressed
        # cv.waitKey(0)
        print("0")
        return 0
    else:
        print("1: Error opening image!")
        return -1


if __name__ == "__main__":
    main()
