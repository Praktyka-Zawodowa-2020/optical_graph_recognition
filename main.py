import os
import sys

import argparse
import cv2 as cv
from argsparser import parse_argument
from preprocessing import preprocess
from segmentation import segment
from topology_recognition import recognize_topology
from postprocesing import graph6_format, graphml_format
from shared import Mode

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
    # source = load_image(file_index=0)
    for i in range(74, 75):
        if i < 10:
            print("article", i, " ", i, end=" ")
            source = cv.imread("./article/0" + str(i) + ".jpg")
            mode=Mode.PRINTED
        elif i <= 50:
            print("article", i, " ", i, end=" ")
            source = cv.imread("./article/" + str(i) + ".jpg")
            mode = Mode.PRINTED
        elif 50 < i < 60:
            print("board", i-50, " ", i, end=" ")
            source = cv.imread("./board/0" + str(i-50) + ".jpg")
            mode = Mode.CLEAN_BG
        elif 60 <= i < 103:
            print("board", i-50, " ", i, end=" ")
            source = cv.imread("./board/" + str(i-50) + ".jpg")
            mode = Mode.CLEAN_BG
        elif 103 <= i < 152:
            print("notebook", i-102, " ", i, end=" ")
            source = cv.imread("./notebook/" + str(i-102) + ".jpg")
            mode = Mode.GRID_BG
        else:
            print("paint", i - 151, " ", i, end=" ")
            source = cv.imread("./paint/" + str(i - 151) + ".jpg")
            mode = Mode.CLEAN_BG

        if source is not None:  # read successful, process image
            file_name = str(i) + ".jpg"
            source, binary, preprocessed = preprocess(source, False,mode,i)
            vertices_list, visualised = segment(source, binary, preprocessed, False,mode,i)
            if len(vertices_list) == 0:
                print("No vertices found")
                return -1
            vertices_list = recognize_topology(vertices_list, preprocessed, visualised, False,i)
            graphml_format(vertices_list, "./tests/74")
            graph6_format(vertices_list, "./tests/74")
        # cv.imshow("source", source)
        # display all windows until key is pressed
        # cv.waitKey(0)
        else:
            print("Error opening image!")
            return -1


if __name__ == "__main__":
    main()
