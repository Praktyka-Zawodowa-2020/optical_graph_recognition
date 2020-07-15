import sys

import cv2 as cv

from preprocessing import preprocess
from segmentation import segment
from postprocesing import graph6_format


def load_image(file_index):
    file_names = [
        "notebook_black_1.jpg",  # 0
        "notebook_blue_1.jpg",  # 1
        "notebook_gray_1.jpg",  # 2
        "tablica.jpg",  # 3
        "tablica2.jpg",  # 4
        "tablica3.jpg",  # 5
        "tablet_graficzny.png",  # 6
        "paint_1.jpg",  # 7
    ]
    source = cv.imread("./graphs/" + file_names[file_index])
    return source


def main(args):
    source = load_image(file_index=0)

    if source is not None:  # read successful, process image

        source, binary, preprocessed = preprocess(source, True)

        vertices_list = segment(source, binary, preprocessed, True)
        graph6_format(vertices_list)
        # display all windows until key is pressed
        cv.waitKey(0)
        return 0
    else:
        print("Error opening image!")
        return -1


if __name__ == "__main__":
    main(sys.argv[1:])
