import sys

import cv2 as cv

from preprocessing import preprocess
from segmentation import segment
from topology_recognition import recognize_topology
from postprocesing import graph6_format, graphml_format


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
        "paint_2.jpg",  # 8
        "paint_3.jpg",  # 9
        "paint_4.jpg",  # 10
        "article.png",
        "article_no_text.png"
    ]
    source = cv.imread("./graphs/" + file_names[file_index])
    # source = cv.imread("../../Praktyki2020/Resources/01.jpg")
    return source


def main(args):
    source = load_image(file_index=0)

    if source is not None:  # read successful, process image

        source, binary, preprocessed = preprocess(source, True)

        vertices_list, visualised = segment(source, binary, preprocessed, True)

        vertices_list = recognize_topology(vertices_list, preprocessed, visualised, True)

        cv.imshow("source", source)
        # display all windows until key is pressed
        cv.waitKey(0)
    else:
        print("Error opening image!")
        return -1


# if __name__ == "__main__":
main(sys.argv[1:])