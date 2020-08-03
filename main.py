import os
import sys

import cv2 as cv
import argparse
from preprocessing import preprocess
from segmentation import segment
from topology_recognition import recognize_topology
from postprocesing import graph6_format, graphml_format

parser = argparse.ArgumentParser("Optical graph recognition")
parser.add_argument("-p", "--path", help="Path to file")


# parser.add_argument("-b", "--background", help="TO DO. Choise background", default=1, choices=['1', '2'])
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
    return source


def parse_argument(file_path: str) -> (int,str):
    file_path.replace(" ", "")
    if 1 < file_path.count('.') < 0:
        print("File path is incorrect. Must be only one dot.")
        return 1,"File path is incorrect. Must be only one dot."
    head, tail = os.path.split(file_path)
    if len(tail) == 0:
        print("File name no exist")
        return 1,'1: File name no exist'

    file_name, file_ext = os.path.splitext(tail)
    if len(file_name) == 0:
        print("File name not found")
        return 1,'1: File name not found'
    save_path = head + '/' + file_name
    return 0,save_path


def main(args=None):
    # source = load_image(file_index=0)
    args = parser.parse_args()
    file_path = args.path
    code,save_path = parse_argument(file_path)

    if code == 1:
        return save_path

    source = cv.imread(file_path)
    if source is not None:  # read successful, process image

        source, binary, preprocessed = preprocess(source, False)

        vertices_list, visualised = segment(source, binary, preprocessed, False)
        if len(vertices_list) == 0:
            print("No vertices found")
            return "1: No vertices found"

        vertices_list = recognize_topology(vertices_list, preprocessed, visualised, False)
        graphml_format(vertices_list, save_path)
        graph6_format(vertices_list, save_path)


        # display all windows until key is pressed
        #cv.waitKey(0)
        return "0"
    else:
        print("Error opening image!")
        return "1: Error opening image!"


if __name__ == "__main__":
    main()