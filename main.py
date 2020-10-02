"""Main module containing script entry - main function"""
import cv2 as cv

from shared import Debug
from argsparser import parser, parse_argument
from preprocessing import preprocess
from segmentation import segment
from topology_recognition import recognize_topology
from postprocessing import postprocess


def main():
    args = parser.parse_args()
    mode, debug, file_path, save_path = parse_argument(args)

    if mode == -1 or debug == -1 or len(save_path) == 0:
        print("1: Error reading input arguments!")
        return -1

    source = cv.imread(file_path)
    if source is not None:  # read successful, process image

        # 1st step - preprocessing
        source, preprocessed, mode, is_rotated = preprocess(source, mode, debug)

        # 2nd step - segmentation
        vertices_list, visualised, preprocessed, edge_thickness = segment(source, preprocessed, mode, debug)
        if len(vertices_list) == 0:
            print("1: No vertices found")
            return -1

        # 3rd step - topology recognition
        vertices_list = recognize_topology(vertices_list, preprocessed, visualised, edge_thickness, mode, debug)

        # 4th step - postprocessing
        postprocess(vertices_list, save_path, is_rotated)

        # if displaying debug info has been enabled keep displayed windows open until key is pressed
        if debug != Debug.NO:
            cv.waitKey(0)

        print("0")
        return 0
    else:
        print("1: Error opening image!")
        return -1


if __name__ == "__main__":
    main()
