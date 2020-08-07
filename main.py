import cv2 as cv
from argsparser import parser, parse_argument
from preprocessing import preprocess
from segmentation import segment
from topology_recognition import recognize_topology
from postprocesing import postprocess


def main():
    args = parser.parse_args()
    mode, file_path, save_path = parse_argument(args)

    if mode == -1 or len(save_path) == 0:
        return -1

    source = cv.imread(file_path)
    if source is not None:  # read successful, process image

        # 1st step - preprocessing
        source, preprocessed, mode = preprocess(source, False, mode)

        # 2nd step - segmentation
        vertices_list, visualised, preprocessed = segment(source, preprocessed, False, mode)
        if len(vertices_list) == 0:
            print("1: No vertices found")
            return -1

        # 3rd step - topology recognition
        vertices_list = recognize_topology(vertices_list, preprocessed, visualised, False)

        # 4th step - postprocessing
        postprocess(vertices_list, save_path)

        print("0")
        return 0
    else:
        print("1: Error opening image!")
        return -1


if __name__ == "__main__":
    main()
