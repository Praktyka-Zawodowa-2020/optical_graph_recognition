"""
Module contains operations that preprocess a graph picture
"""
import cv2 as cv
import numpy as np


# constants
MIN_RESIZE = 960  # px


# performs all steps in preprocessing phase
def preprocess(source: np.ndarray, imshow_enabled: bool) -> (np.ndarray, np.ndarray, np.ndarray):
    # resizing if needed
    resize_factor = MIN_RESIZE/source.shape[1]
    source = source if source.shape[1] <= MIN_RESIZE else cv.resize(source, (0, 0), fx=resize_factor, fy=resize_factor)

    # converting to gray scale
    gray = cv.cvtColor(source, cv.COLOR_BGR2GRAY)

    # thresholding (binarization)
    threshold, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # removing noise
    kernel = np.ones((5, 5), np.uint8)
    filtered = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

    # display results of certain steps
    if imshow_enabled:
        cv.imshow("source " + str(source.shape[1]) + "x" + str(source.shape[0]), source)
        cv.imshow("binary, th=" + str(threshold), binary)
        cv.imshow("filtered", filtered)

    return source, binary, filtered
