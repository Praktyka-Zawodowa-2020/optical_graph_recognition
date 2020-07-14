"""Module with operations that are used to preprocess a graph picture"""
import cv2 as cv
import numpy as np

# constants
MIN_RESIZE: int = 960  # px


def preprocess(source: np.ndarray, imshow_enabled: bool) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Processes source image by resizing, thresholding and filering noise.
    :param source: input image
    :param imshow_enabled: flag determining to display (or not) preprocessing steps
    :return: resized, binarized, and filtered images.
    """
    # Resize if needed
    resize_factor = MIN_RESIZE / source.shape[1]
    source = source if source.shape[1] <= MIN_RESIZE else cv.resize(source, (0, 0), fx=resize_factor, fy=resize_factor)

    # converting image to gray scale
    gray = cv.cvtColor(source, cv.COLOR_BGR2GRAY)

    # Threshold (binarize) image
    threshold, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Remove some holes from image with closing operator (first dilates and then erodes)
    kernel = np.ones((5, 5), np.uint8)
    filtered = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

    # TODO - change parametrization of circles Transform in segmentation that works with this crop
    # Crop images to remove unnecessary background
    # object_pixels = cv.findNonZero(filtered)
    # x, y, w, h = cv.boundingRect(object_pixels)
    # source = source[y:y+h, x:x+w]
    # binary = binary[y:y+h, x:x+w]
    # filtered = filtered[y:y+h, x:x+w]

    # Display results of preprocessing steps
    if imshow_enabled:
        cv.imshow("source " + str(source.shape[1]) + "x" + str(source.shape[0]), source)
        cv.imshow("binary, th=" + str(threshold), binary)
        cv.imshow("filtered", filtered)

    return source, binary, filtered


def delete_characters(image: np.ndarray) -> np.ndarray:
    """
    Remove "characters" (noise) of small sizes
    :param image: Image after binarization
    :return: Image without noise
    """
    # findContours returns 3 variables for getting contours
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv.boundingRect(contour)

        # Ignore large counters
        if w > 60 and h > 60:
            continue

        # draw rectangle on original image
        cv.rectangle(image, (x, y), (x + w, y + h), 0, -1)

    return image
