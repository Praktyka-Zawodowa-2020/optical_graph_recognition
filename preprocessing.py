"""Module with operations that are used to preprocess a graph picture"""
import cv2 as cv
import numpy as np

# constants
WIDTH_LIM: int = 1280  # px
HEIGHT_LIM: int = 800


def preprocess(source: np.ndarray, imshow_enabled: bool) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Processes source image by reshaping, thresholding and noise filering.
    :param source: input image
    :param imshow_enabled: flag determining to display (or not) preprocessing steps
    :return: reshaped, binarized, and filtered images.
    """
    # reshape to standard orientation, and resolution
    reshaped = reshape(source, WIDTH_LIM, HEIGHT_LIM)

    # converting image to gray scale
    gray = cv.cvtColor(reshaped, cv.COLOR_BGR2GRAY)

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
        cv.imshow("reshaped source " + str(reshaped.shape[1]) + "x" + str(reshaped.shape[0]), reshaped)
        cv.imshow("binary, th=" + str(threshold), binary)
        cv.imshow("filtered", filtered)

    return reshaped, binary, filtered


def reshape(image: np.ndarray, width_lim: int, height_lim: int):
    """
    Scale image preserving original width to height ratio
    Do it so that its height and width are less or equal (and close to) given limits.
    Also if image is oriented horizontally, orient it vertically. (TODO - is reshaping breaking OCR?)
    :param image: input image
    :param width_lim: limit for width
    :param height_lim: limit for height
    :return: reshaped (scaled and rotated if needed) image
    """
    img_width = image.shape[1]
    img_height = image.shape[0]
    if img_height > img_width:      # If image is oriented horizontally - rotate to orient vertically
        image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)

    width_factor = width_lim / img_width
    image = cv.resize(image, (0, 0), fx=width_factor, fy=width_factor)

    img_res_height = image.shape[0]     # height after first scaling
    if img_res_height > height_lim:     # scale again if new height is still too large
        height_factor = height_lim / img_res_height
        image = cv.resize(image, (0, 0), fx=height_factor, fy=height_factor)

    return image


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
