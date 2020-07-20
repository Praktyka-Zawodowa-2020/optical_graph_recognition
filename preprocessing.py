"""Module with operations that are used to preprocess a graph picture"""
import cv2 as cv
import numpy as np

# constants:
# for threshold function
GLOBAL_THRESH_FAILED = -1

# constant parameters:
# for reshape function
WIDTH_LIM: int = 1280  # px
HEIGHT_LIM: int = 800  # px
# for threshold function
MIN_BRIGHT_VAL: int = 110  # bright image lower limit of pixel value
MAX_FILL_RATIO: float = 0.14  # ratio of object pixels to all pixels


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
    binary, threshold_value = threshold(gray, MIN_BRIGHT_VAL, MAX_FILL_RATIO)

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
        cv.imshow("binary, th=" + str(threshold_value), binary)
        cv.imshow("filtered", filtered)

    return reshaped, binary, filtered


def reshape(image: np.ndarray, width_lim: int = 1280, height_lim: int = 800):
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
        # image size changed after rotation, we only need new width
        img_width = image.shape[1]

    width_factor = width_lim / img_width
    image = cv.resize(image, (0, 0), fx=width_factor, fy=width_factor)

    img_height = image.shape[0]     # Image height changed after rotation and first scaling
    if img_height > height_lim:     # scale again if new height is still too large
        height_factor = height_lim / img_height
        image = cv.resize(image, (0, 0), fx=height_factor, fy=height_factor)

    return image


def threshold(gray_image: np.ndarray, min_bright_value: int = 128, max_fill_ratio: float = 0.14):
    """
    Apply thresholding (binarization) to a grayscale image.
    Objects pixels are white and background pixels are black.

    :param gray_image: grayscale input image
    :param min_bright_value: lower limit for average pixel color value to consider image background bright
    :param max_fill_ratio: value that specifies success of global threshold (maximum object pixels to all pixels ratio)
    :return: binarized image, estimated global threshold if global thresholding succeeded
    if global thresholding failed and adaptive thresholding was used function returns -1 (GLOBAL_THRESH_FAILED)
    """

    # goal is to have white objects on black background
    # so if image is more bright perform inverse thresholding
    # and if image is more dark perform normal thresholding
    print(str(np.average(gray_image)))
    if np.average(gray_image) >= min_bright_value:  # bright image
        threshold_type = cv.THRESH_BINARY_INV
        sub_sign = 1    # for bright image we want to subtract constant value in adaptive thresholding
    else:   # dark image
        threshold_type = cv.THRESH_BINARY
        sub_sign = -1    # for bright image we want to add (subtract negative) constant value in adaptive thresholding

    # Perform adaptive global thresholding (OTSU)
    threshold_value, binary = cv.threshold(gray_image, 0, 255, threshold_type + cv.THRESH_OTSU)

    # Calculate fill ratio - number of object pixels (255, white) divided by number of all pixel (height * width)
    fill_ratio = np.count_nonzero(binary)/(binary.shape[0]*binary.shape[1])

    # global OTSU thresholding failed if resulted in to many object pixels
    if fill_ratio > max_fill_ratio:        # if it failed apply local adaptive thresholding
        threshold_value = GLOBAL_THRESH_FAILED
        binary = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type, 51, sub_sign*8)

    return binary, threshold_value


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
