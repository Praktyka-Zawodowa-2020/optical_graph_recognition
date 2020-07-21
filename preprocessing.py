"""Module with operations that are used to preprocess a graph picture"""
import cv2 as cv
import numpy as np
from shared import Kernel, Color, Mode

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
# for contour noise filtering
NOISE_FACTOR: float = 0.0001  # px^2


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

    # Transform image (filter noise, remove grid, ...)
    transformed = transform(binary, threshold_value, Mode.PRINTED)  # TODO - mode from command line

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
        cv.imshow("transformed", transformed)

    return reshaped, binary, transformed


def reshape(image: np.ndarray, width_lim: int = 1280, height_lim: int = 800):
    """
    Scale image preserving original width to height ratio
    Do it so that its height and width are less or equal (and close to) given limits.
    Also if image is oriented horizontally, orient it vertically. (TODO - in final version remove rotation)

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
    if np.average(gray_image) >= min_bright_value:  # bright image
        thresh_type = cv.THRESH_BINARY_INV
        sub_sign = 1    # for bright image we want to subtract constant value in adaptive thresholding
    else:   # dark image
        thresh_type = cv.THRESH_BINARY
        sub_sign = -1    # for bright image we want to add (subtract negative) constant value in adaptive thresholding

    # Perform adaptive global thresholding (OTSU)
    threshold_value, binary = cv.threshold(gray_image, 0, Color.OBJECT, thresh_type + cv.THRESH_OTSU)

    # Calculate fill ratio - number of object pixels (255, white) divided by number of all pixel (height * width)
    fill_ratio = np.count_nonzero(binary)/(binary.shape[0]*binary.shape[1])

    # global OTSU thresholding failed if resulted in to many object pixels
    if fill_ratio > max_fill_ratio:        # if it failed apply local adaptive thresholding
        threshold_value = GLOBAL_THRESH_FAILED
        binary = cv.adaptiveThreshold(gray_image, Color.OBJECT, cv.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, 51, sub_sign*8)

    return binary, threshold_value


def transform(binary_image: np.ndarray, thresh_val: int, mode: Mode) -> np.ndarray:
    if mode == Mode.GRID_BG:
        if thresh_val != GLOBAL_THRESH_FAILED:
            transformed = filter_grid(binary_image, 40, 3, NOISE_FACTOR)
        else:
            transformed = filter_grid(binary_image, 30, 3, NOISE_FACTOR)
    elif mode == Mode.CLEAN_BG:
        transformed = cv.medianBlur(binary_image, 3)
        transformed = remove_contour_noise(transformed, NOISE_FACTOR/2.0)
        if thresh_val == GLOBAL_THRESH_FAILED:
            transformed = cv.morphologyEx(transformed, cv.MORPH_CLOSE, Kernel.k5)
    elif mode == Mode.PRINTED:
        transformed = cv.medianBlur(binary_image, 3)
        transformed = remove_contour_noise(transformed, NOISE_FACTOR/4.0)
    else:
        print("Mode is not supported!")
        transformed = np.copy(binary_image)
    return transformed


def remove_contour_noise(image: np.ndarray, max_noise_factor: float) -> np.ndarray:
    """
    Remove small closed contours from the image, that can be considered as a noise.
    Designed mainly to remove small dots that remain after removing the grid from the image.

    :param image: binary image
    :param max_noise_factor: factor to take part of image area as upper limit for noise area
    :return: Image with noise filtered out
    """
    contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for i in range(0, len(contours)):

        if cv.contourArea(contours[i]) <= image.shape[0]*image.shape[1]*max_noise_factor:
            # fill contour area with background color
            cv.drawContours(image, contours, contourIdx=i, color=0, thickness=cv.FILLED)

    return image


def filter_grid(binary_image: np.ndarray, min_distance: int, start_kernel: int, max_noise_factor: float) -> np.ndarray:
    binary = np.copy(binary_image)
    avg = avg_bg_distance(binary)
    if avg < min_distance:
        for i in range(0, 3):
            if avg < min_distance:
                image = cv.medianBlur(binary, start_kernel + i * 2)
                image = remove_contour_noise(image, max_noise_factor)
            else:
                break

            avg = avg_bg_distance(image)
    else:
        image = remove_contour_noise(binary, max_noise_factor)
    # remove remaining straight lines that go across whole image
    horizontal = remove_horizontal_grid(image)
    vertical = remove_vertical_grid(image)
    image = cv.bitwise_and(horizontal, vertical)

    return image


def avg_bg_distance(binary_image: np.ndarray) -> float:
    negative = cv.bitwise_not(binary_image)
    distance = cv.distanceTransform(negative, cv.DIST_L2, 3)
    width = distance.shape[1]
    height = distance.shape[0]
    mid_width = int(width / 2)
    mid_height = int(height / 2)
    avgs = [
        np.average(distance[0: mid_height, 0:mid_width]),
        np.average(distance[0:mid_height, mid_width:width]),
        np.average(distance[mid_height:height, 0:mid_width]),
        np.average(distance[mid_height:height, mid_width:width])
    ]
    return np.min(avgs)


def remove_horizontal_grid(binary_image: np.ndarray) -> np.ndarray:
    width = binary_image.shape[1]
    structure = cv.getStructuringElement(cv.MORPH_RECT, (width // 50, 1))
    horizontal = cv.erode(binary_image, structure)
    horizontal = cv.dilate(horizontal, structure)
    horizontal = cv.dilate(horizontal, Kernel.k3, iterations=1)

    lines = cv.HoughLines(horizontal, 1, np.pi / 180, 500)
    horizontal_mask = np.zeros((horizontal.shape[0], horizontal.shape[1]), np.uint8)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1500 * (-b))
            y1 = int(y0 + 1500 * (a))
            x2 = int(x0 - 1500 * (-b))
            y2 = int(y0 - 1500 * (a))
            cv.line(horizontal_mask, (x1, y1), (x2, y2), 255, 7)
    masked = cv.bitwise_and(horizontal, horizontal_mask)
    masked = cv.erode(masked, Kernel.k3, iterations=1)
    lines = cv.HoughLinesP(masked, 1, np.pi / 180, threshold=20, minLineLength=width // 25, maxLineGap=10)
    image = np.copy(binary_image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(image, (x1, y1), (x2, y2), 0, 2)

    return image


def remove_vertical_grid(binary_image: np.ndarray) -> np.ndarray:

    height = binary_image.shape[0]
    structure = cv.getStructuringElement(cv.MORPH_RECT, (1, height//50))
    vertical = cv.erode(binary_image, structure)
    vertical = cv.dilate(vertical, structure)

    vertical = cv.dilate(vertical, Kernel.k3, iterations=1)
    lines = cv.HoughLines(vertical, 1, np.pi/180, 400)
    vertical_mask = np.zeros((vertical.shape[0], vertical.shape[1]), np.uint8)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1500 * (-b))
            y1 = int(y0 + 1500 * (a))
            x2 = int(x0 - 1500 * (-b))
            y2 = int(y0 - 1500 * (a))
            cv.line(vertical_mask, (x1, y1), (x2, y2), 255, 7)
    masked = cv.bitwise_and(vertical, vertical_mask)
    masked = cv.erode(masked, Kernel.k3, iterations=1)
    lines = cv.HoughLinesP(masked, 1, np.pi / 180, threshold=30, minLineLength=height // 25, maxLineGap=10)
    image = np.copy(binary_image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(image, (x1, y1), (x2, y2), 0, 2)

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
