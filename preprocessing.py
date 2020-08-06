"""Module with operations that are used to preprocess a graph picture"""
import cv2 as cv
import numpy as np
import math
from shared import Kernel, Color, Mode

MAX_R_FACTOR: float = 0.035
MIN_R_FACTOR: float = 0.005
DIST_FACTOR: float = 0.06
INNER_CANNY: int = 200
CIRCLE_THRESHOLD: int = 13
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
# for automatic mode choice
MODE_THRESHOLD: float = 10  # min average distance of bg pixels to object pixels to consider image background clean


def preprocess(source: np.ndarray, imshow_enabled: bool, mode: int) -> (np.ndarray, np.ndarray, Mode):
    """
    Processes source image by reshaping, thresholding, transforming and cropping.
    If auto mode has been selected, choose mode automatically

    :param source: input image
    :param imshow_enabled: flag determining to display (or not) preprocessing steps
    :param mode: GRID_BG, CLEAN_BG, PRINTED
    :return: reshaped and fully preprocessed images, changed (if mode is AUTO) or unchanged mode
    """
    # Reshape image to standard resolution
    reshaped = reshape(source, WIDTH_LIM, HEIGHT_LIM)

    # Convert image to gray scale
    gray = cv.cvtColor(reshaped, cv.COLOR_BGR2GRAY)

    # Threshold (binarize) image
    binary, threshold_value = threshold(gray, MIN_BRIGHT_VAL, MAX_FILL_RATIO)

    # choose mode automatically if selected AUTO mode
    if mode == Mode.AUTO:
        mode = chose_mode(binary)

    # Transform image (filter noise, remove grid, ...)
    transformed = transform(binary, threshold_value, mode)

    # Remove unnecessary background
    transformed, [reshaped, binary] = crop_bg_padding(transformed, [reshaped, binary])

    # Remove characters
    # without_chars = delete_characters(transformed)  # TODO - mode from command line

    # Display results of preprocessing steps
    if imshow_enabled:
        cv.imshow("reshaped source " + str(reshaped.shape[1]) + "x" + str(reshaped.shape[0]), reshaped)
        cv.imshow("binary, th=" + str(threshold_value), binary)
        cv.imshow("transformed", transformed)
        # cv.imshow("Chars deleted", without_chars)

    return reshaped, transformed, mode


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
    if img_height > img_width:  # If image is oriented horizontally - rotate to orient vertically
        image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
        # image size changed after rotation, we only need new width
        img_width = image.shape[1]

    width_factor = width_lim / img_width
    image = cv.resize(image, (0, 0), fx=width_factor, fy=width_factor)

    img_height = image.shape[0]  # Image height changed after rotation and first scaling
    if img_height > height_lim:  # scale again if new height is still too large
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
        sub_sign = 1  # for bright image we want to subtract constant value in adaptive thresholding
    else:  # dark image
        thresh_type = cv.THRESH_BINARY
        sub_sign = -1  # for bright image we want to add (subtract negative) constant value in adaptive thresholding

    # Perform adaptive global thresholding (OTSU)
    threshold_value, binary = cv.threshold(gray_image, 0, Color.OBJECT, thresh_type + cv.THRESH_OTSU)

    # Calculate fill ratio - number of object pixels (255, white) divided by number of all pixel (height * width)
    fill_ratio = np.count_nonzero(binary) / (binary.shape[0] * binary.shape[1])

    # global OTSU thresholding failed if resulted in to many object pixels
    if fill_ratio > max_fill_ratio:  # if it failed apply local adaptive thresholding
        threshold_value = GLOBAL_THRESH_FAILED
        binary = cv.adaptiveThreshold(gray_image, Color.OBJECT, cv.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, 51,
                                      sub_sign * 8)

    return binary, threshold_value


def chose_mode(binary_image: np.ndarray) -> Mode:
    """
    Chose mode automatically based on average background pixel distance from object pixels
    For pictures with grid distance is much smaller than when the picture has clean background
    :param binary_image: input binary image
    :return: input mode for further processing, see shared.py for modes descriptions
    """
    filtered = cv.medianBlur(binary_image, 3)
    avg = avg_bg_distance(filtered)
    if avg <= MODE_THRESHOLD:
        return Mode.GRID_BG
    else:
        return Mode.CLEAN_BG


def transform(binary_image: np.ndarray, thresh_val: int, mode: int) -> np.ndarray:
    """
    Filter image from noise (and sometimes grid) by performing various transformations depending on mode parameter.

    :param binary_image: input binary image
    :param thresh_val: value from thresholding, indicates if global thresholding had been successful
    :param mode: indicates properties of input photo (see shared.py for details)
    :return: Transformed (filtered) image
    """
    if mode == Mode.GRID_BG:  # grid and noise are filtered from the picture
        if thresh_val != GLOBAL_THRESH_FAILED:
            transformed = filter_grid(binary_image, 40, 3, NOISE_FACTOR)
        else:
            transformed = filter_grid(binary_image, 30, 3, NOISE_FACTOR)
    elif mode == Mode.CLEAN_BG:  # medium size noise is filtered
        transformed = cv.medianBlur(binary_image, 3)

        transformed = remove_contour_noise(transformed, NOISE_FACTOR / 2.0)
        if thresh_val == GLOBAL_THRESH_FAILED:  # also remove pepper noise if local thresholding was applied
            transformed = cv.morphologyEx(transformed, cv.MORPH_CLOSE, Kernel.k5)
    elif mode == Mode.PRINTED:  # only salt noise is filtered
        transformed = cv.medianBlur(binary_image, 3)
        transformed = remove_contour_noise(transformed, NOISE_FACTOR / 4.0)
    else:
        print("Mode is not supported! Transformation makes no changes.")
        transformed = np.copy(binary_image)
    return transformed


def remove_contour_noise(image: np.ndarray, max_noise_factor: float) -> np.ndarray:
    """
    Remove small closed contours from the image, that can be considered noise.
    Contour is classified as noise if its area is not greater than image area multiplied by tiny given factor.

    :param image: binary image
    :param max_noise_factor: tiny factor to take part of image area as upper limit for noise area
    :return: Image with noise filtered out
    """
    contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for i in range(0, len(contours)):

        if cv.contourArea(contours[i]) <= image.shape[0] * image.shape[1] * max_noise_factor:
            # fill contour area with background color
            cv.drawContours(image, contours, contourIdx=i, color=0, thickness=cv.FILLED)

    return image


def filter_grid(binary_image: np.ndarray, min_distance: int, start_kernel: int, max_noise_factor: float) -> np.ndarray:
    """
    Filters grid from the image
    Such filtering is based on a fact that vertices should be thicker than grid
    To achieve grid filtering an optimal kernel size for medianBlur is found

    :param binary_image: input binary image
    :param min_distance: minimal average distance of background pixels from object pixels to consider grid filtered out
    :param start_kernel: minimal kernel size for median filter
    :param max_noise_factor: see remove contour noise function description
    :return: image with grid filtered out
    """
    binary = np.copy(binary_image)
    avg = avg_bg_distance(binary)
    if avg < min_distance:  # small average distance indicates that a lot of grid remained in the picture
        # Loop below searches for optimal kernel size for median filter
        # It achieves that, by applying median filters and checking if background distance has reached acceptable level
        for i in range(0, 4):
            if avg < min_distance:
                image = cv.medianBlur(binary, start_kernel + i * 2)
                image = remove_contour_noise(image, max_noise_factor)
            else:
                break

            avg = avg_bg_distance(image)
    else:  # for very clear images only contour filtering is applied
        image = remove_contour_noise(binary, max_noise_factor)

    # remove remaining straight (vertical and horizontal) lines
    # usually the remaining lines are margins which are thicker than the rest of the grid an therefor are not filtered
    horizontal = remove_horizontal_grid(image)
    vertical = remove_vertical_grid(image)
    # include removed pixels from both images
    image = cv.bitwise_and(horizontal, vertical)

    return image


def avg_bg_distance(binary_image: np.ndarray) -> float:
    """
    Calculate average distance from background pixel to nearest object pixel
    Calculations are applied for 4 regions of image (top, bottom)x(left, right) and minimal average distance is returned

    :param binary_image: input binary image
    :return: average distance form background pixel to another
    """
    negative = cv.bitwise_not(binary_image)
    distance = cv.distanceTransform(negative, cv.DIST_L2, 3)
    width = distance.shape[1]
    height = distance.shape[0]
    mid_width = int(width / 2)
    mid_height = int(height / 2)
    averages = [
        np.average(distance[0: mid_height, 0:mid_width]),
        np.average(distance[0:mid_height, mid_width:width]),
        np.average(distance[mid_height:height, 0:mid_width]),
        np.average(distance[mid_height:height, mid_width:width])
    ]
    return np.min(averages)


# TODO - change grid function with HoughLines to only remove margins
def remove_horizontal_grid(binary_image: np.ndarray) -> np.ndarray:
    """
    Remove horizontal grid lines (long lines that go across image y axis)

    :param binary_image: input image
    :return: image without horizontal grid
    """
    # 1st step - extract horizontally aligned pixels from image using structuring element
    width = binary_image.shape[1]
    structure = cv.getStructuringElement(cv.MORPH_RECT, (width // 50, 1))
    horizontal = cv.erode(binary_image, structure)
    horizontal = cv.dilate(horizontal, structure)
    horizontal = cv.dilate(horizontal, Kernel.k3, iterations=1)

    # 2nd step - fit straight lines across whole image to create a mask
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
            y1 = int(y0 + 1500 * a)
            x2 = int(x0 - 1500 * (-b))
            y2 = int(y0 - 1500 * a)
            cv.line(horizontal_mask, (x1, y1), (x2, y2), 255, 7)
    # 3rd step - apply mask to image so that only lines that go across all pictures width remain
    masked = cv.bitwise_and(horizontal, horizontal_mask)
    masked = cv.erode(masked, Kernel.k3, iterations=1)
    # 4th step - find reasonably long lines on masked image and remove them from original image
    lines = cv.HoughLinesP(masked, 1, np.pi / 180, threshold=20, minLineLength=width // 25, maxLineGap=10)
    image = np.copy(binary_image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(image, (x1, y1), (x2, y2), 0, 2)

    return image


def remove_vertical_grid(binary_image: np.ndarray) -> np.ndarray:
    """
    Remove vertical grid lines (long lines that go across image x axis)

    :param binary_image: input image
    :return: image without vertical grid
    """
    # 1st step - extract vertically aligned pixels from image using structuring element

    height = binary_image.shape[0]
    structure = cv.getStructuringElement(cv.MORPH_RECT, (1, height // 50))
    vertical = cv.erode(binary_image, structure)
    vertical = cv.dilate(vertical, structure)
    vertical = cv.dilate(vertical, Kernel.k3, iterations=1)

    # 2nd step - fit straight lines across whole image to create a mask
    lines = cv.HoughLines(vertical, 1, np.pi / 180, 400)

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
    # 3rd step - apply mask to image so that only lines that go across all pictures height remain
    masked = cv.bitwise_and(vertical, vertical_mask)
    masked = cv.erode(masked, Kernel.k3, iterations=1)
    # 4th step - find reasonably long lines on masked image and remove them from original image
    lines = cv.HoughLinesP(masked, 1, np.pi / 180, threshold=30, minLineLength=height // 25, maxLineGap=10)
    image = np.copy(binary_image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(image, (x1, y1), (x2, y2), 0, 2)

    return image


def crop_bg_padding(binary_transformed: np.ndarray, images: list, padding: int = 15) -> (np.ndarray, list):
    """
    Crop images to remove background padding (unnecessary background surrounding graph)
    Usually some padding remains, so morphological operations (e.g. erosion) work properly

    :param binary_transformed: binarized and transformed (noise filtered) input image
    :param images: all other images
    :param padding: size of padding that remain in the picture
    :return: cropped images in the same order as in input - separately: transformed, and all other in list
    """
    # Find all object pixels in transformed image
    object_pixels = cv.findNonZero(binary_transformed)
    # Find minimal rectangular area in transformed image, that includes all object pixels - area without padding
    x, y, width, height = cv.boundingRect(object_pixels)
    # leave padding of given size if possible
    image_h, image_w = binary_transformed.shape[:]
    left = x - padding if (x - padding) > 0 else 0
    right = x + width + padding if (x + width + padding) < image_w else image_w
    top = y - padding if (y - padding) > 0 else 0
    bottom = y + height + padding if (y + height + padding) < image_h else image_h

    # Crop all given images to new minimal size - removes padding
    binary_transformed = binary_transformed[top:bottom, left:right]
    if images is not None:
        for i in range(0, len(images)):
            images[i] = images[i][top:bottom, left:right]
    return binary_transformed, images


def delete_characters(transformed: np.ndarray) -> np.ndarray:
    """
    Remove "characters" (noise) of small sizes

    :param transformed: binarized and transformed input image
    :return: Image without noise
    """
    image = transformed.copy()
    height, width = image.shape[:2]

    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        wider_clipping = False
        hist2 = []
        # get rectangle bounding contour
        [x, y, w, h] = cv.boundingRect(contour)

        # if possible cut out the contour wider and longer than found
        if 1 < x and (x + w + 4) < width and 1 < y and (y + h + 4) < height:
            x = x - 2
            y = y - 2
            w = w + 4
            h = h + 4
            wider_clipping = True
        crop_image = image[y: y + h, x: x + w].copy()

        # if empty vertices of the graph have a thick edge,
        # the findContours function draws the contours inside and outside the vertex.
        # When calculating the average distance of white pixels from the center of the cut image,
        # we filter out the contours detected inside the vertex
        white_img = np.zeros([h, w, 1], dtype=np.uint8)
        white_img.fill(255)
        white_img[int(h / 2)][int(w / 2)] = 0
        dst = cv.distanceTransform(white_img, cv.DIST_C, 3, cv.DIST_LABEL_PIXEL)
        avarage = cv.mean(dst, mask=crop_image)

        if avarage[0] < 0.4 * ((h + w) / 2):

            # sometimes when the vertex contour is thin,
            # "cv.HoughCircles" does not detect it,
            # so when counting the histogram for the eroded image,
            # we ignore such vertices (unfortunately such a filter also leaves noises)
            cv.rectangle(crop_image, (0, 0), (w - 1, h - 1), 0, 1)
            eroded = cv.erode(crop_image, Kernel.k3, iterations=1)
            hist = cv.calcHist([eroded], [0], None, [256], [0, 256])
            if hist[255] / (hist[255] + hist[0]) > 0.005:
                # recognition of vertices in the cut image
                circles = cv.HoughCircles(crop_image, cv.HOUGH_GRADIENT, 1, 20,
                                          param1=30,
                                          param2=20,
                                          minRadius=0,
                                          maxRadius=0)
                if circles is not None:
                    continue

                # recognition of lines in the cut image
                lines = cv.HoughLinesP(crop_image, 2, np.pi / 180, 40, 0, 0)
                is_edge = False
                if lines is not None:
                    black_img = np.zeros([h, w], dtype=np.uint8)
                    for j in range(0, len(lines)):
                        x1 = lines[j][0][0]
                        y1 = lines[j][0][1]
                        x2 = lines[j][0][2]
                        y2 = lines[j][0][3]
                        length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                        if length > 15 and w * h > 625:
                            is_edge = True
                            cv.line(black_img, (x1, y1), (x2, y2), 255, 2)
                    # we calculate the difference of the image cut out and with the lines marked,
                    # and then the histogram of the resulting image
                    # such an algorithm allows you to filter out contours containing the edges of the graph
                    sub_image = crop_image-black_img
                    hist2 = cv.calcHist([sub_image], [0], None, [256], [0, 256])

                if (is_edge is True and hist2[255] / (hist2[255] + hist2[0]) > 0.08) or is_edge is False:
                    cv.drawContours(image, [contour], -1, 0, -1)

    return image
