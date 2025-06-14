#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:00:15 2025

@author: rpalomares
"""

import cv2
import numpy as np


def enhance_contrast(image, use_CLAHE=False):
    """
    Enhance the contrast and brightness of an image

    Parameters
    ----------
    image : Mat
        An image as it is returned by cv.imread
    use_CLAHE : Boolean, optional
        Use CLAHE method (slower) instead of value normalization.
        The default is False.

    Returns
    -------
    result : Mat
        The processed image.

    """

    result = None
    if use_CLAHE:
        # create a CLAHE object (Arguments are optional)
        # CLAHE stands for Contrast Limited Adaptative Histogram Equalization
        # This helps to enhance contrast in an image when different parts of it
        # present different levels of exposition
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        result = clahe.apply(image)
    else:
        norm_img = np.zeros((image.shape[0], image.shape[1]))
        result = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)

    return result


def denoise(image):
    """
    Perform image denoising using Non-local Means Denoising algorithm
    http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/

    Parameters
    ----------
    image : Mat
        An image as it is returned by cv.imread (or other cv functions).

    Returns
    -------
    result : Mat
        The processed image.

    """
    # Filter strength 10, template window size 7, search window size 15
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 15)


def binarize(image, use_OTSU=True):
    """
    Binarize image using Otsu's binarization algorithm

    Parameters
    ----------
    image : Mat
        An image as it is returned by cv.imread (or other cv functions).
    use_OTSU : Boolean, optional
        Use Otsu's binarization algorithm instead of
        GaussianAdaptativeThreshold. The default is True.

    Returns
    -------
    result : Mat
        The processed image.

    """

    result = None
    if use_OTSU:
        blur = cv2.GaussianBlur(image, (5,5), 0)
        _, result = cv2.threshold(blur, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        result = cv2.adaptiveThreshold(image, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    return result


def erode_dilate(image, kernel_size=3):
    """
    Performs erosion + dilation on image to enhance strokes

    Parameters
    ----------
    image : Mat
        An image as it is returned by cv.imread (or other cv functions).
    kernel_size : int, optional
        Kernel size. The default is 3.

    Returns
    -------
    result : Mat
        The processed image.

    """

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    result = cv2.erode(image, kernel, iterations = 1)
    result = cv2.dilate(result, kernel, iterations = 1)
    return result

def show_image(image, title="Result", ratio_factor=2000.0):
    """
    Displays a resized version of the image. Useful for quickly testing
    the effects of previous functions

    Parameters
    ----------
    image : Mat
        An image as it is returned by cv.imread (or other cv functions).
    title : string, optional
        A title for the window. The default is "Result".
    ratio_factor : float, optional
        A ratio factor to resize the image. For large images, bigger values
        will display a smaller version of the image. The default is 2000.0,
        which works well for images about 3x4 MPixels.

    Returns
    -------
    None.

    """

    ratio = image.shape[0] / ratio_factor
    image = cv2.resize(image, (int(image.shape[1] / ratio), 2000))
    cv2.imshow("Binarization", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
