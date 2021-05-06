# ---------- imports ---------- #
import numpy as np
from matplotlib import pyplot as plt
from imageio import imread, imwrite
from skimage.color import rgb2gray
import cv2
import imutils

# ---------- macros ---------- #
TWO_DIM = 2
THREE_DIM = 3
MAX_GRAY_SCALE = 255
GRAY_SCALE = 1


# ---------- code ---------- #

def read_image(file_name, representation=GRAY_SCALE):
    """
    reads an image file and converts it into given representation.
    :param file_name: the filename of an image on disk.
    :param representation: representation code, either 1 or 2 defining whether
    should be a grayscale image (1) or an RGB image (2).
    :return: an image with intensities normalized to the range [0,1]
    """
    im = np.array(imread(file_name))
    img_float = im.astype(np.float32) / MAX_GRAY_SCALE
    if representation == 1:  # return grayscale image
        if len(im.shape) == TWO_DIM:  # image was given in grayscale
            return img_float
        elif len(im.shape) == THREE_DIM:  # image is rgb, convert to grayscale
            return rgb2gray(img_float)
    elif representation == 2:  # return rgb
        return img_float


def filter_image(img):
    """
    clears noise from given image using bilateral Filter.
    :param img: image to filter, assumes its of type 32f
    :return: the filtered image
    """
    return cv2.bilateralFilter(img, 9, 50, 50)


if __name__ == '__main__':
    


