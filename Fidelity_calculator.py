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
THRESHOLD = 210


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


def threshold_image(img, threshold=THRESHOLD):
    """
    thresholds a grayscale image to a binary image.
    :param threshold: default is 210 by trial and error.
     assumes image is not compatible with otsu's binarization.
    :param img: 2D numpy array of type np.float32
    :return: a tuple (threshold, binary image)
    """
    return cv2.threshold(img*MAX_GRAY_SCALE, threshold, MAX_GRAY_SCALE, cv2.THRESH_BINARY)

def find_contours(img):
    pass



if __name__ == '__main__':
    img = read_image("Cat_after.png")
    blurr = filter_image(img)
    ret, thresh = threshold_image(blurr)
    plt.subplot(121), plt.imshow(blurr, cmap="gray"), plt.title("Blurred")
    plt.subplot(122), plt.imshow(thresh, cmap="gray"), plt.title("Binary")
    plt.show()
