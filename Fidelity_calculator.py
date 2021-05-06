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
NO_PARENT = -1
MIN_IMG_SIZE = 30 * 30


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
    img_float = im.astype(np.float32)
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
    return cv2.threshold(img, threshold, MAX_GRAY_SCALE, cv2.THRESH_BINARY)


def find_contours(thresh):
    """
    find contours in image, filters external (not 100%!!)
    :param thresh: binary image as np.array of type np.float32
    :return: returns a list of contours, each
    """
    thresh = thresh.astype(np.uint8)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # filter outer contours
    filtered_cont = []
    for i in range(len(contours)):
        if hierarchy[0, i, 3] == NO_PARENT:
            filtered_cont.append(contours[i])

    return filtered_cont


def mark_contours(contour_arr, img):
    """
    marks the contours on the image and crops them.
    :return: python list containing cropped images.
    """
    marg = 10
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    for contour in contour_arr:
        lower_dim = contour[:, 0]
        x, y = lower_dim[:, 0], lower_dim[:, 1]
        min_x, min_y, max_x, max_y = min(x), min(y), max(x), max(y)

        # remove small noise
        if (max_x - min_x) * (max_y - min_y) < MIN_IMG_SIZE:
            continue
        # avoid index error
        if min_x - marg < 0 or min_y - marg < 0 or max_y + marg > img.shape[1] or max_x + marg > img.shape[0]:
            marg = 0
        ax.plot([min_x - marg, max_x + marg, max_x + marg, min_x - marg, min_x - marg],
                [min_y - marg, min_y - marg, max_y + marg, max_y + marg, min_y - marg], c='r', linewidth=0.5)
    plt.show()


if __name__ == '__main__':
    img = read_image("Cat_after.png")
    ret, thresh = threshold_image(filter_image(img))
    contours = find_contours(thresh)
    mark_contours(contours, np.copy(img))
