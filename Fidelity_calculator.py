# ---------- imports ---------- #
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from imageio import imread, imwrite
from skimage.metrics import structural_similarity as ssim
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


def mark_contours(contour_arr, img, _plot=False):
    """
    marks the contours on the image and crops them.
    :return: python list containing tuples of cropped image and its contour.
    """
    marg = 20
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    sub_images = []  # init array for pictures
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
        # crop only half cause image is symmetric:
        if max_x <= img.shape[0] // 2:
            sub_images.append(img[min_y:max_y, min_x:max_x])
        if _plot:
            ax.plot([min_x - marg, max_x + marg, max_x + marg, min_x - marg, min_x - marg],
                    [min_y - marg, min_y - marg, max_y + marg, max_y + marg, min_y - marg], c='r', linewidth=0.5)
    if _plot:
        plt.savefig("found_subshapes")
        plt.show()
    return sub_images


def rotate_images():
    pass


def resize_image(img, width, height):
    """
    resizes the given image to size width X height, doesnt edit the original
    image.
    :param width: integer representing new width
    :param height: integer representing new height
    :param img: np.array represnting the image to resize
    :return: a resized copy of img.
    """

    return cv2.resize(img, (width, height))


def compare_img(img1, img2, err_function="ALL"):
    """
    Receives two images to compare, img1 being the original. and a string indictating
    which error function to use. doesnt assume images are the same size.
    :param err_function: string indicating which comparison func to use, supports:
    (1) "ALL" - apply all functions. (2) "MSE" - apply MSE err function. (3) "SSIM" - apply structural similarity comparison
    :param img1: np.array of type float32.
    :param img2: np.array of type float32.
    :return: np array containing the errors, if "ALL" is used then array[0]=MSE and array[1] is SSIM and array[2] is L1
    else its a singleton of chosen function.
    """

    # make sure images are the same shape #
    height1, width1, height2, width2 = img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1]
    if img1.shape != img2.shape:
        if width1 * height1 > width2 * height2:
            img1 = resize_image(img1, width2, height2)
        else:
            img2 = resize_image(img2, width1, height1)
    # TODO: create better resize to avoid interpolation when possible
    # compare images#
    func_arr = [mse, ssim, L1_norm]
    err_arr = []
    for func in func_arr:
        if err_function == "ALL" or func.__name__.upper() == err_function:
            err_arr.append(func(img1, img2))
    return np.array(err_arr)


def mse(img1, img2):
    """
    calculates the mean squared diffrence between two given images. assumes
    the images have the same size image
    :param img1:
    :param img2:
    :return:
    """
    err = (np.square(img1 - img2)).mean(axis=None)
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def L1_norm(img1, img2):
    """

    :param img1:
    :param img2:
    :return:
    """
    flattened1 = np.ravel(img1)
    flattened2 = np.ravel(img2)
    return LA.norm((flattened1 - flattened2), ord=1)
